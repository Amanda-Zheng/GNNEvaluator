from argparse import ArgumentParser
import os
import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sys
import random
from dual_gnn.dataset.DomainData import DomainData
from torch_geometric.nn import GraphSAGE, GCN, GAT, GIN, MLP
from meta_feat_acc import *
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.init as init
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from tensorboardX import SummaryWriter


class RegNet2(nn.Module):
    def __init__(self,in_dim, dropout):
        super(RegNet2, self).__init__()
        self.fc3 = nn.Linear(in_dim, 1).apply(kaiming_init)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, f):
        z = torch.cat([x, y, f], dim=1)
        z = self.fc3(z)
        z = torch.sigmoid(z)
        return z


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def test(data, models, encoder, cls_model, mask=None):
    for model in models:
        model.eval()

    if isinstance(encoder, MLP):
        emb_out = encoder(data.x)
    else:
        emb_out = encoder(data.x, data.edge_index)

    logits = cls_model(emb_out) if mask is None else cls_model(emb_out)[mask]
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    final_emb = emb_out if mask is None else emb_out[mask]
    return accuracy, final_emb


def main(args, dist_s_tra_t_full, target_data_pe_mean, target_emb_feat_mu, real_test_acc,writer):
    # data preparation
    acc_np = np.load(os.path.join(args.load_path, 'meta_acc.npy'))
    acc = torch.from_numpy(acc_np).to(device)
    acc_tensor = Variable(acc, requires_grad=True).to(device)

    data_np = np.load(os.path.join(args.load_path, 'meta_feat.npy'))
    data = torch.from_numpy(data_np).to(device)
    data_tensor = Variable(data, requires_grad=True).to(device)

    data_pe_np = np.load(os.path.join(args.load_path, 'meta_pe_L_mean.npy'))
    data_pe = torch.from_numpy(data_pe_np).to(device)
    data_pe_tensor = Variable(data_pe, requires_grad=True).to(device)

    data_mu_np = np.load(os.path.join(args.load_path, 'meat_emb_feat.npy'))
    data_mu = torch.mean(torch.from_numpy(data_mu_np), dim=1).to(device)
    data_mu_tensor = Variable(data_mu, requires_grad=True).to(device)

    indice = args.val_num
    train_data = data_tensor[indice:]
    train_data_pe = data_pe_tensor[indice:]
    train_data_mu = data_mu_tensor[indice:]
    train_acc = acc_tensor[indice:]

    test_data = data_tensor[:indice]
    test_data_pe = data_pe_tensor[:indice]
    test_data_mu = data_mu_tensor[:indice]
    test_acc = acc_tensor[:indice]

    in_dim = train_data.shape[1]+train_data_pe.shape[1]+train_data_mu.shape[1]
    reg_model_nn = RegNet2(in_dim=in_dim, dropout=args.dropout_reg).to(device)
    optimizer_nn = optim.Adam(reg_model_nn.parameters(), lr=args.lr_reg)
    lossfunc = nn.MSELoss()
    dist_s_tra_t_full = torch.as_tensor(dist_s_tra_t_full, dtype=torch.float).reshape(1, -1)
    best_iter=0
    best_loss=1
    best_train = 0
    for epoch in range(args.epochs_reg):
        reg_model_nn.train()
        optimizer_nn.zero_grad()
        output2 = reg_model_nn.forward(train_data, train_data_pe, train_data_mu)
        loss = lossfunc(output2, train_acc)
        loss.backward()
        optimizer_nn.step()
        writer.add_scalar('train_loss', loss.item(), epoch)
        #for parameters in reg_model_nn.parameters():
        #    logging.info(parameters)
        logging.info('Epoch: {}, Training Loss = {:.4f}'.format(epoch, loss.item()))
        if epoch % 1 == 0:
            reg_model_nn.eval()
            output_test = reg_model_nn.forward(test_data, test_data_pe, test_data_mu)
            loss_test = F.mse_loss(output_test, test_acc)

            R2 = r2_score(test_acc.detach().cpu().numpy(), output_test.detach().cpu().numpy())
            RMSE = mean_squared_error(test_acc.detach().cpu().numpy(), output_test.detach().cpu().numpy(), squared=False)
            MAE = mean_absolute_error(test_acc.detach().cpu().numpy(), output_test.detach().cpu().numpy())
            output_target = reg_model_nn(dist_s_tra_t_full, target_data_pe_mean.reshape(1, -1),
                                         target_emb_feat_mu.reshape(1, -1))
            writer.add_scalar('loss_test', loss_test.item(), epoch)
            writer.add_scalar('R2', R2, epoch)
            writer.add_scalar('RMSE', RMSE, epoch)
            writer.add_scalar('MAE', MAE, epoch)
            writer.add_scalar('output_target', output_target.item(), epoch)

            logging.info('Epoch:{}, loss_test = {}, R2 = {}, RMSE = {}, MAE = {}, predict target = {} vs. real target = {}'.format(epoch, loss_test.item(), R2, RMSE, MAE, output_target.item(), real_test_acc))
            #if loss_test<best_loss:
            #    best_loss=loss_test
            #    best_iter=epoch
            #    output_target = reg_model_nn(dist_s_tra_t_full, target_data_pe_mean.reshape(1, -1),target_emb_feat_mu.reshape(1, -1))
                #print(loss_test.item(), output_target.item())

        #logging.info('BEST Epoch: {}, Test Loss = {:.4f}, Target predict = {} vs Real Target = {}'.format(best_iter,
        #                                                                                                  best_loss,
        #                                                                                                  output_target.item(),
        #                                                                                                  real_test_acc))


def load(args, device):
    dataset_s = DomainData("data/{}".format(args.source), name=args.source)
    source_data = dataset_s[0]
    logging.info(source_data)
    source_data = source_data.to(device)

    dataset_t = DomainData("data/{}".format(args.target), name=args.target)
    target_data = dataset_t[0]
    logging.info(target_data)
    target_data = target_data.to(device)
    target_data_pe_mean = np.load(
        os.path.join("data/{}".format(args.target), 'target_data_pe_mean_np_k_' + str(args.k_laplacian) + '_.npy'))
    target_data_pe_mean = torch.as_tensor(target_data_pe_mean, dtype=torch.float).to(device)
    # target_data_pe = laplacian_pe(target_data, k=args.k_laplacian)
    # target_data_pe_mean = torch.mean(target_data_pe,dim=0)
    # target_data_pe_mean_np  = np.array(target_data_pe_mean.detach().cpu().numpy())
    # np.save(os.path.join("data/{}".format(args.target), 'target_data_pe_mean_np_k_'+str(args.k_laplacian)+'_.npy'), target_data_pe_mean_np)
    # assert False
    if args.model == 'GCN':
        encoder = GCN(source_data.num_node_features, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                      num_layers=2).to(device)
    elif args.model == 'SAGE':
        encoder = GraphSAGE(source_data.num_node_features, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                            num_layers=2).to(device)
    elif args.model == 'GAT':
        encoder = GAT(source_data.num_node_features, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                      num_layers=2).to(device)
    elif args.model == 'GIN':
        encoder = GIN(source_data.num_node_features, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                      num_layers=2).to(device)
    elif args.model == 'MLP':
        encoder = MLP(channel_list=[source_data.num_node_features,args.hid_dim, args.encoder_dim]).to(device)

    cls_model = nn.Sequential(nn.Linear(args.encoder_dim, dataset_s.num_classes), ).to(device)

    encoder.load_state_dict(torch.load(os.path.join(args.model_path, 'encoder.pt'), map_location=device))
    cls_model.load_state_dict(torch.load(os.path.join(args.model_path, 'cls_model.pt'), map_location=device))
    encoder.eval()
    cls_model.eval()
    return encoder, cls_model, dataset_s, source_data, dataset_t, target_data, target_data_pe_mean


def real_target_test(source_data, target_data, encoder, cls_model):
    models = [encoder, cls_model]
    for model in models:
        model.eval()
    real_test_acc, t_emb_feat = test(target_data, models, encoder, cls_model)
    encoder.eval()
    cls_model.eval()

    if isinstance(encoder, MLP):
        encoded_source = encoder(source_data.x)
    else:
        encoded_source = encoder(source_data.x, source_data.edge_index)

    s_train_mask = source_data.train_mask.to(torch.bool)
    s_emb_train = encoded_source[s_train_mask, :]
    dist_s_tra_t_full = mmd(s_emb_train, t_emb_feat)
    t_emb_feat_mu = torch.mean(t_emb_feat, dim=0)
    return dist_s_tra_t_full, t_emb_feat_mu, real_test_acc


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, default='acm')
    parser.add_argument("--target", type=str, default='dblp')
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--hid_dim", type=int, default=128)
    parser.add_argument("--encoder_dim", type=int, default=16)
    parser.add_argument("--reg_model", type=str, default='mlp_simple')
    parser.add_argument("--dropout_reg", type=float, default=0.2)
    parser.add_argument("--lr_reg", type=float, default=0.5)
    parser.add_argument("--epochs_reg", type=int, default=200)
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='Learning rate step gamma (default: 0.8)')
    parser.add_argument("--k_laplacian", type=int, default=2)
    parser.add_argument("--walk_length", type=int, default=10)
    parser.add_argument("--val_num", type=int, default=30, help='number of samples for validation in LR')
    parser.add_argument("--model_path", type=str, default='./logs/acm-to-dblp-GCN-full-0-0-20221229-111920-594558/')
    parser.add_argument("--load_path", type=str, default='./logs/Meta-feat-acc-acm-GCN-num-10-0-20230110-154700-625677')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = './' + 'logs/metaLR-{}-to-{}-{}-{}-{}'.format(args.source, args.target, args.model,
                                                            str(args.seed),
                                                            datetime.datetime.now().strftime(
                                                                "%Y%m%d-%H%M%S-%f"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'test.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info('This is the log_dir: {}'.format(log_dir))
    writer = SummaryWriter(log_dir + '/tbx_log')
    logging.info(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    encoder, cls_model, dataset_s, source_data, dataset_t, target_data, target_data_pe_mean = load(args, device)
    dist_s_tra_t_full, target_emb_feat_mu, real_test_acc = real_target_test(source_data, target_data, encoder,
                                                                                cls_model)
    main(args, dist_s_tra_t_full, target_data_pe_mean, target_emb_feat_mu, real_test_acc,writer)
    logging.info(args)
    logging.info('Finish, this is the log dir = {}'.format(log_dir))
