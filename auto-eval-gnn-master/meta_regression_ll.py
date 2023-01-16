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


class RegNet(nn.Module):
    def __init__(self, input_dim, pe_input_dim, hid_dim, dropout):
        super(RegNet, self).__init__()
        self.weight_pe = nn.Linear(pe_input_dim, 1).apply(kaiming_init)
        self.fc1 = nn.Linear(input_dim, hid_dim).apply(kaiming_init)
        self.fc2 = nn.Linear(hid_dim, 1).apply(kaiming_init)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_feat, x_pe):
        x_pe_weight = self.weight_pe(x_pe)
        x_pe_out = x_pe_weight * x_pe

        z = torch.cat([x_feat, x_pe_out], dim=1)
        z = self.fc1(z)
        z = F.relu(z)
        z = self.dropout(z)
        z = self.fc2(z)
        output = z.view(-1)
        return output

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
    emb_out = encoder(data.x, data.edge_index)
    logits = cls_model(emb_out) if mask is None else cls_model(emb_out)[mask]
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    final_emb = emb_out if mask is None else emb_out[mask]
    return accuracy, final_emb


def main(args, dist_s_tra_t_full, target_data_pe_mean, target_emb_feat_mu, real_test_acc):
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

    slr = LinearRegression()
    slr.fit(train_data, train_acc)
    test_pred = slr.predict(test_data)
    logging.info('Linear regression test acc = {}'.format(test_pred))
    dist_s_tra_t_full = dist_s_tra_t_full.reshape(-1, 1).detach().cpu().numpy()
    real_test_pred = slr.predict(dist_s_tra_t_full)

    logging.info('Compare: LR target acc = {} vs. Real target acc = {}'.format(real_test_pred, real_test_acc))

    # plot training dataset
    plt.scatter(train_data, train_acc, color='#0000FF')
    plt.plot(train_data, slr.predict(train_data), color='#FF0000')

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.savefig(os.path.join(log_dir, 'linear_regression_train.png'))
    plt.close()

    # plot testing dataset
    plt.scatter(test_data, test_acc, color='red')
    plt.plot(test_data, slr.predict(test_data), color='blue')
    plt.savefig(os.path.join(log_dir, 'linear_regression_test.png'))

    logging.info(
        'If you could observe the linear correlation from figures, then your implementations are all good!')

    # evaluation with metrics
    logging.info('Test on Validation Set..')
    R2 = r2_score(test_acc, slr.predict(test_data))
    RMSE = mean_squared_error(test_acc, slr.predict(test_data), squared=False)
    MAE = mean_absolute_error(test_acc, slr.predict(test_data))
    logging.info('\nTest set: R2 :{:.4f} RMSE: {:.4f} MAE: {:.4f}\n'.format(R2, RMSE, MAE))

    # analyze the statistical correlation
    rho, pval = stats.spearmanr(test_data, test_acc)
    logging.info('\nRank correlation-rho ={}'.format(rho))
    logging.info('Rank correlation-pval = {}'.format(pval))

    rho, pval = stats.pearsonr(test_data.reshape(-1), test_acc.reshape(-1))
    logging.info('\nPearsons correlation-rho = {}'.format(rho))
    logging.info('Pearsons correlation-pval = {}'.format(pval))


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
        encoder = MLP(source_data.num_node_features, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                      num_layers=2).to(device)

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
    logging.info(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    encoder, cls_model, dataset_s, source_data, dataset_t, target_data, target_data_pe_mean = load(args, device)
    dist_s_tra_t_full, target_emb_feat_mu, real_test_acc = real_target_test(source_data, target_data, encoder,
                                                                            cls_model)
    main(args, dist_s_tra_t_full, target_data_pe_mean, target_emb_feat_mu, real_test_acc)
    logging.info(args)
    logging.info('Finish, this is the log dir = {}'.format(log_dir))
