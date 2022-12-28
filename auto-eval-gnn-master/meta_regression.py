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
from torch_geometric.nn import GraphSAGE,GCN
from meta_train_test import mmd
import torch
import torch.nn.functional as F
from torch import nn



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


def main(args, dist_s_tra_t_full, real_test_acc):
    # data preparation
    acc = np.load(os.path.join(args.load_path, 'meta_acc.npy'))
    data = np.load(os.path.join(args.load_path, 'meta_feat.npy'))

    # Choose some sample sets as validation (also used in NN regression)
    indice = args.val_num
    train_data = data[indice:]
    train_acc = acc[indice:]
    test_data = train_data[:indice]
    test_acc = train_acc[:indice]

    # linear regression
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
    plt.savefig(os.path.join(log_dir,'linear_regression_train.png'))
    plt.close()

    # plot testing dataset
    plt.scatter(test_data, test_acc, color='red')
    plt.plot(test_data, slr.predict(test_data), color='blue')
    plt.savefig(os.path.join(log_dir,'linear_regression_test.png'))

    logging.info('If you could observe the linear correlation from figures, then your implementations are all good!')


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

    if args.model == 'GCN':
        encoder = GCN(source_data.num_node_features, hidden_channels=args.hid_dim, out_channels=args.encoder_dim, num_layers=2).to(device)
    elif args.model == 'SAGE':
        encoder = GraphSAGE(source_data.num_node_features, hidden_channels=args.hid_dim, out_channels=args.encoder_dim, num_layers=2).to(device)

    cls_model = nn.Sequential(nn.Linear(args.encoder_dim, dataset_s.num_classes), ).to(device)

    encoder.load_state_dict(torch.load(os.path.join(args.model_path, 'encoder.pt'), map_location=device))
    cls_model.load_state_dict(torch.load(os.path.join(args.model_path, 'cls_model.pt'), map_location=device))
    encoder.eval()
    cls_model.eval()
    return encoder, cls_model, dataset_s, source_data, dataset_t, target_data


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
    return dist_s_tra_t_full, real_test_acc


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, default='acm')
    parser.add_argument("--target", type=str, default='dblp')
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--hid_dim", type=int, default=128)
    parser.add_argument("--encoder_dim", type=int, default=16)
    parser.add_argument("--val_num", type=int, default=30, help='number of samples for validation in LR')
    parser.add_argument("--model_path", type=str, default='./logs/acm-GCN-full-0-0-20221228-153442-415490/checkpoints/')
    parser.add_argument("--load_path", type=str, default='./logs/acm-GCN-full-0-0-20221228-171336-369140/')

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
    with torch.no_grad():
        encoder, cls_model, dataset_s, source_data, dataset_t, target_data = load(args, device)
        dist_s_tra_t_full, real_test_acc = real_target_test(source_data, target_data, encoder, cls_model)
        main(args, dist_s_tra_t_full, real_test_acc)
    logging.info(args)
    logging.info('Finish, this is the log dir = {}'.format(log_dir))
