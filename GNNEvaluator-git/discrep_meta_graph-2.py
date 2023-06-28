import torch
from torch_geometric.datasets import TUDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from dual_gnn.dataset.DomainData import DomainData
from torch_geometric.nn import GraphSAGE, GCN, GAT, GIN, MLP
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import itertools
import datetime
import sys
import logging
from scipy import linalg
from tensorboardX import SummaryWriter
import torch_geometric as tg
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from argparse import ArgumentParser


def load(args, device):
    dataset_s = DomainData("data/{}".format(args.source), name=args.source)
    source_data = dataset_s[0].to(device)
    logging.info(source_data)
    if args.model == 'GCN':
        encoder = GCN(source_data.num_node_features, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                      num_layers=args.num_layers).to(device)
    elif args.model == 'SAGE':
        encoder = GraphSAGE(source_data.num_node_features, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                            num_layers=args.num_layers).to(device)
    elif args.model == 'GAT':
        encoder = GAT(source_data.num_node_features, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                      num_layers=args.num_layers).to(device)
    elif args.model == 'GIN':
        encoder = GIN(source_data.num_node_features, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                      num_layers=args.num_layers).to(device)
    elif args.model == 'MLP':
        encoder = MLP(channel_list=[source_data.num_node_features, args.hid_dim, args.encoder_dim]).to(device)

    cls_model = nn.Sequential(nn.Linear(args.encoder_dim, dataset_s.num_classes), ).to(device)

    encoder.load_state_dict(torch.load(os.path.join(args.model_path, 'encoder.pt'), map_location=device))
    cls_model.load_state_dict(torch.load(os.path.join(args.model_path, 'cls_model.pt'), map_location=device))

    return encoder, cls_model, source_data

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """计算Gram核矩阵
    source: sample_size_1 * feature_size 的数据
    target: sample_size_2 * feature_size 的数据
    kernel_mul: 计算每个核的bandwith
    kernel_num: 表示的是多核的数量
    fix_sigma: 表示是否使用固定的标准差
        return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
                        矩阵，表达形式:
                        [   K_ss K_st
                            K_ts K_tt ]
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)  # 合并在一起

    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)  # 计算高斯核中的|x-y|

    # 计算多核中每个核的bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # 高斯核的公式，exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val)  # 将多个核合并在一起


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n = int(source.size()[0])
    m = int(target.size()[0])

    kernels = guassian_kernel(source.cpu(), target.cpu(),
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    kernels = kernels.cuda()
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)  # K_ss矩阵，Source<->Source
    XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st矩阵，Source<->Target

    YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)  # K_ts矩阵,Target<->Source
    YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)  # K_tt矩阵,Target<->Target

    loss = (XX + XY).sum() + (YX + YY).sum().cuda()
    return loss

def main(aug_data, encoder, cls_model, s_emb_train):
    models = [encoder, cls_model]
    for model in models:
        model.eval()
    aug_acc, aug_feat = test(aug_data, models, encoder, cls_model)
    dist_mmd = mmd(s_emb_train, aug_feat)

    train_feat = s_emb_train.detach()
    taraug_feat = aug_feat.detach()

    norm1 = torch.norm(train_feat, dim=-1).view(train_feat.shape[0], 1)
    norm2 = torch.norm(taraug_feat, dim=-1).view(1, taraug_feat.shape[0])

    end_norm = torch.matmul(norm1, norm2)
    cos_dist = torch.transpose(torch.matmul(train_feat, taraug_feat.t()) / end_norm, 0, 1)

    edge_dist = aug_data.edge_index

    new_sim_data = tg.data.Data(x=cos_dist, y=aug_acc, edge_index=edge_dist)

    return new_sim_data, dist_mmd, aug_acc


def test(data, models, encoder, cls_model, mask=None):
    for model in models:
        model.eval()

    if isinstance(encoder, MLP):
        emb_out = encoder(data.x)
    else:
        emb_out = encoder(data.x, data.edge_index)

    logits = F.softmax(cls_model(emb_out) if mask is None else cls_model(emb_out)[mask], dim=-1)
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    final_emb = emb_out if mask is None else emb_out[mask]
    return accuracy, final_emb


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, default='acm')
    parser.add_argument("--target", type=str, default='dblp')
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hid_dim", type=int, default=128)
    parser.add_argument("--encoder_dim", type=int, default=16)
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--test_rate", type=float, default=0.2)
    parser.add_argument("--num_metas", type=int, default=5)
    parser.add_argument("--interval", nargs='+', type=int, help='List of integers split for each aug method',
                        default=[100, 200, 300, 400])
    parser.add_argument("--model_path", type=str,
                        default='./logs/Models_tra/acm-to-dblp-GCN-full-0-0-20230402-230813-577875/')
    parser.add_argument("--aug_data_path", type=str,
                        default='./logs/MetaSet/Meta-save-acm-num-300-0-20230424-113801-947245/')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = './' + 'logs/MetaG/Meta-feat-acc-{}-{}-num-{}-{}-{}'.format(args.source, args.model,
                                                                          str(args.num_metas),
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

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    encoder, cls_model, source_data = load(args, device)
    encoder.eval()
    cls_model.eval()
    if isinstance(encoder, MLP):
        emb_source = encoder(source_data.x)
    else:
        emb_source = encoder(source_data.x, source_data.edge_index)
    s_train_mask = source_data.train_mask.to(torch.bool)
    s_emb_train = emb_source[s_train_mask, :]

    new_sim_data_ls = []
    idx_test_ls = []
    for i in range(0, len(args.interval)):
        if i == 0:
            idx_test_ls.append(
                random.sample(range(0, args.interval[i]), int(args.test_rate * (args.interval[i]))))
        else:
            idx_test_ls.append(
                random.sample(range(args.interval[i - 1], args.interval[i]),
                              int(args.test_rate * (args.interval[i]-args.interval[i - 1]))))

    idx_test_ls = np.concatenate(idx_test_ls, axis=None).tolist()
    idx_train_ls = [x for x in range(args.num_metas) if x not in idx_test_ls]
    logging.info('Loading meta sets ...')
    mmd_feat_ls = []
    mmd_acc_ls = []

    for idx in range(0, args.num_metas):
        aug_data = torch.load(os.path.join(args.aug_data_path, 'aug_data_' + str(idx) + '.pt'), map_location=device)
        new_sim_data, dist_mmd, aug_acc = main(aug_data, encoder, cls_model, s_emb_train)
        mmd_feat_ls.append(dist_mmd.detach().cpu().numpy())
        mmd_acc_ls.append(aug_acc.detach().cpu().numpy())
        if idx % 20 == 0:
            logging.info('Idx = {}, with {}'.format(idx, new_sim_data))
        new_sim_data_ls.append(new_sim_data.cpu())

    test_meta_data = [new_sim_data_ls[i] for i in idx_test_ls]
    train_meta_data = [new_sim_data_ls[j] for j in idx_train_ls]

    test_feat_mmd = [mmd_feat_ls[i] for i in idx_test_ls]
    train_feat_mmd= [mmd_feat_ls[j] for j in idx_train_ls]

    test_acc_mmd = [mmd_acc_ls[i] for i in idx_test_ls]
    train_acc_mmd = [mmd_acc_ls[j] for j in idx_train_ls]

    test_feat_mmd_np = np.array(test_feat_mmd)
    test_acc_mmd_np = np.array(test_acc_mmd).reshape(-1, 1)

    train_feat_mmd_np = np.array(train_feat_mmd)
    train_acc_mmd_np = np.array(train_acc_mmd).reshape(-1, 1)

    np.save(os.path.join(log_dir, 'train_mmd_feat.npy'), train_feat_mmd_np)
    np.save(os.path.join(log_dir, 'train_mmd_acc.npy'), train_acc_mmd_np)

    np.save(os.path.join(log_dir, 'test_mmd_feat.npy'), test_feat_mmd_np)
    np.save(os.path.join(log_dir, 'test_mmd_acc.npy'), test_acc_mmd_np)


    logging.info('this is the info of meta set: number of graphs in Train meta = {}, in Test meta = {}'.format(
        len(train_meta_data), len(test_meta_data)))
    import pickle

    # Assume `graph_data_list` is a list where each element is a graph data

    # Save the list to a file using pickle
    with open(os.path.join(log_dir, "test_meta_data.pickle"), "wb") as f1:
        pickle.dump(test_meta_data, f1)
    with open(os.path.join(log_dir, "train_meta_data.pickle"), "wb") as f2:
        pickle.dump(train_meta_data, f2)

    # Load the list from the file using pickle
    with open(os.path.join(log_dir, "test_meta_data.pickle"), "rb") as f1:
        loaded_test_meta_data = pickle.load(f1)
    with open(os.path.join(log_dir, "train_meta_data.pickle"), "rb") as f2:
        loaded_train_meta_data = pickle.load(f2)

    logging.info('Done!')
    logging.info(args)
    logging.info(log_dir)
