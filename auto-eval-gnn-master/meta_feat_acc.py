# coding=utf-8
import copy
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from argparse import ArgumentParser
from dual_gnn.dataset.DomainData import DomainData
from torch_geometric.nn import GraphSAGE, GCN, GAT, GIN, MLP
from meta_data_create import EdgeDrop_all, NodeDrop_val, NodeMixUp_all, NodeFeatureMasking_all
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
import grafog.transforms as T
import math
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
)
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch_scatter import scatter

from torch_geometric.typing import OptTensor

from torch_geometric.utils.num_nodes import maybe_num_nodes


# from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE
def laplacian_pe(data, k):
    from scipy.sparse.linalg import eigs
    num_nodes = data.num_nodes
    edge_index, edge_weight = get_laplacian(
        data.edge_index,
        data.edge_weight,
        normalization='sym',
        num_nodes=num_nodes,
    )
    L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

    eig_vals, eig_vecs = eigs(
        L,
        k=k + 1,
        which='SR',
        return_eigenvectors=True
    )

    eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
    pe = torch.from_numpy(eig_vecs[:, 1:k + 1])
    sign = -1 + 2 * torch.randint(0, 2, (k,))
    pe *= sign

    return pe


def random_walk_pe(data, walk_length):
    from torch_sparse import SparseTensor

    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight

    adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                       sparse_sizes=(num_nodes, num_nodes))

    # Compute D^{-1} A:
    deg_inv = 1.0 / adj.sum(dim=1)
    deg_inv[deg_inv == float('inf')] = 0
    adj = adj * deg_inv.view(-1, 1)

    out = adj
    row, col, value = out.coo()
    pe_list = [get_self_loop_attr((row, col), value, num_nodes)]
    for _ in range(walk_length - 1):
        out = out @ adj
        row, col, value = out.coo()
        pe_list.append(get_self_loop_attr((row, col), value, num_nodes))
    pe = torch.stack(pe_list, dim=-1)
    return pe


def get_self_loop_attr(edge_index: Tensor, edge_attr: OptTensor = None,
                       num_nodes: Optional[int] = None) -> Tensor:
    loop_mask = edge_index[0] == edge_index[1]
    loop_index = edge_index[0][loop_mask]

    if edge_attr is not None:
        loop_attr = edge_attr[loop_mask]
    else:  # A vector of ones:
        loop_attr = torch.ones_like(loop_index, dtype=torch.float)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    full_loop_attr = loop_attr.new_zeros((num_nodes,) + loop_attr.size()[1:])
    full_loop_attr[loop_index] = loop_attr

    return full_loop_attr


def main(args, device, encoder, cls_model, source_data, idx):
    aug_data = torch.load(os.path.join(args.aug_data_path, 'aug_data_' + str(idx) + '.pt'), map_location=device)
    aug_data_L_pe = laplacian_pe(aug_data, k=args.k_laplacian)[aug_data.val_mask.to(torch.bool)]
    mean_aug_data_L_pe = torch.mean(aug_data_L_pe, dim=0)
    #aug_data_rw_pe = random_walk_pe(aug_data.to(cpu), walk_length=args.walk_length)[aug_data.val_mask.to(torch.bool)].cuda()
    #mean_aug_data_rw_pe = torch.mean(aug_data_rw_pe, dim=0)
    aug_data_rw_pe = mean_aug_data_rw_pe = 0
    source_data = source_data.to(device)

    models = [encoder, cls_model]
    for model in models:
        model.eval()
    aug_acc, aug_feat = test(aug_data, models, encoder, cls_model, mask=aug_data.val_mask.to(torch.bool))

    if isinstance(encoder, MLP):
        emb_source = encoder(source_data.x)
    else:
        emb_source = encoder(source_data.x, source_data.edge_index)

    s_train_mask = source_data.train_mask.to(torch.bool)
    s_emb_train = emb_source[s_train_mask, :]

    dist_s_tra_aug_val = mmd(s_emb_train, aug_feat)

    #lcka_score = linear_CKA(torch.t(s_emb_train).cpu().detach().numpy(), torch.t(aug_feat).cpu().detach().numpy())
    #kcka_score = kernel_CKA(torch.t(s_emb_train).cpu().detach().numpy(), torch.t(aug_feat).cpu().detach().numpy())
    lcka_score =kcka_score=0
    return dist_s_tra_aug_val, lcka_score, kcka_score, aug_data_L_pe, mean_aug_data_L_pe, aug_data_rw_pe, mean_aug_data_rw_pe, aug_feat, aug_acc


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K),
                  H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)


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


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """计算Gram核矩阵
    source: sample_size_1 * feature_size 的数据
    target: sample_size_2 * feature_size 的数据
    kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
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

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)  # K_ss矩阵，Source<->Source
    XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st矩阵，Source<->Target

    YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)  # K_ts矩阵,Target<->Source
    YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)  # K_tt矩阵,Target<->Target

    loss = (XX + XY).sum() + (YX + YY).sum()
    return loss


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def load(args, device):
    dataset_s = DomainData("data/{}".format(args.source), name=args.source)
    source_data = dataset_s[0]
    logging.info(source_data)
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
        encoder = MLP(channel_list=[source_data.num_node_features, args.hid_dim, args.encoder_dim]).to(device)

    cls_model = nn.Sequential(nn.Linear(args.encoder_dim, dataset_s.num_classes), ).to(device)

    encoder.load_state_dict(torch.load(os.path.join(args.model_path, 'encoder.pt'), map_location=device))
    cls_model.load_state_dict(torch.load(os.path.join(args.model_path, 'cls_model.pt'), map_location=device))

    return encoder, cls_model, source_data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, default='acm')
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--hid_dim", type=int, default=128)
    parser.add_argument("--encoder_dim", type=int, default=16)
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--num_metas", type=int, default=5)
    parser.add_argument("--k_laplacian", type=int, default=5)
    parser.add_argument("--walk_length", type=int, default=10)
    parser.add_argument("--model_path", type=str, default='./logs/acm-to-dblp-GCN-full-0-0-20221229-111920-594558/')
    parser.add_argument("--aug_data_path", type=str, default='./logs/Meta-save-acm-num-300-0-20221229-142336-945310/')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = './' + 'logs/Meta-feat-acc-{}-{}-num-{}-{}-{}'.format(args.source, args.model,
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    encoder, cls_model, source_data = load(args, device)
    encoder.eval()
    cls_model.eval()
    meta_feat_ls = []
    meta_acc_ls = []
    meta_pe_L_ls = []
    meta_pe_L_mean_ls = []
    meta_pe_rw_ls = []
    meta_pe_rw_mean_ls = []
    meat_emb_feat_ls = []
    for idx in range(0, args.num_metas):
        dist_s_tra_aug_val, linear_cka_s, kernel_cka_s, \
        aug_data_L_pe, mean_aug_data_L_pe, aug_data_rw_pe, mean_aug_data_rw_pe, \
        meat_emb_feat, aug_acc = main(args, device, encoder, cls_model, source_data, idx)
        logging.info('FEAT = {}, linear_cka={}, kernel_cka_s = {}, ACC = {}'.format(dist_s_tra_aug_val, linear_cka_s,
                                                                                    kernel_cka_s, aug_acc))
        meta_feat_ls.append(dist_s_tra_aug_val.detach().cpu().numpy())
        meta_pe_L_ls.append(aug_data_L_pe.detach().cpu().numpy())
        meta_pe_L_mean_ls.append(mean_aug_data_L_pe.detach().cpu().numpy())
        #meta_pe_rw_ls.append(aug_data_rw_pe.detach().cpu().numpy())
        #meta_pe_rw_mean_ls.append(mean_aug_data_rw_pe.detach().cpu().numpy())
        meta_acc_ls.append(aug_acc.detach().cpu().numpy())
        meat_emb_feat_ls.append(meat_emb_feat.detach().cpu().numpy())

    meta_feat_np = np.array(meta_feat_ls).reshape(-1, 1)
    meta_acc_np = np.array(meta_acc_ls).reshape(-1, 1)
    meta_pe_L_np = np.array(meta_pe_L_ls)
    meta_pe_L_mean_np = np.array(meta_pe_L_mean_ls)
    meat_emb_feat_np = np.array(meat_emb_feat_ls)
    #meta_pe_rw_np = np.array(meta_pe_rw_ls)
    #meta_pe_rw_mean_np = np.array(meta_pe_rw_mean_ls)
    #logging.info('The size of meta data: feat = {}, '
    #             'pe_L = {}, mean_pe_L = {}, '
    #             'pe_rw = {}, mean_pe_rw = {}, acc = {}'.format(meta_feat_np.shape,\
    #                                                            meta_pe_L_np.shape, meta_pe_L_mean_np.shape,\
    #                                                            meta_pe_rw_np.shape, meta_pe_rw_mean_np.shape,\
    #                                                            meta_acc_np.shape))
    logging.info('The size of meta data: feat = {}, '
                 'pe_L = {}, mean_pe_L = {}, meta_emb = {}, acc = {}'.format(meta_feat_np.shape,\
                                                                meta_pe_L_np.shape, meta_pe_L_mean_np.shape,\
                                                                meat_emb_feat_np.shape, meta_acc_np.shape))

    np.save(os.path.join(log_dir, 'meta_feat.npy'), meta_feat_np)
    np.save(os.path.join(log_dir, 'meta_acc.npy'), meta_acc_np)
    np.save(os.path.join(log_dir, 'meta_pe_L.npy'), meta_pe_L_np)
    np.save(os.path.join(log_dir, 'meta_pe_L_mean.npy'), meta_pe_L_mean_np)
    np.save(os.path.join(log_dir, 'meat_emb_feat.npy'), meat_emb_feat_np)
    #np.save(os.path.join(log_dir, 'meta_pe_rw.npy'), meta_pe_rw_np)
    #np.save(os.path.join(log_dir, 'meta_pe_rw_mean_np.npy'), meta_pe_rw_mean_np)
    logging.info(args)
    logging.info('Finish, this is the log dir = {}'.format(log_dir))
