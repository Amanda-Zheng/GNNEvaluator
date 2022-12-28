# coding=utf-8
import copy
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from argparse import ArgumentParser
from dual_gnn.cached_gcn_conv import CachedGCNConv
from dual_gnn.dataset.DomainData import DomainData
from dual_gnn.ppmi_conv import PPMIConv
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


def main(args, device, encoder, cls_model, dataset_s, source_data):
    if args.aug_method == 'edge_drop':
        aug_edge_drop_all = EdgeDrop_all(p=args.edge_drop_all_p)
        aug_edge_data = aug_edge_drop_all(source_data).to(device)
        aug_data = aug_edge_data
    elif args.aug_method == 'node_mix':
        aug_node_mix_all = NodeMixUp_all(lamb=args.mix_lamb, num_classes=dataset_s.num_classes)
        aug_node_mix_data = aug_node_mix_all(source_data).to(device)
        aug_data = aug_node_mix_data
    elif args.aug_method == 'node_fmask':
        aug_node_fmask_all = NodeFeatureMasking_all(p=args.node_fmask_all_p)
        aug_node_fmask_data = aug_node_fmask_all(source_data).to(device)
        aug_data = aug_node_fmask_data
    elif args.aug_method == 'combo':
        aug_edge_drop_all = EdgeDrop_all(p=args.edge_drop_all_p)
        aug_node_mix_all = NodeMixUp_all(lamb=args.mix_lamb, num_classes=dataset_s.num_classes)
        aug_node_fmask_all = NodeFeatureMasking_all(p=args.node_fmask_all_p)
        aug_edge_data_1 = aug_edge_drop_all(source_data).to(device)
        aug_node_mix_data_2 = aug_node_mix_all(aug_edge_data_1).to(device)
        aug_node_fmask_data_3 = aug_node_fmask_all(aug_node_mix_data_2).to(device)
        aug_data = aug_node_fmask_data_3

    source_data = source_data.to(device)

    models = [encoder, cls_model]

    aug_acc, aug_feat = test(aug_data, models, encoder, cls_model, "aug_data", aug_data.val_mask.to(torch.bool))
    encoded_source = encode(source_data, encoder, "source")
    s_train_mask = source_data.train_mask.to(torch.bool)
    s_emb_train = encoded_source[s_train_mask, :]
    dist_s_tra_aug_val = mmd(s_emb_train, aug_feat)
    # print(type(aug_acc), type(dist_s_tra_aug_val))
    return dist_s_tra_aug_val, aug_acc


class GNN(torch.nn.Module):
    def __init__(self, base_model=None, type="gcn", num_features=None, encoder_dim=None, **kwargs):
        super(GNN, self).__init__()

        if base_model is None:
            weights = [None, None]
            biases = [None, None]
        else:
            weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias for conv_layer in base_model.conv_layers]

        self.dropout_layers = [nn.Dropout(0.1) for _ in weights]
        self.type = type

        model_cls = PPMIConv if type == "ppmi" else CachedGCNConv

        self.conv_layers = nn.ModuleList([
            model_cls(num_features, 128,
                      weight=weights[0],
                      bias=biases[0],
                      **kwargs),
            model_cls(128, encoder_dim,
                      weight=weights[1],
                      bias=biases[1],
                      **kwargs)
        ])

    def forward(self, x, edge_index, cache_name):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, cache_name)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[i](x)
        return x


def gcn_encode(data, encoder, cache_name, mask=None):
    encoded_output = encoder(data.x, data.edge_index, cache_name)
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output


def sage_encode(data, encoder, mask=None):
    encoded_output = encoder(data.x, data.edge_index)
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output


def encode(data, encoder, cache_name, mask=None):
    if args.model == 'GCN':
        gcn_output = gcn_encode(data, encoder, cache_name, mask)
        output = gcn_output
    elif args.model == 'SAGE':
        sage_output = sage_encode(data, encoder, mask)
        output = sage_output
    return output


def predict(data, encoder, cls_model, cache_name, mask=None):
    encoded_output = encode(data, encoder, cache_name, mask)
    logits = cls_model(encoded_output)
    return logits, encoded_output


def evaluate(preds, labels):
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    return accuracy


def test(data, models, encoder, cls_model, cache_name, mask=None):
    for model in models:
        model.eval()
    logits, encoded_output = predict(data, encoder, cls_model, cache_name, mask)
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    accuracy = evaluate(preds, labels)
    return accuracy, encoded_output


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


def train(epoch, models, encoder, cls_model, optimizer, loss_func, source_data, target_data, meta_data):
    s_train_mask = source_data.train_mask.to(torch.bool)
    s_val_mask = source_data.val_mask.to(torch.bool)
    s_test_mask = source_data.test_mask.to(torch.bool)

    aug_train_mask = meta_data.train_mask.to(torch.bool)
    aug_val_mask = meta_data.val_mask.to(torch.bool)
    aug_test_mask = meta_data.test_mask.to(torch.bool)

    for model in models:
        model.train()

    encoded_source = encode(source_data, encoder, "source")
    encoded_target = encode(target_data, encoder, "target")
    encoded_aug = encode(meta_data, encoder, "meta")

    s_emb_train = encoded_source[s_train_mask, :]
    s_emb_val = encoded_source[s_val_mask, :]
    s_emb_test = encoded_source[s_test_mask, :]

    aug_emb_val = encoded_aug[aug_val_mask, :]

    t_emb_full = encoded_target

    source_logits = cls_model(encoded_source)

    if args.full_s == 1:
        cls_loss = loss_func(source_logits, source_data.y)
    else:
        cls_loss = loss_func(source_logits[s_train_mask], source_data.y[s_train_mask])

    for model in models:
        for name, param in model.named_parameters():
            if "weight" in name:
                cls_loss = cls_loss + param.mean() * 3e-3

    loss = cls_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return s_emb_train, s_emb_val, aug_emb_val, t_emb_full


def load(args,device):
    dataset_s = DomainData("data/{}".format(args.source), name=args.source)
    source_data = dataset_s[0]
    logging.info(source_data)
    if args.model == 'GCN':
        encoder = GNN(type="gcn", num_features=dataset_s.num_features, encoder_dim=args.encoder_dim).to(device)
    elif args.model == 'SAGE':
        from torch_geometric.nn import GraphSAGE
        encoder = GraphSAGE(source_data.num_node_features, hidden_channels=args.encoder_dim, num_layers=2).to(device)

    cls_model = nn.Sequential(nn.Linear(args.encoder_dim, dataset_s.num_classes), ).to(device)

    encoder.load_state_dict(torch.load(os.path.join(args.save_path, 'encoder.pt'), map_location=device))
    cls_model.load_state_dict(torch.load(os.path.join(args.save_path, 'cls_model.pt'), map_location=device))

    return encoder, cls_model, dataset_s, source_data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, default='acm')
    parser.add_argument("--name", type=str, default='UDAGCN')
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--UDAGCN", type=bool, default=False)
    parser.add_argument("--encoder_dim", type=int, default=16)
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--full_s", type=int, default=1)
    parser.add_argument("--num_metas", type=int, default=5)
    parser.add_argument("--save_path", type=str, default='./logs/acm-GCN-full-0-0-20221228-153442-415490/checkpoints/')
    parser.add_argument("--aug_method", type=str, default='edge_drop',
                        choices=['edge_drop', 'node_mix', 'node_fmask', 'combo'], \
                        help='method for augment data for creating the meta dataset')
    parser.add_argument("--node_drop_val_p", type=float, default=0.05)
    parser.add_argument("--edge_drop_all_p", type=float, default=0.05)
    parser.add_argument("--node_fmask_all_p", type=float, default=0.05)
    parser.add_argument("--mix_lamb", type=float, default=0.3, help='rational of mix features of nodes, ranges=[0,1]')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = './' + 'logs/{}-{}-full-{}-{}-{}'.format(args.source, args.model,
                                                              str(args.full_s),
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
    id = "source: {}, seed: {}, UDAGCN: {}, encoder_dim: {}" \
        .format(args.source, args.seed, args.UDAGCN, args.encoder_dim)

    # print(id)
    logging.info(id)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    encoder, cls_model, dataset_s, source_data = load(args, device)
    meta_feat_ls = []
    meta_acc_ls = []
    for i in range(0, args.num_metas):
        # args.node_drop_val_p = np.random.random()
        method_code = np.random.randint(0,4)
        if method_code ==0:
            args.aug_method = 'edge_drop'
        elif method_code ==1:
            args.aug_method ='node_mix'
        elif method_code ==2:
            args.aug_method = 'node_fmask'
        elif method_code ==3:
            args.aug_method = 'combo'
        args.edge_drop_all_p = np.random.random()
        args.node_fmask_all_p = np.random.random()
        args.mix_lamb = np.random.random()
        logging.info('This is the rate of trans with "{}"-method: edge_drop_rate = {}, node_fmask_rate = {}, node_mix_rate = {}' \
                     .format(args.aug_method, args.edge_drop_all_p, args.node_fmask_all_p, args.mix_lamb))
        dist_s_tra_aug_val, aug_acc = main(args, device, encoder, cls_model, dataset_s, source_data)
        logging.info('FEAT = {}, ACC = {}'.format(dist_s_tra_aug_val, aug_acc))
        meta_feat_ls.append(dist_s_tra_aug_val.detach().cpu().numpy())
        meta_acc_ls.append(aug_acc.detach().cpu().numpy())

    meta_feat_np = np.array(meta_feat_ls).reshape(-1,1)
    meta_acc_np = np.array(meta_acc_ls).reshape(-1,1)
    logging.info('The size of meta data: feat = {}, acc = {}'.format(meta_feat_np.shape, meta_acc_np.shape))

    np.save(os.path.join(log_dir,'meta_feat.npy'), meta_feat_np)
    np.save(os.path.join(log_dir,'meta_acc.npy'), meta_acc_np)
    logging.info(args)
    logging.info('Finish, this is the log dir = {}'.format(log_dir))
