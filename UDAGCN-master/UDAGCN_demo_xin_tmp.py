# coding=utf-8
import copy
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from argparse import ArgumentParser
from dual_gnn.cached_gcn_conv import CachedGCNConv
from dual_gnn.dataset.DomainData import DomainData
from dual_gnn.ppmi_conv import PPMIConv
import random
import numpy as np
import torch
import torch.functional as F
from torch import nn
import itertools
import datetime
import sys
import logging
from scipy import linalg
from tensorboardX import SummaryWriter
import torch_geometric as tg
import grafog.transforms as T


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = ArgumentParser()
parser.add_argument("--source", type=str, default='acm')
parser.add_argument("--target", type=str, default='dblp')
parser.add_argument("--name", type=str, default='UDAGCN')
parser.add_argument("--seed", type=int, default=200)
parser.add_argument("--UDAGCN", type=bool, default=False)
parser.add_argument("--encoder_dim", type=int, default=16)
parser.add_argument("--lr", type=float, default=3e-3)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--model", type=str, default='GCN')
parser.add_argument("--full_s", type=int, default=1)
parser.add_argument("--node_drop_val_p", type=float, default=0.05)

args = parser.parse_args()
seed = args.seed
use_UDAGCN = args.UDAGCN
encoder_dim = args.encoder_dim
log_dir = './' + 'logs/{}-to-{}-{}-full-{}-{}-{}'.format(args.source, args.target, args.model, str(args.full_s),
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

id = "source: {}, target: {}, seed: {}, UDAGCN: {}, encoder_dim: {}" \
    .format(args.source, args.target, seed, use_UDAGCN, encoder_dim)

# print(id)
logging.info(id)

rate = 0.0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
dataset = DomainData("data/{}".format(args.source), name=args.source)
source_data = dataset[0]
logging.info(source_data)
dataset = DomainData("data/{}".format(args.target), name=args.target)
target_data = dataset[0]
logging.info(target_data)
s_train_mask = source_data.train_mask.to(torch.bool)
s_val_mask = source_data.val_mask.to(torch.bool)
s_test_mask = source_data.test_mask.to(torch.bool)

#s_train_mask_oh = copy.deepcopy(source_data.train_mask)
#idx = torch.empty(source_data.x.size(0)).uniform_(0, 1)
#print(s_train_mask_oh[torch.where(idx < args.node_drop_val_p)])
#s_train_mask_oh[torch.where(idx < args.node_drop_val_p)] = 0
#print(s_train_mask_oh[torch.where(idx < args.node_drop_val_p)])
#print(sum(s_train_mask_oh-source_data.train_mask))


class NodeDrop(nn.Module):
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p

    def forward(self, data):
        x = copy.deepcopy(data.x)
        y = copy.deepcopy(data.y)
        train_mask = copy.deepcopy(data.train_mask)
        test_mask = copy.deepcopy(data.test_mask)
        val_mask = copy.deepcopy(data.val_mask)
        edge_idx = copy.deepcopy(data.edge_index)
        idx = torch.empty(x.size(0)).uniform_(0, 1)
        print(torch.where(idx < self.p))
        print(sum(train_mask[torch.where(idx < self.p)]),sum(data.train_mask[torch.where(idx < self.p)]))
        train_mask[torch.where(idx < self.p)] = 0
        print('after', sum(train_mask[torch.where(idx < self.p)]), sum(data.train_mask[torch.where(idx < self.p)]))
        print(sum(test_mask[torch.where(idx < self.p)]))
        test_mask[torch.where(idx < self.p)] = 0
        print('after',sum(test_mask[torch.where(idx < self.p)]))
        print(sum(val_mask[torch.where(idx < self.p)]))
        val_mask[torch.where(idx < self.p)] = 0
        print('after',sum(val_mask[torch.where(idx < self.p)]))
        new_data = tg.data.Data(x=x, edge_index=edge_idx, y=y, train_mask=train_mask, val_mask = val_mask, test_mask=test_mask)

        return new_data
'''
class NodeDrop_val(nn.Module):
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p

    def forward(self, data):
        x = data.x
        val_mask = data.val_mask
        idx = torch.empty(x.size(0)).uniform_(0, 1)
        val_mask[torch.where(idx < self.p)] = 0
        return val_mask
'''
#node_aug = T.Compose([T.NodeDrop(p=0.45)])
#new_data = node_aug(source_data)
#diff_train_mask = source_data.train_mask - new_data.train_mask
#print(diff_train_mask)
aug_node_drop_full = NodeDrop(p=0.5)
aug_data = aug_node_drop_full(source_data)
diff_train_mask = sum(source_data.train_mask - aug_data.train_mask)
print('diff_train_mask',diff_train_mask)
#diff_val_mask = sum(source_data.val_mask - aug_data.val_mask)
#diff_test_mask = sum(source_data.test_mask - aug_data.test_mask)
#print(diff_train_mask, diff_val_mask, diff_test_mask)
#aug_node_drop_val = NodeDrop_val(p=args.node_drop_val_p)
#s_new_val_mask = aug_node_drop_val(source_data)
#diff_mask = sum(source_data.test_mask - s_new_val_mask)
# print(source_data.train_mask)
# print(s_train_mask)
# print(s_train_mask.dtype)

source_data = source_data.to(device)
target_data = target_data.to(device)


class GNN(torch.nn.Module):
    def __init__(self, base_model=None, type="gcn", **kwargs):
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
            model_cls(dataset.num_features, 128,
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


'''
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * rate
        return grad_output, None


class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)

'''
loss_func = nn.CrossEntropyLoss().to(device)

if args.model == 'GCN':
    encoder = GNN(type="gcn").to(device)
elif args.model == 'SAGE':
    from torch_geometric.nn import GraphSAGE

    encoder = GraphSAGE(source_data.num_node_features, hidden_channels=encoder_dim, num_layers=2).to(device)

# if use_UDAGCN:
#    ppmi_encoder = GNN(base_model=encoder, type="ppmi", path_len=10).to(device)


cls_model = nn.Sequential(
    nn.Linear(encoder_dim, dataset.num_classes),
).to(device)

'''
domain_model = nn.Sequential(
    GRL(),
    nn.Linear(encoder_dim, 40),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(40, 2),
).to(device)
'''

'''
class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)


    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = F.softmax(self.dense_weight(stacked), dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs


att_model = Attention(encoder_dim).cuda()
'''
# models = [encoder, cls_model, domain_model]
models = [encoder, cls_model]
# if use_UDAGCN:
#    models.extend([ppmi_encoder, att_model])
params = itertools.chain(*[model.parameters() for model in models])
optimizer = torch.optim.Adam(params, lr=args.lr)


def gcn_encode(data, cache_name, mask=None):
    encoded_output = encoder(data.x, data.edge_index, cache_name)
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output


def sage_encode(data, mask=None):
    encoded_output = encoder(data.x, data.edge_index)
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output


'''
def ppmi_encode(data, cache_name, mask=None):
    encoded_output = ppmi_encoder(data.x, data.edge_index, cache_name)
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output
'''


def encode(data, cache_name, mask=None):
    if args.model == 'GCN':
        gcn_output = gcn_encode(data, cache_name, mask)
        output = gcn_output
    elif args.model == 'SAGE':
        sage_output = sage_encode(data, mask)
        output = sage_output
    # if use_UDAGCN:
    #    ppmi_output = ppmi_encode(data, cache_name, mask)
    #    outputs = att_model([gcn_output, ppmi_output])
    #    return outputs
    # else:
    return output


def predict(data, cache_name, mask=None):
    encoded_output = encode(data, cache_name, mask)
    logits = cls_model(encoded_output)
    return logits


def evaluate(preds, labels):
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    return accuracy


def test(data, cache_name, mask=None):
    for model in models:
        model.eval()
    logits = predict(data, cache_name, mask)
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    accuracy = evaluate(preds, labels)
    return accuracy


epochs = args.epochs

'''
class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])

        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)


    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss
'''


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


def train(epoch):
    for model in models:
        model.train()
    optimizer.zero_grad()

    # global rate
    # rate = min((epoch + 1) / epochs, 0.05)

    encoded_source = encode(source_data, "source")
    encoded_target = encode(target_data, "target")
    # mmd_diss = MMD_loss()

    t_emb_full = encoded_target

    s_emb_train = encoded_source[s_train_mask, :]
    s_emb_val = encoded_source[s_val_mask, :]
    s_emb_test = encoded_source[s_test_mask, :]

    s_train_mu = torch.mean(s_emb_train, dim=0).cpu().detach().numpy()
    s_train_sigma = torch.cov(s_emb_train).cpu().detach().numpy()
    # s_train_sigma = np.cov(s_emb_train, rowvar=False)

    # s_val_mu = np.mean(s_emb_val, axis=0)
    # s_val_sigma = np.cov(s_emb_val, rowvar=False)

    s_val_mu = torch.mean(s_emb_val, dim=0).cpu().detach().numpy()
    s_val_sigma = torch.cov(s_emb_val).cpu().detach().numpy()

    dist_s_tra_s_val = mmd(s_emb_train, s_emb_val)
    dist_s_tra_t_ful = mmd(s_emb_train, t_emb_full)
    dist_s_val_t_ful = mmd(s_emb_val, t_emb_full)
    # dist = mmd_diss.forward(s_emb_train,s_emb_val)
    # fd_value = calculate_frechet_distance(s_train_mu, s_train_sigma, s_val_mu, s_val_sigma)
    logging.info(
        'this is the mmd_value in {}-th epoch: dist_s_tra_s_val = {}, dist_s_tra_t_ful = {}, dist_s_val_t_ful = {}'.format(
            epoch, dist_s_tra_s_val, dist_s_tra_t_ful, dist_s_val_t_ful))
    writer.add_scalar('curve/mmd_dist_s_tra_s_val_seed_' + str(seed), dist_s_tra_s_val, epoch)
    writer.add_scalar('curve/mmd_dist_s_tra_t_ful_seed_' + str(seed), dist_s_tra_t_ful, epoch)
    writer.add_scalar('curve/mmd_dist_s_val_t_ful_seed_' + str(seed), dist_s_val_t_ful, epoch)
    # encoded_target = encode(target_data, "target")
    source_logits = cls_model(encoded_source)

    # use source classifier loss:
    # cls_loss = loss_func(source_logits, source_data.y)
    # use the training set of the source dataset
    if args.full_s == 1:
        cls_loss = loss_func(source_logits, source_data.y)
    else:
        cls_loss = loss_func(source_logits[s_train_mask], source_data.y[s_train_mask])

    for model in models:
        for name, param in model.named_parameters():
            if "weight" in name:
                cls_loss = cls_loss + param.mean() * 3e-3

    '''
    if use_UDAGCN:
        # use domain classifier loss:
        source_domain_preds = domain_model(encoded_source)
        target_domain_preds = domain_model(encoded_target)

        source_domain_cls_loss = loss_func(
            source_domain_preds,
            torch.zeros(source_domain_preds.size(0)).type(torch.LongTensor).to(device)
        )
        target_domain_cls_loss = loss_func(
            target_domain_preds,
            torch.ones(target_domain_preds.size(0)).type(torch.LongTensor).to(device)
        )
        loss_grl = source_domain_cls_loss + target_domain_cls_loss
        loss = cls_loss + loss_grl

        # use target classifier loss:
        target_logits = cls_model(encoded_target)
        target_probs = F.softmax(target_logits, dim=-1)
        target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)

        loss_entropy = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))

        loss = loss + loss_entropy* (epoch / epochs * 0.01)


    else:
        loss = cls_loss
    '''
    loss = cls_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


best_source_acc = 0.0
best_target_acc = 0.0
best_epoch = 0.0
for epoch in range(1, epochs):
    train(epoch)
    source_correct = test(source_data, "source", source_data.test_mask.to(torch.bool))
    target_correct = test(target_data, "target")
    # print("Epoch: {}, source_acc: {}, target_acc: {}".format(epoch, source_correct, target_correct))
    logging.info('Epoch: {}, source_acc: {}, target_acc: {}'.format(epoch, source_correct, target_correct))
    writer.add_scalar('curve/acc_target_seed_' + str(seed), target_correct, epoch)
    if target_correct > best_target_acc:
        best_target_acc = target_correct
        best_source_acc = source_correct
        best_epoch = epoch
# print("=============================================================")
logging.info("=============================================================")
line = "{} - Epoch: {}, best_source_acc: {}, best_target_acc: {}" \
    .format(id, best_epoch, best_source_acc, best_target_acc)

logging.info(line)
logging.info(args)
logging.info('Finish!, this is the log dir: {}'.format(log_dir))
