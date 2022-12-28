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


def main(args, device):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_s = DomainData("data/{}".format(args.source), name=args.source)
    source_data = dataset_s[0]
    logging.info(source_data)

    source_data = source_data.to(device)

    loss_func = nn.CrossEntropyLoss().to(device)
    if args.model == 'GCN':
        encoder = GNN(type="gcn", num_features=dataset_s.num_features, encoder_dim=args.encoder_dim).to(device)
    elif args.model == 'SAGE':
        from torch_geometric.nn import GraphSAGE
        encoder = GraphSAGE(source_data.num_node_features, hidden_channels=args.encoder_dim, num_layers=2).to(device)

    cls_model = nn.Sequential(nn.Linear(args.encoder_dim, dataset_s.num_classes), ).to(device)

    models = [encoder, cls_model]

    params = itertools.chain(*[model.parameters() for model in models])
    optimizer = torch.optim.Adam(params, lr=args.lr)

    best_source_acc = 0.0
    best_epoch = 0.0
    for epoch in range(1, args.epochs):
        train(models, encoder, cls_model, optimizer, loss_func,source_data)
        source_correct = test(source_data, models, encoder, cls_model, "source", source_data.test_mask.to(torch.bool))
        logging.info('Epoch: {}, source_acc: {}'.format(epoch, source_correct))
        writer.add_scalar('curve/acc_source_seed_' + str(args.seed), source_correct, epoch)
        if source_correct > best_source_acc:
            best_source_acc = source_correct
            best_epoch = epoch
            torch.save(encoder.state_dict(),os.path.join(log_dir, args.save_m_pt, 'encoder.pt'))
            torch.save(cls_model.state_dict(), os.path.join(log_dir,args.save_m_pt, 'cls_model.pt'))

    line = "{} - Epoch: {}, best_source_acc: {}" \
        .format(id, best_epoch, best_source_acc)
    logging.info(line)
    logging.info(args)
    logging.info('Finish!, this is the log dir = {} and cpk dir = {}'.format(log_dir, args.save_m_pt))


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
    return logits


def evaluate(preds, labels):
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    return accuracy


def test(data, models, encoder, cls_model, cache_name, mask=None):
    for model in models:
        model.eval()
    logits = predict(data, encoder, cls_model, cache_name, mask)
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    accuracy = evaluate(preds, labels)
    return accuracy


def train(models, encoder, cls_model, optimizer, loss_func, source_data):
    s_train_mask = source_data.train_mask.to(torch.bool)

    for model in models:
        model.train()

    encoded_source = encode(source_data, encoder, "source")
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



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, default='acm')
    parser.add_argument("--name", type=str, default='UDAGCN')
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--UDAGCN", type=bool, default=False)
    parser.add_argument("--encoder_dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--full_s", type=int, default=1)
    parser.add_argument("--save_m_pt", type=str, default='checkpoints/')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = './' + 'logs/{}-{}-full-{}-{}-{}'.format(args.source, args.model,
                                                                    str(args.full_s),
                                                                    str(args.seed),
                                                                    datetime.datetime.now().strftime(
                                                                        "%Y%m%d-%H%M%S-%f"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir,args.save_m_pt)):
        os.makedirs(os.path.join(log_dir,args.save_m_pt))
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
    main(args, device)
