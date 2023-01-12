# coding=utf-8
import copy
import os
from argparse import ArgumentParser
from torch_geometric.nn import GraphSAGE, GCN
from dual_gnn.dataset.DomainData import DomainData
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

    dataset_t = DomainData("data/{}".format(args.target), name=args.target)
    target_data = dataset_t[0]
    logging.info(target_data)

    source_data = source_data.to(device)
    target_data = target_data.to(device)

    loss_func = nn.CrossEntropyLoss().to(device)

    if args.model == 'GCN':
        encoder = GCN(source_data.num_node_features, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                      num_layers=2).to(device)
    elif args.model == 'SAGE':
        encoder = GraphSAGE(source_data.num_node_features, hidden_channels=args.hid_dim, out_channels=args.encoder_dim,
                            num_layers=2).to(device)

    cls_model = nn.Sequential(nn.Linear(args.encoder_dim, dataset_s.num_classes), ).to(device)

    models = [encoder, cls_model]

    params = itertools.chain(*[model.parameters() for model in models])
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    best_s_val_acc = best_s_test_acc = 0.0
    best_epoch = 0.0
    for epoch in range(1, args.epochs):
        s_train_acc, s_train_loss = train(models, encoder, cls_model, optimizer, loss_func, source_data)
        s_val_acc = test(source_data, models, encoder, cls_model, mask=source_data.val_mask.to(torch.bool))
        s_test_acc = test(source_data, models, encoder, cls_model, mask=source_data.test_mask.to(torch.bool))
        logging.info('Epoch: {}, source_train_loss: {}, source_train_acc: {}, source_val_acc: {}, source_test_acc:{}'. \
                     format(epoch, s_train_loss, s_train_acc, s_val_acc, s_test_acc))
        writer.add_scalar('curve/acc_source_train_seed_' + str(args.seed), s_train_acc, epoch)
        writer.add_scalar('curve/acc_source_val_seed_' + str(args.seed), s_val_acc, epoch)
        writer.add_scalar('curve/acc_source_test_seed_' + str(args.seed), s_test_acc, epoch)
        writer.add_scalar('curve/loss_source_train_seed_' + str(args.seed), s_train_loss, epoch)
        if s_val_acc > best_s_val_acc:
            best_s_val_acc = s_val_acc
            best_s_test_acc = s_test_acc
            best_epoch = epoch
            best_target_acc = test(target_data, models, encoder, cls_model)
            torch.save(encoder.state_dict(), os.path.join(log_dir, 'encoder.pt'))
            torch.save(cls_model.state_dict(), os.path.join(log_dir, 'cls_model.pt'))

    line = "Best Epoch: {}, best_source_test_acc: {}, best_source_val_acc: {}, best_target_acc: {}" \
        .format(best_epoch, best_s_test_acc, best_s_val_acc, best_target_acc)
    logging.info(line)
    logging.info(args)
    logging.info('Finish!, this is the log dir = {}'.format(log_dir))


def test(data, models, encoder, cls_model, mask=None):
    for model in models:
        model.eval()
    emb_out = encoder(data.x, data.edge_index)
    logits = cls_model(emb_out) if mask is None else cls_model(emb_out)[mask]
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    return accuracy


def train(models, encoder, cls_model, optimizer, loss_func, source_data):
    for model in models:
        model.train()
    s_train_mask = source_data.train_mask.to(torch.bool)
    emb_source = encoder(source_data.x, source_data.edge_index)
    source_logits = cls_model(emb_source)

    if args.full_s == 1:
        cls_loss = loss_func(source_logits, source_data.y)
        preds = source_logits.argmax(dim=1)
        labels = source_data.y
    else:
        cls_loss = loss_func(source_logits[s_train_mask], source_data.y[s_train_mask])
        preds = source_logits.argmax(dim=1)[s_train_mask]
        labels = source_data.y[s_train_mask]

    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()

    loss = cls_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return accuracy, loss.item()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, default='acm')
    parser.add_argument("--target", type=str, default='dblp')
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--hid_dim", type=int, default=128)
    parser.add_argument("--encoder_dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--full_s", type=int, default=1)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = './' + 'logs/{}-to-{}-{}-full-{}-{}-{}'.format(args.source, args.target, args.model,
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
    writer = SummaryWriter(log_dir + '/tbx_log')
    logging.info(args)
    main(args, device)
