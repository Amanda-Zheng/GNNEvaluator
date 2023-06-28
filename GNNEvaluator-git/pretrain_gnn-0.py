# coding=utf-8
import copy
import os
from argparse import ArgumentParser
from torch_geometric.nn import GraphSAGE, GCN, GAT, GIN, MLP
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
from torch_geometric import seed


def main(args, device):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset_s = DomainData("data/{}".format(args.source), name=args.source)
    source_data = dataset_s[0]
    logging.info(source_data)

    dataset_t = DomainData("data/{}".format(args.target), name=args.target)
    target_data = dataset_t[0]
    logging.info(target_data)

    dataset_t2 = DomainData("data/{}".format(args.target2), name=args.target2)
    target_data2 = dataset_t2[0]
    logging.info(target_data2)


    source_data = source_data.to(device)
    target_data = target_data.to(device)
    target_data2 = target_data2.to(device)

    loss_func = nn.CrossEntropyLoss().to(device)

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
        encoder = MLP(channel_list=[source_data.num_node_features,args.hid_dim, args.encoder_dim]).to(device)

    #note that here is no softmax activation function
    cls_model = nn.Sequential(nn.Linear(args.encoder_dim, dataset_s.num_classes), ).to(device)
    nn.init.xavier_uniform_(cls_model[0].weight)
    nn.init.constant_(cls_model[0].bias, 0.0)

    models = [encoder, cls_model]
    params = itertools.chain(*[model.parameters() for model in models])
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    best_s_val_acc = 0.0
    best_s_test_acc = 0.0
    best_epoch = 0.0
    t_full_acc_ls =[]
    t_full_acc_ls2 = []
    #s_train_acc_ls = []
    #s_val_acc_ls = []
    #s_test_acc_ls =[]
    for epoch in range(1, args.epochs):
        s_train_acc, s_train_loss = train(models, encoder, cls_model, optimizer, loss_func, source_data)
        s_val_acc = test(source_data, models, encoder, cls_model, mask=source_data.val_mask.to(torch.bool))
        s_test_acc = test(source_data, models, encoder, cls_model, mask=source_data.test_mask.to(torch.bool))
        t_full_acc = test(target_data, models, encoder, cls_model)
        t_full_acc_ls.append(t_full_acc.item())

        t_full_acc2 = test(target_data2, models, encoder, cls_model)
        t_full_acc_ls2.append(t_full_acc2.item())
        #s_train_acc_ls.append(s_train_acc.item())
        #s_val_acc_ls.append(s_val_acc.item())
        #s_test_acc_ls.append(s_test_acc.item())
        #torch.save(t_full_acc_ls, os.path.join(log_dir,'t_full_acc_ls.pth'))
        #torch.save(s_train_acc_ls, os.path.join(log_dir,'s_train_acc_ls.pth'))
        #torch.save(s_val_acc_ls, os.path.join(log_dir,'s_val_acc_ls.pth'))
        #torch.save(s_test_acc_ls, os.path.join(log_dir,'s_test_acc_ls.pth'))
        #if s_train_acc > best_s_train_acc:
        #    best_s_train_acc = s_train_acc
        #    best_epoch = epoch
        logging.info('Epoch: {}, source_train_loss: {:.4f}, source_train_acc: {:.4f}, source_val_acc: {:.4f}, source_test_acc:{:.4f}'. \
                     format(epoch, s_train_loss, s_train_acc, s_val_acc, s_test_acc))
        logging.info('Epoch: {}, target_full_acc: {:.4f},target_full_acc2: {:.4f}'. \
                     format(epoch, t_full_acc,t_full_acc2))
        writer.add_scalar('curve/acc_source_train_seed_' + str(args.seed), s_train_acc, epoch)
        writer.add_scalar('curve/acc_source_val_seed_' + str(args.seed), s_val_acc, epoch)
        writer.add_scalar('curve/acc_source_test_seed_' + str(args.seed), s_test_acc, epoch)
        writer.add_scalar('curve/loss_source_train_seed_' + str(args.seed), s_train_loss, epoch)
        writer.add_scalar('curve/acc_target_full_seed_' + str(args.seed), t_full_acc, epoch)
        writer.add_scalar('curve/acc_target2_full_seed_' + str(args.seed), t_full_acc2, epoch)
        if s_val_acc > best_s_val_acc:
            best_s_val_acc = s_val_acc
            best_s_test_acc = s_test_acc
            best_epoch = epoch
            best_target_acc = test(target_data, models, encoder, cls_model)
            best_target_acc2 = test(target_data2, models, encoder, cls_model)
            torch.save(encoder.state_dict(), os.path.join(log_dir, 'encoder.pt'))
            torch.save(cls_model.state_dict(), os.path.join(log_dir, 'cls_model.pt'))

    line = "Best Epoch: {}, best_source_test_acc: {}, best_source_val_acc: {}, best_target_acc: {}, best_target_acc2: {}" \
        .format(best_epoch, best_s_test_acc, best_s_val_acc, best_target_acc, best_target_acc2)
    #line = "Best Epoch: {}, best_source_test_acc: {}, best_source_val_acc: {}" \
    #    .format(best_epoch, best_s_test_acc, best_s_val_acc)
    #line = "Best Epoch: {}, best_source_train_acc: {}" \
    #    .format(best_epoch, best_s_train_acc)
    logging.info(line)
    logging.info(args)
    logging.info('Finish!, this is the log dir = {}'.format(log_dir))


def test(data, models, encoder, cls_model, mask=None):
    for model in models:
        model.eval()

    if isinstance(encoder, MLP):
        emb_out = encoder(data.x)
    else: 
        emb_out = encoder(data.x, data.edge_index)

    logits = cls_model(emb_out) if mask is None else cls_model(emb_out)[mask]
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    return accuracy


def train(models, encoder, cls_model, optimizer, loss_func, source_data):
    for model in models:
        model.train()
    s_train_mask = source_data.train_mask.to(torch.bool)
    if isinstance(encoder, MLP):
        emb_source = encoder(source_data.x)
    else: 
        emb_source = encoder(source_data.x, source_data.edge_index)

    source_logits = cls_model(emb_source)
    source_probs = F.softmax(source_logits,dim=1)

    if args.full_s == 1:
        cls_loss = loss_func(source_probs, source_data.y)
        preds = source_probs.argmax(dim=1)
        labels = source_data.y
    else:
        cls_loss = loss_func(source_probs[s_train_mask], source_data.y[s_train_mask])
        preds = source_probs.argmax(dim=1)[s_train_mask]
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
    parser.add_argument("--target2", type=str, default='network')
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--hid_dim", type=int, default=128)
    parser.add_argument("--encoder_dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--full_s", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=2)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = './' + 'logs/Models_tra/{}-to-{}-{}-full-{}-{}-{}'.format(args.source, args.target, args.model,
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
