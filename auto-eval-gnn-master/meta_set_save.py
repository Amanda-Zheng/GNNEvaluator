# coding=utf-8
import copy
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from argparse import ArgumentParser
from dual_gnn.dataset.DomainData import DomainData
from torch_geometric.nn import GraphSAGE,GCN
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


def main(args, device, idx):
    dataset_s = DomainData("data/{}".format(args.source), name=args.source)
    source_data = dataset_s[0]
    logging.info(source_data)

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

    torch.save(aug_data,os.path.join(log_dir, 'aug_data_'+str(idx)+'.pt'))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, default='acm')
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--num_metas", type=int, default=5)
    parser.add_argument("--aug_method", type=str, default='edge_drop',
                        choices=['edge_drop', 'node_mix', 'node_fmask', 'combo'], \
                        help='method for augment data for creating the meta dataset')
    parser.add_argument("--node_drop_val_p", type=float, default=0.05)
    parser.add_argument("--edge_drop_all_p", type=float, default=0.05)
    parser.add_argument("--node_fmask_all_p", type=float, default=0.05)
    parser.add_argument("--mix_lamb", type=float, default=0.3, help='rational of mix features of nodes, ranges=[0,1]')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = './' + 'logs/Meta-save-{}-num-{}-{}-{}'.format(args.source,
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

    for idx in range(0, args.num_metas):
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
        main(args, device,idx)

    logging.info(args)
    logging.info('Finish, this is the log dir = {}'.format(log_dir))
