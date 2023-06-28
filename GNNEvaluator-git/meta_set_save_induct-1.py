# coding=utf-8
import copy
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from argparse import ArgumentParser
from dual_gnn.dataset.DomainData import DomainData
from torch_geometric.nn import GraphSAGE, GCN, GCN, GAT, GIN, MLP
from meta_data_create import EdgeDrop_induct, NodeMixUp_induct, NodeFeatureMasking_induct, G_Sample_induct
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
from torch_geometric.utils import dropout_adj, to_networkx, to_undirected, degree, to_scipy_sparse_matrix, \
    from_scipy_sparse_matrix, sort_edge_index, add_self_loops


# import grafog.transforms as T


def main(args, device, new_sub_G, idx):

    if args.aug_method == 'edge_drop':
        aug_edge_drop_all = EdgeDrop_induct(p=args.edge_drop_all_p)
        aug_edge_data = aug_edge_drop_all(new_sub_G).to(device)
        aug_data = aug_edge_data
        logging.info('EdgeDrop!, p= {}'.format(args.edge_drop_all_p))
    elif args.aug_method == 'node_mix':
        aug_node_mix_all = NodeMixUp_induct(lamb=args.mix_lamb, num_classes=dataset_s.num_classes)
        aug_node_mix_data = aug_node_mix_all(new_sub_G).to(device)
        aug_data = aug_node_mix_data
        logging.info('NodeMixUp!, p= {}'.format(args.mix_lamb))
    elif args.aug_method == 'node_fmask':
        aug_node_fmask_all = NodeFeatureMasking_induct(p=args.node_fmask_all_p)
        aug_node_fmask_data = aug_node_fmask_all(new_sub_G).to(device)
        aug_data = aug_node_fmask_data
        logging.info('NodeFeatureMasking!, p= {}'.format(args.node_fmask_all_p))
    elif args.aug_method == 'g_sample':
        if args.sample_p == None:
            args.sample_p = 0.5
        aug_node_g_sample = G_Sample_induct(sample_size=int(new_sub_G.x.shape[0]*args.sample_p))
        print(new_sub_G.x.shape, new_sub_G.y.shape, new_sub_G.edge_index.shape, 'before')
        aug_node_g_sample_data = aug_node_g_sample(new_sub_G).to(device)
        aug_data = aug_node_g_sample_data
        logging.info('G_Sampling!, size= {}'.format(int(new_sub_G.x.shape[0]*args.sample_p)))
    #elif args.aug_method == 'combo1':
    #    aug_edge_drop_all = EdgeDrop_all(p=args.edge_drop_all_p)
    #    aug_node_fmask_all = NodeFeatureMasking_all(p=args.node_fmask_all_p)
    #    aug_edge_data_1 = aug_edge_drop_all(new_sub_G).to(device)
    #    aug_node_fmask_data_2 = aug_node_fmask_all(aug_edge_data_1).to(device)
    #    aug_data = aug_node_fmask_data_2
    #    logging.info('Combo!, p-edge-drop= {},p-node-fmask= {}' \
    #                 .format(args.edge_drop_all_p, args.node_fmask_all_p))
    logging.info(aug_data)
    torch.save(aug_data, os.path.join(log_dir, 'aug_data_' + str(idx) + '.pt'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, default='acm')
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--num_metas", type=int, default=5)
    parser.add_argument("--specific", type=int, default=0, choices=[0,1])
    parser.add_argument("--aug_method", type=str, default='edge_drop',
                        choices=['edge_drop', 'node_mix', 'node_fmask', 'g_sample'], \
                        help='method for augment data for creating the meta dataset')
    parser.add_argument("--node_drop_val_p", type=float, default=0.05)
    parser.add_argument("--edge_drop_all_p", type=float, default=0.05)
    parser.add_argument("--node_fmask_all_p", type=float, default=0.05)
    parser.add_argument("--sample_p", type=float, default=0.05)
    parser.add_argument("--mix_lamb", type=float, default=0.3, help='rational of mix features of nodes, ranges=[0,1]')
    parser.add_argument("--interval", nargs='+', type=int, help='List of integers split for each aug method', default=[100,200,300,400])


    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = './' + 'logs/MetaSet/Meta-save-{}-num-{}-{}-{}'.format(args.source,
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
    dataset_s = DomainData("data/{}".format(args.source), name=args.source)
    source_data = dataset_s[0]
    val_idx = torch.nonzero(source_data.val_mask).reshape(-1)
    test_idx = torch.nonzero(source_data.test_mask).reshape(-1)
    train_idx = torch.nonzero(source_data.train_mask).reshape(-1)
    new_idx = torch.cat((val_idx,test_idx))
    adj = to_scipy_sparse_matrix(source_data.edge_index).tocsr()
    edge_index_new = from_scipy_sparse_matrix(adj[new_idx, :][:, new_idx])
    y_new = source_data.y[new_idx]
    x_new = source_data.x[new_idx]

    new_sub_G = tg.data.Data(x=x_new, y=y_new, edge_index=edge_index_new[0])
    for idx in range(0, args.num_metas):
        if args.specific==1:
            logging.info('Please identify the augmentation method at the same time...')
            logging.info('The augmentation method is = {}'.format(args.aug_method))
            if args.aug_method == 'edge_drop':
                args.edge_drop_all_p = np.random.random()
            elif args.aug_method == 'node_mix':
                args.mix_lamb = np.random.random()
            elif args.aug_method == 'node_fmask':
                args.node_fmask_all_p = np.random.random()
            elif args.aug_method == 'g_sample':
                args.sample_p = np.random.uniform(1e-5,0.5,1)
            else:
                args.edge_drop_all_p = args.node_fmask_all_p = args.mix_lamb = args.sample_p = None
                if args.edge_drop_all_p==None or args.node_fmask_all_p == None or args.mix_lamb == None or args.sample_p==None:
                    logging.info('Fault! Please identify the aug menthod first')
                    assert False
            main(args, device, new_sub_G, idx)

        else:
            args.edge_drop_all_p = args.node_fmask_all_p = args.mix_lamb = args.sample_p = None
            if idx < args.interval[0]:
                args.aug_method = 'edge_drop'
                args.edge_drop_all_p = np.random.random()
            elif args.interval[0]<=idx < args.interval[1]:
                args.aug_method = 'node_fmask'
                args.node_fmask_all_p = np.random.random()
            elif args.interval[1] <= idx < args.interval[2]:
                args.aug_method = 'node_mix'
                args.mix_lamb = np.random.uniform(1e-5,0.3,1)[0]
            elif args.interval[2] <= idx < args.interval[3]:
                args.aug_method = 'g_sample'
                args.sample_p = np.random.uniform(0.6,1,1)[0]
            main(args, device, new_sub_G, idx)
            '''
            method_code = np.random.randint(0, 4)
            args.edge_drop_all_p = args.node_fmask_all_p = args.mix_lamb = args.sample_p = None
            if method_code == 0:
                args.aug_method = 'edge_drop'
                args.edge_drop_all_p = np.random.random()
            elif method_code == 1:
                args.aug_method = 'node_fmask'
                args.node_fmask_all_p = np.random.random()
            elif method_code == 2:
                args.aug_method = 'node_mix'
                args.mix_lamb = np.random.uniform(1e-5,0.3,1)[0]
            elif method_code == 3:
                args.aug_method = 'g_sample'
                args.sample_p = np.random.uniform(1e-5,0.5,1)[0]
            '''
            logging.info(
                'This is the rate of trans with "idx= {} with {}"-method: edge_drop_rate = {}, node_fmask_rate = {}, node_mix_rate = {}, g_sample = {}' \
                .format(idx, args.aug_method, args.edge_drop_all_p, args.node_fmask_all_p, args.mix_lamb, args.sample_p))


    logging.info('This is the log_dir: {}'.format(log_dir))

