# coding=utf-8
import copy
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from argparse import ArgumentParser
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
import math
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
)
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch_scatter import scatter
import torch_sparse
from torch_geometric.utils import from_scipy_sparse_matrix

from torch_geometric.typing import OptTensor

from torch_geometric.utils.num_nodes import maybe_num_nodes


# from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE

def main(aug_data, encoder, cls_model, s_emb_train):
    models = [encoder, cls_model]
    for model in models:
        model.eval()
    aug_acc, aug_feat = test(aug_data, models, encoder, cls_model)


    train_feat = s_emb_train.detach()
    taraug_feat = aug_feat.detach()

    norm1 = torch.norm(train_feat, dim=-1).view(train_feat.shape[0], 1)
    norm2 = torch.norm(taraug_feat, dim=-1).view(1, taraug_feat.shape[0])

    end_norm = torch.matmul(norm1, norm2)
    cos_dist = torch.transpose(torch.matmul(train_feat, taraug_feat.t()) / end_norm, 0, 1)

    edge_dist = aug_data.edge_index

    new_sim_data = tg.data.Data(x=cos_dist, y=aug_acc, edge_index=edge_dist)
    return new_sim_data


def real_tar(target_data, device, encoder, cls_model, s_emb_train):
    models = [encoder, cls_model]
    for model in models:
        model.eval()

    GT_target_acc, target_data_feat = test(target_data, models, encoder, cls_model)

    train_feat = s_emb_train.detach()
    taraug_feat = target_data_feat.detach()

    norm1 = torch.norm(train_feat, dim=-1).view(train_feat.shape[0], 1)
    norm2 = torch.norm(taraug_feat, dim=-1).view(1, taraug_feat.shape[0])

    end_norm = torch.matmul(norm1, norm2)
    cos_dist = torch.transpose(torch.matmul(train_feat, taraug_feat.t()) / end_norm, 0, 1)
    edge_dist = target_data.edge_index

    new_tar_data = tg.data.Data(x=cos_dist, y=GT_target_acc, edge_index=edge_dist)

    return new_tar_data, GT_target_acc


def test(data, models, encoder, cls_model, mask=None):
    for model in models:
        model.eval()

    if isinstance(encoder, MLP):
        emb_out = encoder(data.x)
    else:
        emb_out = encoder(data.x, data.edge_index)

    logits = F.softmax(cls_model(emb_out) if mask is None else cls_model(emb_out)[mask],dim=-1)
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    final_emb = emb_out if mask is None else emb_out[mask]
    return accuracy, final_emb


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


from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN_pred(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, out_channels,dropout):
        super(GCN_pred, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.lin = Linear(out_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.sigmoid(self.lin(x))
        out = x.reshape(-1)
        return out


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, default='acm')
    parser.add_argument("--target", type=str, default='dblp')
    parser.add_argument("--target2", type=str, default='network')
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hid_dim", type=int, default=128)
    parser.add_argument("--eval_hid_dim", type=int, default=128)
    parser.add_argument("--encoder_dim", type=int, default=16)
    parser.add_argument("--eval_out_dim", type=int, default=16)
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--num_metas", type=int, default=5)
    parser.add_argument("--pre_epochs", type=int, default=100)
    parser.add_argument("--pre_lr", type=float, default=1e-4)
    parser.add_argument("--pre_drop", type=float, default=0.5)
    parser.add_argument("--pre_wd", type=float, default=5e-5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--early_stop", type=int, default=5)
    parser.add_argument("--early_stop_train", type=int, default=5)
    parser.add_argument("--model_path", type=str,
                        default='./logs/Models_tra/acm-to-dblp-GCN-full-0-0-20230402-230813-577875/')
    parser.add_argument("--metaG_path", type=str,
                        default='./logs/MetaSet/Meta-save-acm-num-300-0-20230424-113801-947245/')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = './' + 'logs/Pred/Meta-feat-acc-{}-{}-num-{}-{}-{}'.format(args.source, args.model,
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
    if isinstance(encoder, MLP):
        emb_source = encoder(source_data.x)
    else:
        emb_source = encoder(source_data.x, source_data.edge_index)
    s_train_mask = source_data.train_mask.to(torch.bool)
    s_emb_train = emb_source[s_train_mask, :]

    meta_feat_ls = []
    meta_adj_ls = []
    meta_acc_ls = []
    aug_data_ls = []

    '''
    for idx in range(0, args.num_metas):
        aug_data = torch.load(os.path.join(args.aug_data_path, 'aug_data_' + str(idx) + '.pt'), map_location=device)
        aug_data_ls.append(aug_data)
        
    from torch_geometric.loader import DataLoader
    train_ls = aug_data_ls[:-args.test_num_metas]
    test_ls = aug_data_ls[args.test_num_metas:]
    train_loader = DataLoader(train_ls, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ls, batch_size=4, shuffle=False)
    '''

    feat_in = source_data.x[source_data.train_mask.to(torch.bool), :].shape[0]
    pred_model = GCN_pred(feat_in, hidden_channels=args.eval_hid_dim, out_channels=args.eval_out_dim, dropout=args.pre_drop).to(device)
    pre_optimizer = torch.optim.Adam(pred_model.parameters(), lr=args.pre_lr, weight_decay=args.pre_wd)
    pre_criterion = torch.nn.MSELoss()
    #new_sim_data_ls = []
    #logging.info('Loading meta sets ...')
    #for idx in range(0, args.num_metas):
    #    aug_data = torch.load(os.path.join(args.aug_data_path, 'aug_data_' + str(idx) + '.pt'), map_location=device)
    #    new_sim_data = main(aug_data, encoder, cls_model, s_emb_train)
    #    if idx%20==0:
    #        logging.info('Idx = {}, with {}'.format(idx, new_sim_data))
    #    new_sim_data_ls.append(new_sim_data.cpu())
    #logging.info('Done!')
    #train_ls = new_sim_data_ls[:-args.test_num_metas]
    #test_ls = new_sim_data_ls[-args.test_num_metas:]
    #train_loader = DataLoader(train_ls, batch_size=args.train_batch_size, shuffle=True)
    #test_loader = DataLoader(test_ls, batch_size=args.test_batch_size, shuffle=False)
    from torch_geometric.loader import DataLoader

    dataset_t = DomainData("data/{}".format(args.target), name=args.target)
    target_data = dataset_t[0].to(device)
    logging.info(target_data)

    dataset_t2 = DomainData("data/{}".format(args.target2), name=args.target2)
    target_data2 = dataset_t2[0].to(device)
    logging.info(target_data2)

    new_tar_data = real_tar(target_data, device, encoder, cls_model, s_emb_train)
    tar_loader = DataLoader([new_tar_data], batch_size=1, shuffle=False)

    new_tar_data2 = real_tar(target_data2, device, encoder, cls_model, s_emb_train)
    tar_loader2 = DataLoader([new_tar_data2], batch_size=1, shuffle=False)

    import pickle
    with open(os.path.join(args.metaG_path,"test_meta_data.pickle"), "rb") as f1:
        test_meta_data = pickle.load(f1)
    with open(os.path.join(args.metaG_path,"train_meta_data.pickle"), "rb") as f2:
        train_meta_data = pickle.load(f2)

    train_loader = DataLoader(train_meta_data, batch_size=args.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_meta_data, batch_size=args.test_batch_size, shuffle=False)

    best_RMSE_test = best_MAE_test = 1e20
    writer = SummaryWriter(log_dir + '/tbx_log')
    best_train_loss = 1e20
    best_train_patience = 0
    # best_RMSE_test = best_MAE_test = len(test_loader) * 100.
    for epochs_pre in range(0, args.pre_epochs+1):
        pred_model.train()
        cnt=0
        loss_epch = 0
        for tra_data in train_loader:
            tra_data = tra_data.to(device)
            out = pred_model(tra_data.x, tra_data.edge_index, tra_data.batch)  # Perform a single forward pass.
            loss = pre_criterion(out, tra_data.y)
            loss_epch +=loss.item()
            current_step = cnt + epochs_pre*len(train_loader)
            cnt += 1
            writer.add_scalar('train_loss', loss.item(), current_step)# Compute the loss.
            loss.backward()  # Derive gradients.
            pre_optimizer.step()  # Update parameters based on gradients.
            pre_optimizer.zero_grad()

            #if current_step% 20==0:
            #    logging.info('Steps = {} in Total = {}, Loss = {:.4f} in Training'.format(current_step, args.pre_epochs*len(train_loader), loss.item()))

        writer.add_scalar('train_loss_epoch', loss_epch, epochs_pre)
        logging.info('Epoch  = {}, Train Loss on all data= {:.4f}' \
                     .format(epochs_pre, loss_epch))
        if loss_epch <= best_train_loss:
            best_train_loss = loss_epch
        else:
            best_train_patience+=1
            if best_train_patience== args.early_stop_train:
                logging.info('breaking at {}-th epoch for training overfitting...'.format(loss_epch))
                break

        if epochs_pre % 1 == 0:
            pred_model.eval()
            out_ls = []
            y_ls = []
            for tes_data in test_loader:  # Iterate in batches over the training/test dataset.
                tes_data = tes_data.to(device)
                out = pred_model(tes_data.x, tes_data.edge_index, tes_data.batch)
                out_ls.append(out.detach().cpu().numpy())
                y_ls.append(tes_data.y.detach().cpu().numpy())
            out_in = np.concatenate(out_ls, axis=0)
            y_in = np.concatenate(y_ls,axis=0)
            test_RMSE = mean_squared_error(out_in, y_in, squared=False)
            writer.add_scalar('test_rmse', test_RMSE, epochs_pre)
            test_MAE = mean_absolute_error(out_in, y_in)
            writer.add_scalar('test_mae', test_MAE, epochs_pre)
            logging.info('Epoch  = {}, Test: RMSE = {:.4f}, MAE = {:.4f}' \
                         .format(epochs_pre, test_RMSE, test_MAE))
            for tar_data in tar_loader:
                tar_data = tar_data[0].to(device)
                out_tar = pred_model(tar_data.x, tar_data.edge_index, tar_data.batch)
                MAE_tar = mean_absolute_error(out_tar.detach().cpu().numpy(), tar_data.y.detach().cpu().numpy())
                writer.add_scalar('target_mae', MAE_tar, epochs_pre)
                logging.info('TARGET_DATASET = {}, Out_target = {:.2f} vs. GT Out_target = {:.2f} in DIFF = {}' \
                             .format(args.target, out_tar.detach().cpu().numpy()[0] * 100.,
                                     tar_data.y.detach().cpu().numpy()[0] * 100.,
                                     MAE_tar* 100.))
            for tar_data2 in tar_loader2:
                tar_data2 = tar_data2[0].to(device)
                out_tar2 = pred_model(tar_data2.x, tar_data2.edge_index, tar_data2.batch)
                MAE_tar2 = mean_absolute_error(out_tar2.detach().cpu().numpy(), tar_data2.y.detach().cpu().numpy())
                writer.add_scalar('target2_mae', MAE_tar2, epochs_pre)
                logging.info('TARGET_DATASET = {}, Out_target2 = {:.2f} vs. GT Out_target = {:.2f} in DIFF = {}' \
                             .format(args.target2, out_tar2.detach().cpu().numpy()[0] * 100.,
                                     tar_data2.y.detach().cpu().numpy()[0] * 100.,
                                     MAE_tar2* 100.))



            # out_ls.append(out.item())
            # y_ls.append(tes_data.y.item())

            # RMSE = mean_squared_error(np.array(out_ls), np.array(y_ls), squared=False)
            # MAE = mean_absolute_error(np.array(out_ls), np.array(y_ls))

            if test_RMSE < best_RMSE_test:
                best_RMSE_test = test_RMSE
                best_MAE_test = test_MAE
                best_test_epoch = epochs_pre
                best_OUT_tar = out_tar.detach().cpu().numpy()[0] * 100.
                best_MAE_tar = MAE_tar* 100.

                best_OUT_tar2 = out_tar2.detach().cpu().numpy()[0] * 100.
                best_MAE_tar2 = MAE_tar2* 100.
                torch.save(pred_model.state_dict(), os.path.join(log_dir, 'pre_model.pt'))
                patience = 0
            else:
                patience += 1
                if patience == args.early_stop:
                    break
            # Early stopping


    logging.info('Test: Best RMSE = {:.2f}, Best MAE = {:.2f}, with in Best Epoch = {}' \
                 .format(best_RMSE_test, best_MAE_test, best_test_epoch))
    logging.info('For in Test for {}: Best Out_target = {:.2f} vs. GT Out_target = {:.2f} in DIFF = {}' \
                 .format(args.target, best_OUT_tar,
                         tar_data.y.detach().cpu().numpy()[0] * 100.,
                         best_MAE_tar))
    logging.info('For in Test for {}: Best Out_target = {:.2f} vs. GT Out_target = {:.2f} in DIFF = {}' \
                 .format(args.target2, best_OUT_tar2,
                         tar_data2.y.detach().cpu().numpy()[0] * 100.,
                         best_MAE_tar2))
    logging.info(args)
    logging.info('Finish!, this is the finish logdir = {}'.format(log_dir))
