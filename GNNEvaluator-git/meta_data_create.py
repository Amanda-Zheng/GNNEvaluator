import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import copy
#from torch_geometric.loader import RandomNodeLoader
import torch_geometric as tg
from torch_geometric.utils import dropout_adj, to_networkx, to_undirected, degree, to_scipy_sparse_matrix, \
    from_scipy_sparse_matrix, sort_edge_index, add_self_loops

class NodeDrop_val(nn.Module):
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
        val_mask[torch.where(idx < self.p)] = 0
        new_data = tg.data.Data(x=x, edge_index=edge_idx, y=y, train_mask=train_mask, val_mask = val_mask, test_mask=test_mask)

        return new_data

class NodeDrop_all(nn.Module):
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
        train_mask[torch.where(idx < self.p)] = 0
        test_mask[torch.where(idx < self.p)] = 0
        val_mask[torch.where(idx < self.p)] = 0
        new_data = tg.data.Data(x=x, edge_index=edge_idx, y=y, train_mask=train_mask, val_mask = val_mask, test_mask=test_mask)

        return new_data

class G_Sample_induct(nn.Module):
    def __init__(self, sample_size=3000):
        super().__init__()
        self.sample_size = sample_size

    def forward(self, data):
        x = copy.deepcopy(data.x)
        y = copy.deepcopy(data.y)
        edge_index = copy.deepcopy(data.edge_index)

        idx = torch.randperm(edge_index.max())[:self.sample_size]

        adj = to_scipy_sparse_matrix(edge_index).tocsr()

        x_sampled = x[idx]
        edge_index_sampled = from_scipy_sparse_matrix(adj[idx, :][:, idx])
        y_sampled = y[idx]
        new_data = tg.data.Data(x=x_sampled, y=y_sampled, edge_index=edge_index_sampled[0])
        return new_data


class EdgeDrop_all(nn.Module):
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

        edge_idx = edge_idx.permute(1, 0)
        idx = torch.empty(edge_idx.size(0)).uniform_(0, 1)
        edge_idx = edge_idx[torch.where(idx >= self.p)].permute(1, 0)
        new_data = tg.data.Data(x=x, y=y, edge_index=edge_idx, train_mask=train_mask, test_mask=test_mask, val_mask = val_mask)
        return new_data

class EdgeDrop_induct(nn.Module):
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p

    def forward(self, data):
        x = copy.deepcopy(data.x)
        y = copy.deepcopy(data.y)
        edge_idx = copy.deepcopy(data.edge_index)

        edge_idx = edge_idx.permute(1, 0)
        idx = torch.empty(edge_idx.size(0)).uniform_(0, 1)
        edge_idx = edge_idx[torch.where(idx >= self.p)].permute(1, 0)
        new_data = tg.data.Data(x=x, y=y, edge_index=edge_idx)
        return new_data

class NodeMixUp_induct(nn.Module):
    def __init__(self, lamb, num_classes):
        super().__init__()
        self.lamb = lamb
        self.num_classes = num_classes

    def forward(self, data):
        x = copy.deepcopy(data.x)
        y = copy.deepcopy(data.y)
        edge_idx = copy.deepcopy(data.edge_index)

        n, d = x.shape

        pair_idx = torch.randperm(n)
        x_b = x[pair_idx]
        y_b = y[pair_idx]
        y_a_oh = F.one_hot(y, self.num_classes)
        y_b_oh = F.one_hot(y_b, self.num_classes)

        x_mix = (self.lamb * x) + (1 - self.lamb) * x_b
        y_mix = (self.lamb * y_a_oh) + (1 - self.lamb) * y_b_oh
        new_y = y_mix.argmax(1)

        # new_x = torch.vstack([x, x_mix])
        # new_y = torch.vstack([y_a_oh, y_mix])

        new_data = tg.data.Data(x=x_mix, y=new_y, edge_index=edge_idx)
        return new_data


class NodeFeatureMasking_all(nn.Module):
    def __init__(self, p=0.15):
        super().__init__()
        self.p = p

    def forward(self, data):
        x = copy.deepcopy(data.x)
        y = copy.deepcopy(data.y)
        train_mask = copy.deepcopy(data.train_mask)
        test_mask = copy.deepcopy(data.test_mask)
        val_mask = copy.deepcopy(data.val_mask)
        edge_idx = copy.deepcopy(data.edge_index)

        n, d = x.shape

        idx = torch.empty((d,), dtype=torch.float32).uniform_(0, 1) < self.p
        x = x.clone()
        x[:, idx] = 0

        new_data = tg.data.Data(x=x, y=y, edge_index=edge_idx, train_mask=train_mask, test_mask=test_mask, val_mask=val_mask)
        return new_data

class NodeFeatureMasking_induct(nn.Module):
    def __init__(self, p=0.15):
        super().__init__()
        self.p = p

    def forward(self, data):
        x = copy.deepcopy(data.x)
        y = copy.deepcopy(data.y)
        edge_idx = copy.deepcopy(data.edge_index)

        n, d = x.shape

        idx = torch.empty((d,), dtype=torch.float32).uniform_(0, 1) < self.p
        x = x.clone()
        x[:, idx] = 0

        new_data = tg.data.Data(x=x, y=y, edge_index=edge_idx)
        return new_data

#train_loader = RandomNodeLoader(data, num_parts=40, shuffle=True,
#                                num_workers=5)