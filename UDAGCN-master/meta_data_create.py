import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import copy
import torch_geometric as tg

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