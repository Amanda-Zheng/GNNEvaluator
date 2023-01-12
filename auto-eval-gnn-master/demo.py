from argparse import ArgumentParser
import os
import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sys
import random
from dual_gnn.dataset.DomainData import DomainData
from torch_geometric.nn import GraphSAGE, GCN
from meta_feat_acc import *
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.init as init
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable


class RegNet2(nn.Module):
    def __init__(self):
        super(RegNet2, self).__init__()
        self.fc3 = nn.Linear(24, 1)
        #self.fc4 = nn.Linear(8, 1)
        # self.dropout3 = nn.Dropout2d(0.5)

    def forward(self, x, y, f):
        z = torch.cat([x, y, f], dim=1)  # mean, variance, and fid
        z = self.fc3(z)
        # z = self.dropout3(z)
       # z = self.fc4(z)
        z = torch.sigmoid(z)
        return z


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


model = RegNet2()
lossfunc = nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-2)

data1 = Variable(torch.randn(4, 8), requires_grad=True)
data2 = Variable(2*torch.ones(4, 8), requires_grad=True)
data3 =Variable(torch.randn(4, 8), requires_grad=True)
y = Variable(torch.ones(4, 1), requires_grad=True)

for epoch in range(10):
    optimizer.zero_grad()
    output = model(data1, data2, data3)
    loss = lossfunc(output, y)  # 计算两者的误差
    loss.backward()
    #print(output.grad,loss.grad, data1.grad)
    optimizer.step()
    print(epoch, loss.item())

