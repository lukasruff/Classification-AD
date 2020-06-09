import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class Toy_Net(BaseNet):

    def __init__(self, rep_dim=128, bias_terms=True):
        super().__init__()

        self.rep_dim = rep_dim

        self.fc1 = nn.Linear(2, 256, bias=bias_terms)
        nn.init.xavier_normal_(self.fc1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn1 = nn.BatchNorm1d(256, affine=bias_terms)

        self.fc2 = nn.Linear(256, self.rep_dim, bias=bias_terms)
        nn.init.xavier_normal_(self.fc2.weight, gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, x):
        x = F.elu(self.bn1(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x
