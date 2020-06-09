import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class CIFAR10_LeNet(BaseNet):

    def __init__(self, rep_dim=256, bias_terms=False):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder network
        self.conv1 = nn.Conv2d(3, 32, 5, bias=bias_terms, padding=2)
        nn.init.xavier_normal_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=bias_terms)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=bias_terms, padding=2)
        nn.init.xavier_normal_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=bias_terms)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=bias_terms, padding=2)
        nn.init.xavier_normal_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=bias_terms)
        self.fc1 = nn.Linear(128 * 4 * 4, 512, bias=bias_terms)
        nn.init.xavier_normal_(self.fc1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn1d1 = nn.BatchNorm1d(512, eps=1e-04, affine=bias_terms)
        self.fc2 = nn.Linear(512, self.rep_dim, bias=bias_terms)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(int(x.size(0)), -1)
        x = self.fc1(x)
        x = F.leaky_relu(self.bn1d1(x))
        x = self.fc2(x)
        return x
