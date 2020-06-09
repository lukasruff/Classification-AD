import torch.nn as nn

from base.base_net import BaseNet
from .mnist_LeNet import MNIST_LeNet
from .cifar10_LeNet import CIFAR10_LeNet
from .imagenet_WideResNet import ImageNet_WideResNet
from .toy_Net import Toy_Net


class ClassifierNet(BaseNet):

    def __init__(self, net_name, rep_dim=64, bias_terms=False):
        super().__init__()

        if net_name == 'mnist_LeNet_classifier':
            self.network = MNIST_LeNet(rep_dim=rep_dim, bias_terms=bias_terms)
        if net_name == 'cifar10_LeNet_classifier':
            self.network = CIFAR10_LeNet(rep_dim=rep_dim, bias_terms=bias_terms)
        if net_name == 'imagenet_WideResNet_classifier':
            self.network = ImageNet_WideResNet(rep_dim=rep_dim)
        if net_name == 'toy_Net_classifier':
            self.network = Toy_Net(rep_dim=rep_dim)

        self.linear = nn.Linear(self.network.rep_dim, 1)

    def forward(self, x):
        x = self.network(x)
        x = self.linear(x)
        return x
