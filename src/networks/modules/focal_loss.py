import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


# Implementation of the focal loss from https://arxiv.org/abs/1708.02002
class FocalLoss(_Loss):
    def __init__(self, gamma=2.0, eps=1e-7):
        super(FocalLoss, self).__init__(size_average=None, reduce=None, reduction='mean')
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        pt = pt.clamp(self.eps, 1. - self.eps)
        F_loss = (1 - pt).pow(self.gamma) * BCE_loss
        return F_loss.mean()
