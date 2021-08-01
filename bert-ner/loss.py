"""Utilities for token classification loss."""
import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    DiceLoss implemented from 'Dice Loss for Data-imbalanced NLP Tasks'
    Useful in dealing with unbalanced data
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        '''
        input: [N, C]
        target: [N, ]
        '''
        prob = torch.softmax(input, dim=1)
        prob = torch.gather(prob, dim=1, index=target.unsqueeze(1))
        dsc_i = 1 - ((1 - prob) * prob) / ((1 - prob) * prob + 1)
        dice_loss = dsc_i.mean()
        return dice_loss


class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction,
                                                                 ignore_index=self.ignore_index)
