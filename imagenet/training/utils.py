import torch
import numpy as np
import time
import torch.nn.functional as F
import types

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def CE_loss(outputs, labels):
    if len(labels.shape) == 1:
        return F.cross_entropy(outputs, labels)
    else:
        loss = torch.nn.KLDivLoss(reduction='batchmean')
        return loss(F.log_softmax(outputs, dim=1), labels)
def KD_loss(t_outputs, s_outputs):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(s_outputs, dim=1), F.softmax(t_outputs, dim=1))
def LS_loss(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)
def SKD_loss(t_outputs, s_outputs, labels):
    perm = torch.randperm(t_outputs.shape[1], device=t_outputs.device)
    inv = torch.zeros_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    t_outputs_shuffled = t_outputs[:, perm]
    tmp_outputs_1 = torch.gather(t_outputs_shuffled, 1, inv[labels].reshape(-1, 1))
    tmp_outputs_2 = torch.gather(t_outputs_shuffled, 1, labels.reshape(-1, 1))
    t_outputs_shuffled.scatter_(1, inv[labels].reshape(-1, 1), tmp_outputs_2)
    t_outputs_shuffled.scatter_(1, labels.reshape(-1, 1), tmp_outputs_1)
    t_outputs_shuffled = t_outputs_shuffled.clone().detach()
    return KD_loss(t_outputs_shuffled, s_outputs)
def RKD_loss(t_outputs, s_outputs, labels):
    one_hot = F.one_hot(labels, num_classes=t_outputs.shape[1]).float()
    t_outputs_modified = t_outputs + one_hot * (t_outputs.abs().max() + 1)
    perm = torch.argsort(t_outputs_modified, dim=1)
    t_outputs_shuffled = t_outputs.gather(1, perm)
    perm[:, :-1] = torch.flip(perm[:, :-1], dims=[1])
    t_outputs_shuffled = torch.scatter(t_outputs_shuffled, 1, perm, t_outputs_shuffled)
    return KD_loss(t_outputs_shuffled, s_outputs)