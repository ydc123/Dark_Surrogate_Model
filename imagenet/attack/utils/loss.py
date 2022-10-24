import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy


def HEM_CrossEntropy(output, targets):
    likelihood = F.softmax(output, dim=1)[range(output.shape[0]), target]
    loss = torch.sum(torch.log(1. - likelihood))
    return loss


def CW_CrossEntropy(output, targets):
    one_hot = torch.nn.functional.one_hot(target, num_classes=output.shape[1]).cuda().float()
    real = torch.sum(one_hot * output, dim=-1)
    other = torch.max((1-one_hot) * logits - 10000000* one_hot, dim=-1)[0]
    loss =  torch.mean(other - real)
    return loss
def Logits(output, targets):
    return -output.gather(1, targets.reshape(-1, 1)).sum()



def cal_dis(feat1, feat2):
    return F.cosine_similarity(feat1, feat2)


class WeightCosine():

    def __init__(self, model):
        self.model = model
        self.linear = None
        for mod in self.model.modules():
            if isinstance(mod, nn.Linear):
                self.linear = mod
        assert self.linear is not None
        self.linear.register_forward_hook(self.fetch_feat_hook_wrapper())

    def fetch_feat_hook_wrapper(self, ):
        def fetch_feat_hook(module, input, output):
            self.feat = input[0]
        return fetch_feat_hook

    def __call__(self, outputs, targets):
        loss = 0
        for i in range(targets.size(0)):
            feat_tensor = self.feat[i:i+1]
            ind = targets[i]
            weight_tensor = self.linear.weight.data[ind:ind+1, :]
            loss += cal_dis(feat_tensor, weight_tensor)
        return -loss

class LossListModule():

    def __init__(self, loss_list):
        self.loss_list = loss_list
    
    def __call__(self, outputs, targets):
        loss = 0
        for loss_module in self.loss_list:
            loss += loss_module(outputs, targets)
        return loss


def LossFactory(loss_function, **kwargs):
    loss_list = []
    if 'CE' in loss_function.split(','):
        loss_list.append( nn.CrossEntropyLoss().cuda())
    if 'HEM' in loss_function.split(','):
        loss_list.append(HEM_CrossEntropy)
    if 'CW' in loss_function.split(','):
        loss_list.append(CW_CrossEntropy)
    if 'WeightCosine' in loss_function.split(','):
        loss_list.append(WeightCosine(model=kwargs['model']))
    if 'Logits' in loss_function.split(','):
        loss_list.append(Logits)
    
    if len(loss_list) == 0:
        raise NotImplementedError
    return LossListModule(loss_list)
