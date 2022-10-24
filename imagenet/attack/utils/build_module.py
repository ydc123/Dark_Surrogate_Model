import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import pretrainedmodels
import pretrainedmodels.utils as utils
from collections import OrderedDict

class ToBGR():
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, image):
        dim = self.dim
        if dim == 0:
            image = image[[2, 1, 0], :, :, :]
        elif dim == 1:
            image = image[:, [2, 1, 0], :, :]
        elif dim == 2:
            image = image[:, :, [2, 1, 0], :, :]
        elif dim == 3:
            image = image[ :, :, :, [2, 1, 0]]
        return image


class Normalize():
    def __init__(self, mean, std, range255):
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
        self.range255 = range255

    def __call__(self, image):
        dtype = image.type()
        N, C, H, W = image.size()
        if self.range255:
            image = image * 255
        mean = self.mean.clone().type(dtype).view(1, -1, 1, 1)
        std =  self.std.clone().type(dtype).view(1, -1, 1, 1)
        mean = mean.expand_as(image)
        std = std.expand_as(image)
        return image.sub(mean).div(std)


class ModuleList():
    def __init__(self, module_list):
        self.module_list = module_list

    def __call__(self, image):
        x = image
        for module in self.module_list:
            x = module(x)
        return x

    def to_list(self):
        return self.module_list

    def cuda(self):
        pass


def build_preprocess(input_range, input_mean, input_std, range255):
    process = []
    if input_range == 'BGR':
        process.append(ToBGR())
    if input_mean is None:
        input_mean = [0, 0, 0]
    if input_std is None:
        input_std = [1, 1, 1]

    process.append(Normalize(input_mean, input_std, range255))
    return ModuleList(process)


def build_pretrained_model(arch):
    model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained='imagenet').cuda()
    setting = pretrainedmodels.pretrained_settings[arch]['imagenet']
    print(setting)
    input_size = setting['input_size']
    if input_size[0] == 3:
        C, H, W = input_size
    else:
        H, W, C = input_size
    preprocess = build_preprocess(setting['input_space'], setting['mean'], setting['std'], range255=max(setting['input_range'])==255)
    return model, preprocess

def build_model(arch, path, num_classes=1000):
    if 'inception' in arch:
        model = torchvision.models.__dict__[arch](num_classes=num_classes, aux_logits=False)
    else:
        model = torchvision.models.__dict__[arch](num_classes=num_classes)
    info = torch.load(path, 'cpu')
    if 'state_dict' in info.keys(): # our models
        state_dict = info['state_dict']
    else: # Pretrained slightly robust model
        state_dict = info['model']
    cur_state_dict = model.state_dict()
    state_dict_keys = state_dict.keys()
    for key in cur_state_dict:
        if key in state_dict_keys:
            cur_state_dict[key].copy_(state_dict[key])
        elif key.replace('module.','') in state_dict_keys:
            cur_state_dict[key].copy_(state_dict[key.replace('module.','')])
        elif 'module.'+key in state_dict_keys:
            cur_state_dict[key].copy_(state_dict['module.'+key])
        elif 'module.attacker.model.'+key in state_dict_keys:
            cur_state_dict[key].copy_(state_dict['module.attacker.model.'+key])
    model.load_state_dict(cur_state_dict)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    preprocess = build_preprocess('RGB', mean, std, False)
    return model, preprocess
