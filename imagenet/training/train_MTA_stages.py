from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import warnings
import time
import random
import numpy as np
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from utils import AverageMeter, MTA_loss

from tqdm import tqdm
import shutil
import types
#from dataset import get_imagenet_iter_dali

model_names = sorted(name for name in torchvision.models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision.models.__dict__[name]))

def save_checkpoint(state, is_best, checkpoint, filename):
    filepath = os.path.join(checkpoint, filename + '.pth.tar')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, '{}_best.pth.tar'.format(filename)))
    
    return
def load_state(model, state_dict):
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
    return
if __name__=='__main__':
    paths = ['/data2/yangdingcheng/ILSVRC2012', '/data/yangdingcheng/ILSVRC2012', '/data/home/yangdingcheng/ILSVRC2012',
        '/data/data1/yangdingcheng/ILSVRC2012', '/home/yangdc/data/ILSVRC2012']
    for path in paths:
        if os.path.exists(path):
            imagenet_datapath = path
            break
    print('ImageNet Data Path: {}'.format(imagenet_datapath))
    parser = argparse.ArgumentParser(description='PyTorch ImageNet ResNet Example')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--seed', default=11037, type=int,
                    help='seed for initializing training. ')
    parser.add_argument('--max_iteration', default=100000, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--eps_c', default=3000, type=int)
    parser.add_argument('--attack_decay_iter', default=4000, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    default=False, help='evaluate model on validation set')
    parser.add_argument('--savename', type=str, default='demo')
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--ckpt_teacher', type=str, default=None)
    parser.add_argument('--arch_teacher', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--logname', type=str)
    parser.add_argument('--save_dir', default='saved_models/', type=str)
    args = parser.parse_args()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    nclass=1000
    traindir = os.path.join(imagenet_datapath,'train')
    testdir = os.path.join(imagenet_datapath,'val')

    transform_train = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()]
    transform_train = transforms.Compose(transform_train)
    trainset = torchvision.datasets.ImageFolder(root=traindir,transform=transform_train)

    max_iteration = args.max_iteration
    batch_size = args.batch_size
    eps_c = args.eps_c
    attack_decay_iter = args.attack_decay_iter

    model = torchvision.models.resnet18()
    model_teacher = torchvision.models.resnet18()
    if args.ckpt_teacher != None:
        info = torch.load(args.ckpt_teacher, 'cpu')
        if 'state_dict' in info.keys():
            load_state(model_teacher, info['state_dict'])
        else:
            load_state(model_teacher, info['model'])
    if args.pretrained != None:
        info = torch.load(args.pretrained, 'cpu')
        if 'state_dict' in info.keys():
            load_state(model, info['state_dict'])
        else:
            load_state(model, info['model'])
    model = model.cuda()
    if model_teacher != None:
        model_teacher = model_teacher.cuda()
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1:
        model = nn.DataParallel(model)
        if model_teacher != None:
            model_teacher = nn.DataParallel(model_teacher)
    log_file = open(f'logs/{args.logname}.txt', 'w')
    losses = []
    x = torch.randn(1, 3, 224, 224).cuda()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=8)
    num_classes = 1000
    bestacc = 0
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    model_teacher.eval()
    it = iter(trainloader)
    for batch_idx in tqdm(range(max_iteration)):
        try:
            inputs, targets = next(it)
        except StopIteration:
            it = iter(trainloader)
            inputs, targets = next(it)
        inputs, targets = inputs.cuda(), targets.cuda()
        alpha = eps_c / 255.0 * (0.9 ** (batch_idx // attack_decay_iter))
        loss = MTA_loss(model, model_teacher, inputs, targets,
            alpha)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 10)
        optimizer.step()
        log_file.write(f'{batch_idx}/{max_iteration}: {loss.item()}\n')
        log_file.flush()
        if batch_idx % 100 == 0:
            log_file.write('=====check if nan=====')
            isnan = False
            for param in model.parameters():
                if param.isnan().any():
                    isnan = True
            log_file.write(f'{isnan}\n')
        losses.append(loss.item())
        save_checkpoint({
                'iteration': batch_idx + 1,
                'state_dict': model.state_dict(),
                'loss': losses,
                'optimizer' : optimizer.state_dict(),
            }, False, checkpoint=args.save_dir, filename=args.savename)