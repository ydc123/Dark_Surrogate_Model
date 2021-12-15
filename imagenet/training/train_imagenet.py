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
from utils import AverageMeter, CE_loss, KD_loss, LS_loss, SKD_loss

from tqdm import tqdm
import shutil
import types
#from dataset import get_imagenet_iter_dali

model_names = sorted(name for name in torchvision.models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision.models.__dict__[name]))

def save_state(model, accs, epoch, loss, args,optimizer, isbest):
    dirpath = 'saved_models/'
    suffix = '.pth.tar'
    state = {
            'acc': accs,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'isbest': isbest,
            'loss': loss,
            }
    filename = str(args.savename)+suffix
    torch.save(state,dirpath+filename)
    if isbest:
        shutil.copyfile(dirpath+filename, dirpath+'best.'+filename)
    
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


def adjust_learning_rate(optimizer, epoch, args):
    if epoch < 30:
        lr = args.lr
    elif epoch < 60:
        lr = args.lr  * 0.1
    elif epoch < 90:
        lr = args.lr * 0.01
    else:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def test(val_loader, model, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()


    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):
            images, target = images.cuda(), target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


    print('test_loss: {}, test_top1_acc: {}, test_top5_acc: {}'.format(losses.avg, top1.avg, top5.avg))


    return (top1.avg, top5.avg), losses.avg


def train(train_loader,optimizer, model, model_teacher, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    model.train()
    if model_teacher != None:
        model_teacher.eval()
    for i, (images, targets) in enumerate(tqdm(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)
        images, targets = images.cuda(), targets.cuda()
        if args.cutmix:
            images = cutmix_data(images)
        if args.mixup:
            images = mixup_data(images)
        outputs = model(images)
        if args.loss == 'KD':
            with torch.no_grad():
                t_outputs = model_teacher(images)
            loss = KD_loss(t_outputs, outputs)
        elif args.loss == 'CE':
            loss = CE_loss(outputs, targets)
        elif args.loss == 'LS':
            one_hot_label = F.one_hot(targets, num_classes=10).float()
            smooth_labels = one_hot_label * (1 - args.lambd_LS) + (1 - one_hot_label) * args.lambd_LS / 9
            loss = LS_loss(outputs, smooth_labels)
        elif args.loss == 'SKD':
            with torch.no_grad():
                t_outputs = model_teacher(images)
            loss = SKD_loss(t_outputs, outputs, targets)
        else:
            raise NotImplementedError

        # measure accuracy and record loss
        losses.update(loss.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # use gradients of Bi to update Wi
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def mixup_data(x, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    return x

class Cutout(object):
	def __init__(self, length):
		self.length = length

	def __call__(self, img):
		h, w = img.size(1), img.size(2)
		mask = np.ones((h, w), np.float32)
		y = np.random.randint(h)
		x = np.random.randint(w)

		y1 = np.clip(y - self.length // 2, 0, h)
		y2 = np.clip(y + self.length // 2, 0, h)
		x1 = np.clip(x - self.length // 2, 0, w)
		x2 = np.clip(x + self.length // 2, 0, w)

		mask[y1: y2, x1: x2] = 0.
		mask = torch.from_numpy(mask)
		mask = mask.expand_as(img)
		img *= mask
		return img
if __name__=='__main__':
    paths = ['/data2/yangdingcheng/ILSVRC2012', '/data/yangdingcheng/ILSVRC2012', '/data/home/yangdingcheng/ILSVRC2012',
        '/data/data1/yangdingcheng/ILSVRC2012', '/home/yangdc/ILSVRC2012']
    for path in paths:
        if os.path.exists(path):
            imagenet_datapath = path
            break
    print('ImageNet Data Path: {}'.format(imagenet_datapath))
    parser = argparse.ArgumentParser(description='PyTorch ImageNet ResNet Example')
    parser.add_argument('--epochs', type=int, default=90, metavar='N',
            help='number of epochs to train')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--pretrained', default=None, nargs='+',
            help='pretrained model ( for mixtest \
            the first pretrained model is the big one \
            and the sencond is the small net)')
    parser.add_argument('--seed', default=11037, type=int,
                    help='seed for initializing training. ')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    default=False, help='evaluate model on validation set')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('--savename', type=str, default='demo')
    parser.add_argument('--ckpt_teacher', type=str, default=None)
    parser.add_argument('--arch_teacher', type=str, default=None)
    parser.add_argument('--update', help='whether to update centers at each iter', action='store_true')
    parser.add_argument('--bs', help='batch size', type=int, default=256)
    parser.add_argument('--cutout', action='store_true',
                    default=False, help='whether to use cutout')
    parser.add_argument('--cutmix', action='store_true',
                    default=False, help='whether to use cutmix')
    parser.add_argument('--mixup', action='store_true',
                    default=False, help='whether to use mixup')
    parser.add_argument('--loss', type=str,
        help='loss function')

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

    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = [
        #transforms.Resize(256),
        #transforms.CenterCrop(args.crop),
        #transforms.RandomCrop(args.crop),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize]
    if args.cutout:
        transform_train.append(Cutout(112))
    transform_train = transforms.Compose(transform_train)
    trainset = torchvision.datasets.ImageFolder(root=traindir,transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs,
                                          shuffle=True, num_workers=8)
    width = 224 if 'inception' not in args.arch else 299
    print(width)
    testset = torchvision.datasets.ImageFolder(root=testdir,transform=
                                       transforms.Compose([
                                           transforms.Resize(256),
                                           transforms.CenterCrop(width),
                                           transforms.ToTensor(),
                                           normalize,
                                           ]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs,
                                         shuffle=False, num_workers=8)
    num_classes = 1000


    model_teacher = None
    model = torchvision.models.__dict__[args.arch]()
    if args.arch_teacher != None:
        model_teacher = torchvision.models.__dict__[args.arch_teacher](pretrained=True)
    bestacc = 0
    
    if args.ckpt_teacher != None:
        info = torch.load(args.ckpt_teacher, 'cpu')
        if 'state_dict' in info.keys():
            load_state(model_teacher, info['state_dict'])
        else:
            load_state(model_teacher, info['model'])
    model = model.cuda()
    if model_teacher != None:
        model_teacher = model_teacher.cuda()
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1:
        model = nn.DataParallel(model)
        if model_teacher != None:
            model_teacher = nn.DataParallel(model_teacher)
    optimizer = optim.SGD(model.parameters(), 
                lr=args.lr, momentum=args.momentum, weight_decay= args.weight_decay)

    if not args.pretrained:
        bestacc = 0
        accs = []
        losses = []
    else:
        pretrained_model = torch.load(args.pretrained)
        bestacc = max([x[0] for x in pretrained_model['acc']])
        accs = pretrained_model['acc']
        losses = pretrained_model['loss']
        args.start_epoch = pretrained_model['epoch']
        load_state(model, pretrained_model['state_dict'])
        optimizer.load_state_dict(pretrained_model['optimizer'])





    ''' evaluate model accuracy and loss only '''
    if args.evaluate:
        test(testloader, model, args.start_epoch, args)
        exit()

    ''' train model '''

    
    for epoch in tqdm(range(len(accs), args.epochs)):
        print(args)
        adjust_learning_rate(optimizer, epoch, args)
        print('epoch  =  {} learning rate = {}'.format(epoch, optimizer.param_groups[0]['lr']))
        train(trainloader, optimizer, model, model_teacher, epoch, args)
        acc, loss = test(testloader, model, epoch, args)
        accs.append(acc)
        losses.append(loss)
        isbest = acc[0] > bestacc
        if acc[0] > bestacc:
            bestacc = acc[0]
        save_state(model, accs, epoch, losses, args, optimizer, isbest)
        print('best acc so far:{:4.2f}'.format(bestacc))

