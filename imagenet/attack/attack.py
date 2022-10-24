import os, sys
import torch
import random
import numpy as np
import torch.nn as nn
from torchvision import transforms
from utils.build_module import build_model
from utils.attacker import Attacker
from utils.dataset import load_images, save_images
from utils.input_diversity import input_diversity
from utils.loss import LossFactory
from options.attack_options import AttackOptions
import cv2

torch.multiprocessing.set_sharing_strategy('file_system')

opt = AttackOptions().parse()
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if opt.save_dir is None:
    opt.save_dir = '{}_{}_{}'.format(opt.method, opt.eps, opt.num_iter)
if not os.path.exists('{}/attack/'.format(opt.save_dir)):
    os.makedirs('{}/attack/'.format(opt.save_dir))
exit()


def main(arch_list, path_list):

    # model
    model_list = arch_list.split('+')
    path_list = path_list.split('+')
    models = []
    preprocesses = []
    for arch, path in zip(model_list, path_list):
        model, preprocess = build_model(arch, path)
        model = model.to(device)
        model.eval()
        models.append(model)
        preprocesses.append(preprocess)
        if 'inception' in arch:
            C, H, W = 3, 299, 299
        else:
            C, H, W = 3, 224, 224

    batch_shape = [opt.batch_size, C, H, W]

    # attack parameter
    multiplier = 1.0
    T = opt.num_iter
    alpha = opt.alpha / 255.


    attacker = Attacker(alpha=alpha, momentum=opt.momentum, targeted=opt.targeted, method=opt.attack_method)

    # loss function
    criterion = LossFactory(opt.loss_function, model=model) # nn.CrossEntropyLoss().cuda()

    anno_file = open(os.path.join(opt.save_dir, 'annotation.txt'), 'w')
    for i, (filenames, images, labels) in enumerate(load_images(opt.input_dir, batch_shape,
        opt.targeted, opt.img_num)):
        for filename, label in zip(filenames, labels):
            anno_file.write('{},{}\n'.format(filename, int(round(label))))
        inputs = torch.from_numpy(images).float().to(device)
        targets = torch.from_numpy(labels).long().to(device)
        input_images = inputs.clone()
        attack_images = inputs.clone()
        history_grad = 0
        for iter in range(T):
            cur_grad = 0
            for ens_iter in range(opt.ensemble_num):
                for model, preprocess in zip(models, preprocesses):
                    attack_images.requires_grad = True
                    model.zero_grad()
                    outputs = attack_images
                    if 'NI' in opt.attack_method:
                        outputs = outputs + alpha * opt.momentum * history_grad
                    outputs = preprocess(input_diversity(outputs * pow(opt.pow_base, -ens_iter), opt))
                    outputs = model(outputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    cur_grad += attack_images.grad.data
                    attack_images = attack_images.detach().clone()
                    for admix_iter in range(opt.m2_admix):
                        attack_images.requires_grad = True
                        model.zero_grad()
                        index = torch.randperm(opt.batch_size).cuda()
                        mixed_images = attack_images + opt.eta_admix * input_images[index, :].detach()
                        outputs = preprocess(input_diversity(mixed_images * pow(opt.pow_base, -ens_iter), opt))
                        outputs = model(outputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        cur_grad += attack_images.grad.data
                        attack_images = attack_images.detach().clone()


            cur_grad, history_grad = attacker.attack(cur_grad, history_grad)
            add_grad = torch.clamp(cur_grad + attack_images - input_images, -opt.eps / 255., opt.eps / 255.)
            attack_images = torch.clamp((input_images + add_grad).detach().clone(), 0, 1)

        noise = attack_images - input_images
        attack_images = input_images + noise * multiplier
        assert torch.abs(noise).cpu().data.numpy().max() * 255 <= opt.eps + 1e-3, 'Noise Exceeding EPS {}'.format(opt.eps)

        attack_images = attack_images.cpu().data.numpy()
        resized_images = []
        for idx in range(attack_images.shape[0]):
            resized_image = cv2.resize(attack_images[idx].transpose((1, 2, 0)), (299, 299))
            resized_image = resized_image.transpose([2, 0, 1])
            resized_images.append(resized_image)
        attack_images = np.array(resized_images)
        save_images(attack_images, filenames, '{}/attack/'.format(opt.save_dir))
        if (i + 1) % opt.print_interval == 0:
            print('{}/{}'.format((i+1) * opt.batch_size, opt.img_num))

if __name__ == '__main__':
    main(opt.arch, opt.path)
