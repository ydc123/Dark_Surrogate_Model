import os, sys
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import torch.nn as nn

#utils
from utils.dataset import load_images
from utils.build_module import build_pretrained_model


from options.eval_options import EvalOptions
import csv


opt = EvalOptions().parse()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def evaluate(models, preprocesses, accuracy):
    batch_shape = (opt.batch_size, 3, None, None) # hard code

    for k, v in models.items():
        accuracy[k] = 0
    for i, (filenames, images, label) in enumerate(load_images(os.path.join(opt.save_dir, 'attack'), batch_shape, opt.targeted, opt.img_num)):
        inputs = torch.from_numpy(images).float().to(device)
        targets = torch.from_numpy(label).long().to(device)
        for k in models.keys():
            outputs = inputs
            model = models[k]
            preprocess = preprocesses[k]
            with torch.no_grad():
                if outputs.shape[2] != model.input_size[1]:
                    outputs = nn.functional.interpolate(outputs, size=(model.input_size[1], model.input_size[1]), mode='nearest')
                outputs = model(preprocess(outputs))
                _, pred = outputs.max(1)

            if opt.eval_metric == 'cross_entropy':
                if opt.targeted:
                    accuracy[k] += pred.eq(targets).sum().item()
                else:
                    accuracy[k] += targets.size(0) - pred.eq(targets).sum().item()


    return accuracy

if __name__ == '__main__':
    models = {}
    preprocesses = {}
    accuracy = {}
    for arch in opt.model_list.split(','):
        model, preprocess = build_pretrained_model(arch)
        model.to(device)
        model.eval()
        models[arch] = model
        preprocesses[arch] = preprocess
    accuracy = evaluate(models, preprocesses, accuracy)
    del models
    del preprocesses
    tf_arches = [arch for arch in opt.model_list.split(',')
                if arch in ['AdvInceptionV3', 'Ens3AdvInceptionV3', 'Ens4AdvInceptionV3', 'EnsAdvInceptionResnetV2']]

    model_names = []
    success_samples = []
    for k, v in accuracy.items():
        print('{} {}/{}'.format(k, v, opt.img_num))
        model_names.append(k)
        success_samples.append(v)

    with open(opt.csv_file, 'w', encoding='utf-8') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(model_names)
        csv_write.writerow(success_samples)
