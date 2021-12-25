CUDA_VISIBLE_DEVICES=6,7 python train_imagenet.py -a resnet18 --savename self_resnet18_cutout\
    --arch_teacher resnet18 --cutout --loss KD

