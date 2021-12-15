CUDA_VISIBLE_DEVICES=0,7 python train_imagenet.py -a resnet18 --savename resnet18_cutmix\
    --arch_teacher resnet50 --cutmix --loss KD

