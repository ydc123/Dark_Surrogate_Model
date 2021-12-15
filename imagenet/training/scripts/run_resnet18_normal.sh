CUDA_VISIBLE_DEVICES=5 python train_imagenet.py -a resnet18 --savename resnet18_normal\
    --arch_teacher resnet50 --loss KD

