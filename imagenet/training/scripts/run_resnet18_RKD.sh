CUDA_VISIBLE_DEVICES=4,5 python train_imagenet.py -a resnet18 --savename resnet18_RKD\
    --arch_teacher resnet50 --loss RKD

