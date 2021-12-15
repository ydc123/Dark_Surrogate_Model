CUDA_VISIBLE_DEVICES=2,3 python train_imagenet.py -a resnet18 --savename resnet18_cutout\
    --arch_teacher resnet50 --cutout --loss KD

