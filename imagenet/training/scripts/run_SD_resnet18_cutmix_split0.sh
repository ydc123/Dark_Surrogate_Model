CUDA_VISIBLE_DEVICES=4,5,6,7 python train_imagenet_split.py -a resnet18 --savename SD_resnet18_cutmix_split0\
    --arch_teacher resnet18 --cutmix --loss KD \
    --ckpt_teacher saved_models/resnet18_CE.pth.tar \
    --version 0