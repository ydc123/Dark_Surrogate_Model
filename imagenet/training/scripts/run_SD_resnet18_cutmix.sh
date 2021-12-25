CUDA_VISIBLE_DEVICES=0,1 python train_imagenet.py -a resnet18 --savename SD_resnet18_cutmix\
    --arch_teacher resnet18 --cutmix --loss KD \
    --ckpt_teacher saved_models/resnet18_CE.pth.tar

