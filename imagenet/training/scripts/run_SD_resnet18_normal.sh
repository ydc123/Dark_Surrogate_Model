CUDA_VISIBLE_DEVICES=0,1 python train_imagenet.py -a resnet18 --savename SD_resnet18_normal\
    --arch_teacher resnet18 --loss KD \
    --ckpt_teacher saved_models/resnet18_CE.pth.tar

