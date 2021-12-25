CUDA_VISIBLE_DEVICES=6,7 python train_imagenet.py -a resnet18 --savename SD_resnet18_rotate\
    --arch_teacher resnet18 --rotate --loss KD \
    --ckpt_teacher saved_models/resnet18_CE.pth.tar

