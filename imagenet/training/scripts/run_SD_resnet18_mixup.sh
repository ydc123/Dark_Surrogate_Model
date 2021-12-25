CUDA_VISIBLE_DEVICES=2,3 python train_imagenet.py -a resnet18 --savename SD_resnet18_mixup\
    --arch_teacher resnet18 --mixup --loss KD \
    --ckpt_teacher saved_models/resnet18_CE.pth.tar

