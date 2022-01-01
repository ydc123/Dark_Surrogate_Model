CUDA_VISIBLE_DEVICES=6,7 python train_imagenet.py -a resnet18 --savename SD_resnet18_cutmix_0.5\
    --arch_teacher resnet18 --cutmix --loss KD --alpha 0.5\
    --ckpt_teacher saved_models/resnet18_CE.pth.tar

