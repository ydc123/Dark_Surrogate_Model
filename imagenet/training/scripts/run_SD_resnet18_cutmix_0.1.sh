CUDA_VISIBLE_DEVICES=0,1 python train_imagenet.py -a resnet18 --savename SD_resnet18_cutmix_0.1\
    --arch_teacher resnet18 --cutmix --loss KD --alpha 0.1\
    --ckpt_teacher saved_models/resnet18_CE.pth.tar

