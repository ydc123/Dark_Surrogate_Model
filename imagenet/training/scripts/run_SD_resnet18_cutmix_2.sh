CUDA_VISIBLE_DEVICES=2,3 python train_imagenet.py -a resnet18 --savename SD_resnet18_cutmix_2\
    --arch_teacher resnet18 --cutmix --loss KD --alpha 2\
    --ckpt_teacher saved_models/resnet18_CE.pth.tar

