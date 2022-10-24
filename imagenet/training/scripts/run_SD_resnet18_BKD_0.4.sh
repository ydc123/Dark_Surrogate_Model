CUDA_VISIBLE_DEVICES=4,5 python train_imagenet.py -a resnet18 --savename SD_resnet18_BKD_0.4\
    --arch_teacher resnet18 --loss BKD --beta 0.4\
    --ckpt_teacher saved_models/resnet18_CE.pth.tar

