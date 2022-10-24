CUDA_VISIBLE_DEVICES=0,1 python train_imagenet.py -a resnet18 --savename SD_resnet18_BKD_1.0\
    --arch_teacher resnet18 --loss BKD --beta 1.0\
    --ckpt_teacher saved_models/resnet18_CE.pth.tar

