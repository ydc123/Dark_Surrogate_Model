CUDA_VISIBLE_DEVICES=2,3 python train_imagenet.py -a resnet18 --savename SD_resnet18_BKD_0.8\
    --arch_teacher resnet18 --loss BKD --beta 0.8\
    --ckpt_teacher saved_models/resnet18_CE.pth.tar

