CUDA_VISIBLE_DEVICES=2,3 python train_imagenet.py -a resnet18 --savename SD_resnet18_BKD_0.6\
    --arch_teacher resnet18 --loss BKD --beta 0.6\
    --ckpt_teacher saved_models/resnet18_CE.pth.tar

