CUDA_VISIBLE_DEVICES=4,5 python train_imagenet.py -a resnet18 --savename SD_resnet18_cutout\
    --arch_teacher resnet18 --cutout --loss KD \
    --ckpt_teacher saved_models/resnet18_CE.pth.tar

