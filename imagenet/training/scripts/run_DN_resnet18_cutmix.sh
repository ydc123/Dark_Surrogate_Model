CUDA_VISIBLE_DEVICES=6,7 python train_imagenet.py -a resnet18 --savename DN_resnet18_cutmix\
    --arch_teacher densenet121 --cutmix --loss KD \
    --ckpt_teacher saved_models/densenet121_CE.pth.tar

