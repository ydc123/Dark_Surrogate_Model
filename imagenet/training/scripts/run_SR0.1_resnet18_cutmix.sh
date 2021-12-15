CUDA_VISIBLE_DEVICES=6,7 python train_imagenet.py -a resnet18 --savename SR_0.1_resnet18_cutmix\
    --arch_teacher resnet50 --ckpt_teacher saved_models/resnet50_l2_eps0.1.ckpt --cutmix --loss KD

