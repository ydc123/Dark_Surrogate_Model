CUDA_VISIBLE_DEVICES=6,7 python train_imagenet.py -a shufflenet_v2_x1_0 --savename shufflenet_v2_x1_0_cutmix\
    --arch_teacher resnet50 --cutmix --loss KD

