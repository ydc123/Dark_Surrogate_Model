CUDA_VISIBLE_DEVICES=6,7 python train_imagenet.py -a shufflenet_v2_x1_0 --savename RN18_shufflenet_v2_x1_0_CE\
    --arch_teacher resnet18 --cutmix --loss KD \
    --ckpt_teacher saved_models/resnet18_CE.pth.tar

