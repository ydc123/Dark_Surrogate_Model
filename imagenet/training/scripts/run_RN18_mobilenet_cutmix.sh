CUDA_VISIBLE_DEVICES=1,2,3,4 python train_imagenet.py -a mobilenet_v2 --savename RN18_mobilenet_v2_cutmix\
    --arch_teacher resnet18 --cutmix --loss KD \
    --ckpt_teacher saved_models/resnet18_CE.pth.tar

