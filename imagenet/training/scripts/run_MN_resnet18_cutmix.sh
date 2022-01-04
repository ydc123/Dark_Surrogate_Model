CUDA_VISIBLE_DEVICES=4,5 python train_imagenet.py -a resnet18 --savename MN_resnet18_cutmix\
    --arch_teacher mobilenet_v2 --cutmix --loss KD \
    --ckpt_teacher saved_models/mobilenet_v2_CE.pth.tar

