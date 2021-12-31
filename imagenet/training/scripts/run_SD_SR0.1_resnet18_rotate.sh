CUDA_VISIBLE_DEVICES=4,5 python train_imagenet.py -a resnet18 --savename SD_SR_0.1_resnet18_rotate\
    --arch_teacher resnet18 --rotate --loss KD \
    --ckpt_teacher saved_models/resnet18_l2_eps0.1.pth.tar 
