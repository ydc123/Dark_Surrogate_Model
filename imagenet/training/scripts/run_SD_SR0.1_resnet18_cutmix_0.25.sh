CUDA_VISIBLE_DEVICES=0,1 python train_imagenet.py -a resnet18 --savename SD_SR_0.1_resnet18_cutmix_0.25\
    --arch_teacher resnet18 --cutmix --loss KD --alpha 0.25\
    --ckpt_teacher saved_models/resnet18_l2_eps0.1.pth.tar 

