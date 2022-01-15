# CUDA_VISIBLE_DEVICES=6,7 python train_imagenet.py -a resnet18 --savename SD_SR_0.1_resnet18_cutmix\
#     --arch_teacher resnet18 --cutmix --loss KD \
#     --ckpt_teacher saved_models/resnet18_l2_eps0.1.pth.tar 

CUDA_VISIBLE_DEVICES=1,2,3,4 python train_imagenet.py -a resnet18\
    --savename SD_SR_0.1_resnet18_2G_cutmix\
    --arch_teacher resnet18 --cutmix --loss KD \
    --ckpt_teacher saved_models/SD_SR_0.1_resnet18_cutmix.pth.tar 

CUDA_VISIBLE_DEVICES=1,2,3,4 python train_imagenet.py -a resnet18\
    --savename SD_SR_0.1_resnet18_3G_cutmix\
    --arch_teacher resnet18 --cutmix --loss KD \
    --ckpt_teacher saved_models/SD_SR_0.1_resnet18_2G_cutmix.pth.tar 

CUDA_VISIBLE_DEVICES=1,2,3,4 python train_imagenet.py -a resnet18\
    --savename SD_SR_0.1_resnet18_4G_cutmix\
    --arch_teacher resnet18 --cutmix --loss KD \
    --ckpt_teacher saved_models/SD_SR_0.1_resnet18_3G_cutmix.pth.tar 

CUDA_VISIBLE_DEVICES=1,2,3,4 python train_imagenet.py -a resnet18\
    --savename SD_SR_0.1_resnet18_5G_cutmix\
    --arch_teacher resnet18 --cutmix --loss KD \
    --ckpt_teacher saved_models/SD_SR_0.1_resnet18_4G_cutmix.pth.tar 
