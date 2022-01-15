# CUDA_VISIBLE_DEVICES=1,2,3,4 python train_imagenet.py -a densenet121 --savename SD_densenet121_cutmix\
#     --arch_teacher densenet121 --cutmix --loss KD \
#     --ckpt_teacher saved_models/densenet121_CE.pth.tar


CUDA_VISIBLE_DEVICES=0,1,2,3 python train_imagenet.py -a densenet121 --savename SD_densenet121_2G_cutmix\
    --arch_teacher densenet121 --cutmix --loss KD\
    --ckpt_teacher saved_models/SD_densenet121_cutmix.pth.tar


CUDA_VISIBLE_DEVICES=0,1,2,3 python train_imagenet.py -a densenet121 --savename SD_densenet121_3G_cutmix\
    --arch_teacher densenet121 --cutmix --loss KD\
    --ckpt_teacher saved_models/SD_densenet121_2G_cutmix.pth.tar


CUDA_VISIBLE_DEVICES=0,1,2,3 python train_imagenet.py -a densenet121 --savename SD_densenet121_4G_cutmix\
    --arch_teacher densenet121 --cutmix --loss KD\
    --ckpt_teacher saved_models/SD_densenet121_3G_cutmix.pth.tar

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_imagenet.py -a densenet121 --savename SD_densenet121_5G_cutmix\
    --arch_teacher densenet121 --cutmix --loss KD\
    --ckpt_teacher saved_models/SD_densenet121_4G_cutmix.pth.tar

