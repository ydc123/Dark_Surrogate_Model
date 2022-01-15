# CUDA_VISIBLE_DEVICES=2,3,6,7 python train_imagenet.py -a mobilenet_v2 \
#     --savename SD_mobilenet_v2_cutmix \
#     --arch_teacher mobilenet_v2 --cutmix --loss KD \
#     --ckpt_teacher saved_models/mobilenet_v2_CE.pth.tar


CUDA_VISIBLE_DEVICES=4,5,6,7 python train_imagenet.py -a mobilenet_v2 \
    --savename SD_mobilenet_v2_2G_cutmix \
    --arch_teacher mobilenet_v2 --cutmix --loss KD\
    --ckpt_teacher saved_models/SD_mobilenet_v2_cutmix.pth.tar

CUDA_VISIBLE_DEVICES=4,5,6,7 python train_imagenet.py -a mobilenet_v2 \
    --savename SD_mobilenet_v2_3G_cutmix \
    --arch_teacher mobilenet_v2 --cutmix --loss KD\
    --ckpt_teacher saved_models/SD_mobilenet_v2_2G_cutmix.pth.tar

CUDA_VISIBLE_DEVICES=4,5,6,7 python train_imagenet.py -a mobilenet_v2 \
    --savename SD_mobilenet_v2_4G_cutmix \
    --arch_teacher mobilenet_v2 --cutmix --loss KD\
    --ckpt_teacher saved_models/SD_mobilenet_v2_3G_cutmix.pth.tar

CUDA_VISIBLE_DEVICES=4,5,6,7 python train_imagenet.py -a mobilenet_v2 \
    --savename SD_mobilenet_v2_5G_cutmix \
    --arch_teacher mobilenet_v2 --cutmix --loss KD\
    --ckpt_teacher saved_models/SD_mobilenet_v2_4G_cutmix.pth.tar
