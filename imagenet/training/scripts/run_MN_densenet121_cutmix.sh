CUDA_VISIBLE_DEVICES=4,5,6,7 python train_imagenet.py -a densenet121\
    --savename MN_densenet121_cutmix\
    --arch_teacher mobilenet_v2 --cutmix --loss KD \
    --ckpt_teacher saved_models/mobilenet_v2_CE.pth.tar

