CUDA_VISIBLE_DEVICES=0,1,2,3 python train_imagenet.py -a densenet121 --savename RN_densenet121_cutmix\
    --arch_teacher resnet18 --cutmix --loss KD \
    --ckpt_teacher saved_models/resnet18_CE.pth.tar

