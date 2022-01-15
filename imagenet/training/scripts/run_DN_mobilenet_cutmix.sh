CUDA_VISIBLE_DEVICES=0,1,2,3 python train_imagenet.py -a mobilenet_v2 --savename DN_mobilenet_v2_cutmix\
    --arch_teacher densenet121 --cutmix --loss KD \
    --ckpt_teacher saved_models/densenet121_CE.pth.tar

