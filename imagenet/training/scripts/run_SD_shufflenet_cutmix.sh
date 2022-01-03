CUDA_VISIBLE_DEVICES=0,1 python train_imagenet.py -a shufflenet_v2_x1_0 --savename SD_shufflenet_v2_x1_0_CE\
    --arch_teacher shufflenet_v2_x1_0 --cutmix --loss KD \
    --ckpt_teacher saved_models/shufflenet_v2_x1_0_CE.pth.tar

