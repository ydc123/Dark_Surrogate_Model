CUDA_VISIBLE_DEVICES=4,5 python train_imagenet.py -a shufflenet_v2_x1_0\
    --savename SR_0.1_shufflenet_v2_x1_0_cutmix --arch_teacher resnet50 --cutmix --loss KD \
    --ckpt_teacher saved_models/resnet50_l2_eps0.1.ckpt 

