CUDA_VISIBLE_DEVICES=2,3 python train_imagenet.py -a shufflenet_v2_x1_0 --savename shufflenet_v2_x1_0_cutout\
    --arch_teacher resnet50 --cutout --loss KD

