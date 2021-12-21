CUDA_VISIBLE_DEVICES=2,3,4,5 python train_imagenet.py -a mobilenet_v2 --savename mobilenet_v2_cutmix\
    --arch_teacher resnet50 --cutmix --loss KD --resume saved_models/mobilenet_v2_cutmix.pth.tar

