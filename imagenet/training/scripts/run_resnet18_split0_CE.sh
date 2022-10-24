CUDA_VISIBLE_DEVICES=0,1 python train_imagenet_split.py -a resnet18 --savename resnet18_CE_split0 \
    --loss CE --version 0

