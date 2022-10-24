CUDA_VISIBLE_DEVICES=2,3 python train_imagenet_split.py -a resnet18 --savename resnet18_CE_split1 \
    --loss CE --version 1

