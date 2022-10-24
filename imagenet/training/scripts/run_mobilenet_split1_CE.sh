CUDA_VISIBLE_DEVICES=4,5,6,7 python train_imagenet_split.py -a mobilenet_v2 \
    --savename mobilenet_v2_CE_split1 \
    --loss CE --version 1

