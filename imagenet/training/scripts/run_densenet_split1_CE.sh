CUDA_VISIBLE_DEVICES=0,1,2,3 python train_imagenet_split.py -a densenet121 \
    --savename densenet121_CE_split1 \
    --loss CE --version 1

