CUDA_VISIBLE_DEVICES=6,7 python train_imagenet.py -a resnet18 --savename resnet18_LS\
    --loss LS --lambd_LS 0.1

