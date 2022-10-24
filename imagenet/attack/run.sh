CUDA_VISIBLE_DEVICES=0 python attack.py --save_dir /data1/yangdc/output_imagenet_benchmark/16_DIFGSM_untargeted \
    --batch_size 20 \
    --num_iter 10 \
    --targeted False \
    --momentum 1.0 \
    --arch resnet18 \
    --loss CE \
    --path ../training/saved_models/resnet18_CE.pth.tar \
    --diversity_method pad \
    --diversity_prob 0.7 \
    --attack_method DIFGSM \
    --alpha 2 \
    --eps 16
TORCH_HOME=/data1/yangdc CUDA_VISIBLE_DEVICES=0 python evaluate.py --save_dir /data1/yangdc/output_imagenet_benchmark/16_DIFGSM_untargeted \
    --targeted False \
    --model_list densenet201 \
    --csv_file results.csv \