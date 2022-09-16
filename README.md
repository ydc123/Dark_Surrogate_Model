**FLAG**: I decide to release more codes in Octorber.

I have released three ResNet models on Baidu Netdist ([link, 链接](https://pan.baidu.com/s/1wSH8GKH6KlXJv5NSjpq-NQ) with password j7fg). They can be used as baselines for comparison. They includes:
* resnet18_CE, a ResNet18 model trained with cross-entropy loss function.
* SD_resnet18_cutmix, a dark ResNet18 trained with teacher model ''resnet18_CE''. The CutMix skill is used during training.
* SR_0.1_resnet18_cutmix, a dark ResNet18 trained with a slightly robust teacher model ''resnet18_l2_eps0.1'', which is the default model in paper "A Little Robustness Goes a Long Way: Leveraging Universal Features for Targeted Transfer Attacks". You can download the teacher model in [here](https://github.com/microsoft/robust-models-transfer).

Please feel free to contact us if you have any questions.

---

This repository is the official repository of our paper "Boosting the Adversarial Transferability of Surrogate Model with Dark Knowledge". We will release more codes as soon as possible.

You can refer to `imagenet/training/train_imagenet.py` to train your own dark surrogate model.

For example, you can train a normal ResNet18 as:
``
python train_imagenet.py -a resnet18 --savename resnet18_CE --loss CE
``

Then, you can train a dark ResNet18 by learning from the normal ResNet18 as:
``
python train_imagenet.py -a resnet18 --savename SD_resnet18_cutmix\
    --arch_teacher resnet18 --cutmix --loss KD \
    --ckpt_teacher saved_models/resnet18_CE.pth.tar
``
The CutMix skill is used in this example. You can also use other pretrained models as the teacher model by setting the input arguments ``arch_teacher`` and ``ckpt_teacher``.

Then, you can evaluate their adversarial transferability by generating adversarial examples based on the normal ResNet18 and the dark ResNet18, respectively. You can refer to any of the repositories for transfer-based attack. We will also release our codes for attacking as soon as possible.

The codes for the experiments on the face verification will also be released as soon as possible. We mainly refer to repository [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) to train the normal/dark surrogate models.

If you benefit from our work in your research, please consider to cite the following paper:
```
@article{yang2022boosting,
  title={Boosting the Adversarial Transferability of Surrogate Model with Dark Knowledge},
  author={Yang, Dingcheng and Xiao, Zihao and Yu, Wenjian},
  journal={arXiv preprint arXiv:2206.08316},
  year={2022}
}
```
