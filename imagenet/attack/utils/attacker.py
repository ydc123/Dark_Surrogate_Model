import torch
from utils.input_diversity import GaussianSmooth


class Attacker():
    def __init__(self, alpha, momentum, targeted, method, kernel_size=21):
        self.alpha = alpha
        self.momentum = momentum
        self.targeted = targeted
        self.method = method
        self.kernel_size = kernel_size

    def attack(self, cur_grad, history_grad):
        if 'TI' in self.method:
            cur_grad = GaussianSmooth(cur_grad, self.kernel_size)
        grads = cur_grad / torch.abs(cur_grad).mean(dim=[1, 2, 3], keepdim=True)
        history_grad = history_grad * self.momentum + grads
        if 'FGSM' in self.method:
            noise = torch.sign(history_grad)
        else:
            noise = history_grad
        noise = self.alpha * noise
        if self.targeted:
            noise = -noise

        return noise, history_grad




