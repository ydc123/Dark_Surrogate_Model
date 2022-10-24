import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn as nn
from torch.autograd import Function


def CutOut(length=16, img_shape=(112, 112)):
    h, w = img_shape
    mask = np.ones((h, w), np.float32)
    x, y = np.random.randint(w), np.random.randint(h)
    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x - length // 2, 0, w)
    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    mask[y1:y2, x1:x2] = 0.
    return mask


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel



def GaussianSmooth(x, kernel_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if kernel_size == 1:
        return x
    kernel = gkern(kernel_size, np.random.randint(3)*2+3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3)
    weight = torch.from_numpy(stack_kernel).permute([2, 3, 0, 1]).to(device)
    padv = (kernel_size - 1) // 2
    x = nn.functional.pad(x, pad=(padv, padv, padv, padv), mode='replicate')
    src = nn.functional.conv2d(
        x, weight, bias=None, stride=1, padding=0,
        dilation=1, groups=3)
    return src


def affine(x, vgrid, device='cuda'):
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(device)
    mask = nn.functional.grid_sample(mask, vgrid)
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1
    return output * mask


def warp(x, flo, device='cuda'):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = flo.size()
    H_in, W_in = x.size()[-2:]
    vgrid = torch.rand((B, 2, H, W)).to(device)
    # mesh grid


    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * flo[:, 0, :, :].clone() / max(W_in - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * flo[:, 1, :, :].clone() / max(H_in - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    return affine(x, vgrid)


def WarpPerspective(x, tmatrix, out_H=None, out_W=None, dstsize=None, device='cuda', inverse=False):
    '''
    formulation: http://www.cnblogs.com/zipeilu/p/6138423.html
    input:
        x(torch.tensor): NxCxHxW
        tmatrix(numpy.array or list): 3x3
    output:
        warp_res(torch.tensor): NxCxHxW
    '''

    assert (len(x.size()) == 4)

    if inverse:
        tmatrix = np.linalg.inv(tmatrix)
    H, W = x.size()[2:]
    if out_H is None and out_W is None:
        out_H, out_W = H, W
    if dstsize is not None:
        out_H, out_W = dstsize

    flow = torch.zeros(2, out_H, out_W).to(device)
    identity = torch.ones(out_H, out_W).to(device)
    xx = torch.arange(0, out_W).view(1, -1).repeat(out_H, 1).type_as(identity).to(device)
    yy = torch.arange(0, out_H).view(-1, 1).repeat(1, out_W).type_as(identity).to(device)
    _A = (tmatrix[1][1] - tmatrix[2][1] * yy)
    _B = (tmatrix[2][2] * xx - tmatrix[0][2])
    _C = (tmatrix[0][1] - tmatrix[2][1] * xx)
    _D = (tmatrix[2][2] * yy - tmatrix[1][2])
    _E = (tmatrix[0][0] - tmatrix[2][0] * xx)
    _F = (tmatrix[1][0] - tmatrix[2][0] * yy)
    xa = _A * _B - _C * _D
    xb = _A * _E - _C * _F
    ya = _F * _B - _E * _D
    yb = _F * _C - _E * _A
    flow[0] = xa / xb
    flow[1] = ya / yb
    flow = flow.view(1, 2, out_H, out_W).repeat(x.size(0), 1, 1, 1)
    return warp(x, flow, device=device)


class WarpFunction(Function):

    @staticmethod
    def forward(ctx, input, matrix, dstsize=None):
        ctx.save_for_backward(input, torch.from_numpy(matrix))
        return WarpPerspective(input, matrix, dstsize=dstsize)


    @staticmethod
    def backward(ctx, grad_output):
        input, matrix = ctx.saved_variables
        dstsize = input.size()[-2:]
        return WarpPerspective(grad_output, matrix.cpu().numpy(), dstsize=dstsize, inverse=True), None, None



def Resize(x, device='cuda'):
    '''
    input:
        x: (N, 299, 299)
    output:
        (N, 224, 224)
    '''
    scale_factor = 2.0 / 223
    N = x.size(0)
    grid = torch.zeros((N, 224, 224, 2))
    grid[:, :, :, 0] = torch.arange(0, 224, dtype=torch.float32).view((1, 1, 224)).repeat(N, 224, 1) * scale_factor - 1
    grid[:, :, :, 1] = torch.arange(0, 224, dtype=torch.float32).view((1, 224, 1)).repeat(N, 1, 224) * scale_factor - 1
    grid = grid.to(device)
    x = x.to(device)
    return affine(x, grid, device=device)

def RandomCrop(x, device='cuda'):
    '''
    input:
        x: (N, 299, 299)
    output:
        (N, 224, 224)
    '''
    scale_factor = 2.0 / 223
    N = x.size(0)
    grid = torch.zeros((N, 224, 224, 2))
    start = torch.randint(0, (299 - 224) / 2, (N, 2))
    sx = start[:, 0].view(N, 1, 1).float()
    sy = start[:, 1].view(N, 1, 1).float()
    grid[:, :, :, 0] = (sx + torch.arange(0, 224, dtype=torch.float32).view((1, 1, 224)).repeat(N, 224, 1)) * scale_factor - 1
    grid[:, :, :, 1] = (sy + torch.arange(0, 224, dtype=torch.float32).view((1, 224, 1)).repeat(N, 1, 224)) * scale_factor - 1
    grid = grid.to(device)
    x = x.to(device)
    return affine(x, grid, device=device)

def Resize_and_padding(x, scale_factor):
    ori_size = x.size()[-2:]
    x = nn.functional.interpolate(x, scale_factor=scale_factor)
    new_size = x.size()[-2:]

    delta_w = ori_size[1] - new_size[1]
    delta_h = ori_size[0] - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    x = nn.functional.pad(x, pad=(left,right,top,bottom), value=255)
    return x

def Rotate(x, theta, device='cuda'):
    rotation = np.zeros((2, 3, x.size(0)))
    cos = np.cos(theta).ravel()
    sin = np.sin(theta).ravel()
    rotation[0, 0] = cos
    rotation[0, 1] = sin
    rotation[1, 0] = -sin
    rotation[1, 1] = cos
    rotation = torch.Tensor(rotation.transpose((2, 0, 1))).to(device)
    grid = torch.nn.functional.affine_grid(rotation, size=x.size())
    return affine(x, grid, device)

def padding(x, new_size):
    ori_size = x.size(-1)
    delta_w = new_size - ori_size
    delta_h = new_size - ori_size
    top = random.randint(0, delta_h)
    left = random.randint(0, delta_w)
    bottom = delta_h - top
    right = delta_w - left
    return nn.functional.pad(x, pad=(left, right, top, bottom), value=0)


def input_diversity(x, opt):
    if 'DI' in opt.attack_method:
        if random.uniform(0, 1) < opt.diversity_prob:
            if 'pad' in opt.diversity_method:
                size = x.size(2)+31
                x = padding(x, size)
            if 'rotate' in opt.diversity_method:
                n = x.size(0)
                theta = np.random.normal(scale=opt.rotate_std, size=(n, 1))
                x = Rotate(x, theta)
    return x


