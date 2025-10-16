# oftta_edtn.py
import torch
import torch.nn as nn
from copy import deepcopy
import math

class WeightedBN1d(nn.Module):
    def __init__(self, bn: nn.BatchNorm1d, prior: float):
        super().__init__()
        assert 0.0 <= prior <= 1.0
        self.layer = bn
        self.layer.eval()
        self.prior = float(prior)
        self.running_mean = deepcopy(bn.running_mean.detach().clone())
        self.running_std  = deepcopy(torch.sqrt(bn.running_var.detach().clone() + bn.eps))

    def forward(self, x):  # x: (N, C, L)
        dim = (0, 2)
        batch_mean = x.mean(dim)
        batch_std  = torch.sqrt(x.var(dim, unbiased=False) + self.layer.eps)
        if self.running_mean is None:
            self.running_mean = batch_mean.detach().clone()
            self.running_std  = batch_std.detach().clone()
        mean = self.prior * self.running_mean + (1 - self.prior) * batch_mean.detach()
        std  = self.prior * self.running_std  + (1 - self.prior) * batch_std.detach()
        x = (x - mean[None, :, None]) / std[None, :, None]
        return x * self.layer.weight[None, :, None] + self.layer.bias[None, :, None]

class WeightedBN2d(nn.Module):
    def __init__(self, bn: nn.BatchNorm2d, prior: float):
        super().__init__()
        assert 0.0 <= prior <= 1.0
        self.layer = bn
        self.layer.eval()
        self.prior = float(prior)
        self.running_mean = deepcopy(bn.running_mean.detach().clone())
        self.running_std  = deepcopy(torch.sqrt(bn.running_var.detach().clone() + bn.eps))

    def forward(self, x):  # x: (N, C, H, W)
        dim = (0, 2, 3)
        batch_mean = x.mean(dim)
        batch_std  = torch.sqrt(x.var(dim, unbiased=False) + self.layer.eps)
        if self.running_mean is None:
            self.running_mean = batch_mean.detach().clone()
            self.running_std  = batch_std.detach().clone()
        mean = self.prior * self.running_mean + (1 - self.prior) * batch_mean.detach()
        std  = self.prior * self.running_std  + (1 - self.prior) * batch_std.detach()
        x = (x - mean[None, :, None, None]) / std[None, :, None, None]
        return x * self.layer.weight[None, :, None, None] + self.layer.bias[None, :, None, None]

def _gen_priors(num_bn, alpha0=0.7, lam=0.7):
    # alpha_i = 1 - alpha0 * exp(-lam * i)   （i 从 0 起，越深 alpha 越接近 1 ⇒ 更偏 CBN）
    return [float(1.0 - alpha0 * math.exp(-lam * i)) for i in range(num_bn)]

def patch_bn_to_edtn(model: nn.Module, alpha0=0.7, lam=0.7):
    # 收集 BN 顺序
    bns = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            bns.append(m)
    if not bns:
        return model  # 没有 BN，直接返回
    priors = _gen_priors(len(bns), alpha0, lam)
    # 逐个替换
    idx = 0
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.BatchNorm1d):
                setattr(parent, child_name, WeightedBN1d(child, priors[idx])); idx += 1
            elif isinstance(child, nn.BatchNorm2d):
                setattr(parent, child_name, WeightedBN2d(child, priors[idx])); idx += 1
    return model
def patch_bn_to_edtn(model: nn.Module, alpha0=0.7, lam=0.7, verbose=True):
    bns = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            bns.append(m)
    if not bns:
        if verbose:
            print("[EDTN] No BatchNorm found. Skipping.")
        return model
    priors = _gen_priors(len(bns), alpha0, lam)

    replaced = 0
    idx = 0
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.BatchNorm1d):
                setattr(parent, child_name, WeightedBN1d(child, priors[idx])); idx += 1; replaced += 1
            elif isinstance(child, nn.BatchNorm2d):
                setattr(parent, child_name, WeightedBN2d(child, priors[idx])); idx += 1; replaced += 1
    if verbose:
        print(f"[EDTN] Replaced {replaced} BN layers. prior[0]={priors[0]:.3f}, prior[-1]={priors[-1]:.3f}")
    return model
