import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, center_only=True, eps=1e-4, affine=False, num_channels=1):
        super().__init__()
        self.center_only = center_only
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))
            self.beta  = nn.Parameter(torch.zeros(1, num_channels, 1))

    def forward(self, x):
        # 期望 x: [B, C, T]（若是 [B,T] 会在外面先 unsqueeze 成 [B,1,T]）
        if x.dim() == 2:
            x = x.unsqueeze(1)

        mu = x.mean(dim=2, keepdim=True)
        if self.center_only:
            x_norm = x - mu
        else:
            std = x.std(dim=2, keepdim=True, unbiased=False)
            std = torch.clamp(std, min=self.eps)
            x_norm = (x - mu) / std

        if self.affine:
            x_norm = x_norm * self.gamma + self.beta
        return x_norm
