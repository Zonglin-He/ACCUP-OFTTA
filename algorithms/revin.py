# utils/revin.py
import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    RevIN：默认做“均值 + 方差”标准化（不是 center-only）。
    可选 affine 仿射恢复（一般金融数据不开也行）。
    输入期望 [B, C, T]，若是 [B, T] 会自动扩为 [B,1,T]。
    """
    def __init__(self, center_only: bool = False, eps: float = 1e-5,
                 affine: bool = False, num_channels: int = 1):
        super().__init__()
        self.center_only = bool(center_only)   # 建议传 False
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))
            self.beta  = nn.Parameter(torch.zeros(1, num_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B,T] -> [B,1,T]

        mu = x.mean(dim=2, keepdim=True)  # [..., T]
        if self.center_only:
            x_norm = x - mu
        else:
            std = x.std(dim=2, keepdim=True, unbiased=False).clamp_min(self.eps)
            x_norm = (x - mu) / std

        if self.affine:
            x_norm = x_norm * self.gamma + self.beta
        return x_norm
