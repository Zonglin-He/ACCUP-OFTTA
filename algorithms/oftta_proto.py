# algorithms/oftta_proto.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class OFTTAPrototypeHead(nn.Module):
    """
    - warmup: 以线性分类器权重初始化 supports/labels/ents/cls_scores/conf
    - 在线: 低熵 + (可选) 双视图一致性筛选进入支持集, 各类保留 Top-K
    - 预测: 置信度加权的类中心 + 温度缩放的余弦相似度
    """
    def __init__(self,
                 classifier: nn.Linear,
                 num_classes: int,
                 filter_K: int = 10,
                 tau: float = 1.0,             # 建议默认 1.0
                 proto_scale: float = 20.0,     # 新增放大系数 s
                 conf_weight: bool = True):
        super().__init__()
        assert isinstance(classifier, nn.Linear), "OFTTAPrototypeHead 需要最后一层是 nn.Linear"

        self.classifier  = classifier
        self.num_classes = int(num_classes)
        self.filter_K    = int(filter_K)

        self.tau   = float(tau)           # 温度 T
        self.scale = float(proto_scale)   # 余弦放大系数 s
        self.conf_weight = bool(conf_weight)

        with torch.no_grad():
            # ---- warmup：用线性层权重当作初始支持 ----
            self.supports = self.classifier.weight.data.clone()           # (C, D)
            warmup_logits = self.classifier(self.supports)                # (C, C)
            p0            = warmup_logits.softmax(dim=1)                  # (C, C)
            self.labels   = F.one_hot(warmup_logits.argmax(1),
                                      num_classes=self.num_classes).float()
            self.ents       = -(p0 * warmup_logits.log_softmax(1)).sum(1) # (C,)
            self.cls_scores = p0                                          # (C, C)
            self.conf        = p0.max(dim=1).values                       # (C,) 置信度

    @staticmethod
    def _entropy(logits: torch.Tensor) -> torch.Tensor:
        p = logits.softmax(dim=1)
        return -(p * logits.log_softmax(dim=1)).sum(dim=1)

    def _select_supports(self):
        """各类保留 Top-K 低熵样本；所有缓存同步裁剪"""
        y_hat   = self.labels.argmax(1).long()
        device  = self.supports.device
        all_idx = torch.arange(self.supports.size(0), device=device)
        keep_chunks = []
        for c in range(self.num_classes):
            idx_c = all_idx[y_hat == c]
            if idx_c.numel() == 0:
                continue
            ent_c = self.ents[idx_c]
            topk  = min(self.filter_K, idx_c.numel())
            order = torch.argsort(ent_c)            # 低熵优先
            keep_chunks.append(idx_c[order[:topk]])
        if keep_chunks:
            keep = torch.cat(keep_chunks, dim=0)
            self.supports   = self.supports[keep]
            self.labels     = self.labels[keep]
            self.ents       = self.ents[keep]
            self.cls_scores = self.cls_scores[keep]
            self.conf       = self.conf[keep]

    def forward(self,
                feat: torch.Tensor,
                logits_raw: torch.Tensor = None,
                logits_aug: torch.Tensor = None,
                accept_by_consistency: bool = True,
                ent_thresh: float | torch.Tensor | None = None):
        """
        feat: (B,D) —— 建议传 z = (r_feat + a_feat)/2
        logits_raw/logits_aug: (B,C) —— 一致性门控（可选）
        ent_thresh: float/tensor 或 None —— 进入支持集的熵阈
        """
        # 1) 线性头产生伪标签/熵/softmax
        logits = self.classifier(feat)                      # (B, C)
        p      = logits.softmax(dim=1)                      # (B, C)
        yhat   = F.one_hot(logits.argmax(1), num_classes=self.num_classes).float()
        ent    = self._entropy(logits)                      # (B,)

        # 2) 低熵 +（可选）双视图一致性 → 纳入支持集
        with torch.no_grad():
            mask = torch.ones_like(ent, dtype=torch.bool)
            if ent_thresh is not None:
                mask &= (ent <= ent_thresh)
            if accept_by_consistency and (logits_raw is not None) and (logits_aug is not None):
                mask &= (logits_raw.argmax(1) == logits_aug.argmax(1))
            if mask.any():
                self.supports   = torch.cat([self.supports,   feat[mask]], dim=0)
                self.labels     = torch.cat([self.labels,     yhat[mask]], dim=0)
                self.ents       = torch.cat([self.ents,       ent[mask]],  dim=0)
                self.cls_scores = torch.cat([self.cls_scores, p[mask]],    dim=0)
                self.conf       = torch.cat([self.conf,       p[mask].max(1).values], dim=0)
                self._select_supports()

        # 3) 置信度加权的类原型（先归一化 supports，再做按类加权平均）
        S = F.normalize(self.supports, dim=1)                                  # (M, D)
        if self.conf_weight:
            w = self.conf.clamp_min(1e-6).unsqueeze(1)                         # (M,1)
            lab_w = self.labels * w                                            # (M,C)
        else:
            lab_w = self.labels
        denom = lab_w.sum(dim=0, keepdim=True).clamp_min_(1e-12)               # (1,C)
        centroids = (lab_w / denom).T @ S                                      # (C,D)
        centroids = F.normalize(centroids, dim=1)

        # 4) 温度缩放的余弦相似度
        feat_n = F.normalize(feat, dim=1)
        sim = feat_n @ centroids.T
        out    = (self.scale * sim) / max(self.tau, 1e-6)
        return out
