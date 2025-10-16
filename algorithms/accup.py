# algorithms/accup.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_tta_algorithm import BaseTestTimeAlgorithm, softmax_entropy
from loss.sup_contrast_loss import domain_contrastive_loss
from .oftta_proto import OFTTAPrototypeHead  # 原型分类器头

class ACCUP(BaseTestTimeAlgorithm):
    """
    ACCUP + OFTTA-Prototype
    - 双视图 raw/aug 前向
    - 原型头维护 supports/labels/ents/cls_scores（各类 Top-K 低熵，支持一致性门控）
    - 比较“原型 logits”与“基础平均 logits”的熵，逐样本选更低熵的预测
    """

    # 兼容不同调用方式的宽签名
    def __init__(self, configs, hparams=None, model=None, optimizer=None, **kwargs):
        if hparams is None:
            hparams = {}
        super().__init__(configs, hparams, model, optimizer)

        assert model is not None, "ACCUP 需要传入预训练模型 model"
        self.featurizer = model.feature_extractor
        self.classifier = model.classifier
        self.num_classes = int(configs.num_classes)

        # ---- 超参数（带默认值）----
        self.filter_K = int(hparams.get("filter_K", 10))
        self.tau = float(hparams.get("tau", 1.0))  # 原型温度 T（分母）
        self.proto_scale = float(hparams.get("proto_scale", 20.0))  # 原型余弦放大系数 s（分子）
        self.temperature = float(hparams.get("temperature", 0.7))  # 对比学习温度
        self.proto_consistency = bool(hparams.get("proto_consistency", True))
        self.proto_ent_q = float(hparams.get("proto_ent_q", 0.0))  # 0 表示不用熵阈

        # ---- 取线性分类层（兼容 classifier.logits 或直接 nn.Linear）----
        linear_head = getattr(self.classifier, "logits", None)
        if not isinstance(linear_head, nn.Linear):
            assert isinstance(self.classifier, nn.Linear), \
                "ACCUP 需要最终分类层为 nn.Linear 或在 classifier.logits 中包含 nn.Linear"
            linear_head = self.classifier

        # ---- 原型头 ----
        self.oftta_head = OFTTAPrototypeHead(
            classifier=linear_head,
            num_classes=self.num_classes,
            filter_K=self.filter_K,
            tau=self.tau,  # ← 新增传参
            conf_weight=True,
        )

        print(f"[ProtoHead] tau={self.tau}, proto_scale={self.proto_scale}")

        self._dbg_step = 0

    # 只读属性，方便 trainer 调用或调试
    @property
    def supports(self):
        return getattr(self.oftta_head, "supports", None)

    @property
    def labels(self):
        return getattr(self.oftta_head, "labels", None)

    @property
    def ents(self):
        return getattr(self.oftta_head, "ents", None)

    @property
    def cls_scores(self):
        return getattr(self.oftta_head, "cls_scores", None)

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        raw_data, aug_data = batch_data[0], batch_data[1]

        # 1) 双视图前向
        r_feat, _ = model.feature_extractor(raw_data)
        a_feat, _ = model.feature_extractor(aug_data)
        r_logits  = model.classifier(r_feat)
        a_logits  = model.classifier(a_feat)

        # 2) 增强集成（特征与基础 logits）
        z = (r_feat + a_feat) / 2.0
        base_logits = (r_logits + a_logits) / 2.0

        # 基础分支的熵
        base_ent = softmax_entropy(base_logits, base_logits)

        # 3) 批内分位数作为进入支持集的熵阈（可选）
        ent_thresh = None
        if self.proto_ent_q > 0.0:
            # quantile 返回标量 Tensor，原型头里也支持 float 或 Tensor
            ent_thresh = torch.quantile(base_ent.detach(), q=self.proto_ent_q)

        # 4) 原型头输出（内部执行：一致性/熵阈过滤 → 维护支持集 → 置信度加权类中心 → 温度缩放余弦相似度）
        proto_logits = self.oftta_head(
            z,
            logits_raw=r_logits,
            logits_aug=a_logits,
            accept_by_consistency=self.proto_consistency,
            ent_thresh=ent_thresh
        )

        # 5) 逐样本选择熵更低者（proto vs base）
        proto_ent = softmax_entropy(proto_logits, proto_logits)
        use_proto = (proto_ent < base_ent).unsqueeze(1)              # (B,1) -> 广播到 (B,C)
        select_logits = torch.where(use_proto, proto_logits, base_logits)

        # 6) 域对比损失（raw/aug/selected 三视图）
        with torch.no_grad():
            pseudo_labels = select_logits.argmax(1)
        cat_p = torch.cat([r_logits, a_logits, select_logits], dim=0)
        cat_pseudo = pseudo_labels.repeat(3)
        loss = domain_contrastive_loss(
            cat_p, cat_pseudo, temperature=self.temperature, device=z.device
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 7) 前几步打印调试信息（观察熵与支持池动态）
        self._dbg_step += 1
        if self._dbg_step <= 3:
            with torch.no_grad():
                supports = getattr(self.oftta_head, "supports", None)
                ents     = getattr(self.oftta_head, "ents", None)
                m = int(supports.size(0)) if supports is not None else -1
                pool_e = float(ents.mean().item()) if (ents is not None and ents.numel() > 0) else float("nan")
                pe = proto_ent.mean().item()
                be = base_ent.mean().item()
                qv = (ent_thresh.item() if isinstance(ent_thresh, torch.Tensor) else ent_thresh)
                print(f"[OFTTA] step={self._dbg_step} supports={m} "
                      f"pool_mean_ent={pool_e:.3f} mean_ent(proto)={pe:.3f} "
                      f"mean_ent(base)={be:.3f} ent_q={self.proto_ent_q} thr={qv}")

        return select_logits

    def configure_model(self, model):
        """
        只训练 backbone 里的 Conv1d（以及外部 Trainer 已做的 BN/EDTN 相关替换）。
        """
        model.train()
        model.requires_grad_(False)

        # 如果你希望 BN 的 affine 可学，这里也可放开：
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if getattr(module, "weight", None) is not None:
                    module.weight.requires_grad_(True)
                if getattr(module, "bias", None) is not None:
                    module.bias.requires_grad_(True)

        for name, module in model.feature_extractor.named_children():
            if name in ("conv_block1", "conv_block2", "conv_block3"):
                for sub in module.children():
                    if isinstance(sub, nn.Conv1d):
                        sub.requires_grad_(True)
        return model


# 可留作备用
def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


def softmax_kl_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction="none")
    return kl_div
