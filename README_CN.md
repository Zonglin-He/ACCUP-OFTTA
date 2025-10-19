# 🇨🇳 README（中文）

# ACCUP + OFTTA：时间序列测试时自适应（TTA）在 HAR / EEG / FD 上的实现

**Augmented Contrastive Clustering with Uncertainty-Aware Prototyping + Optimization-Free TTA（PyTorch）**

> 本仓库将 **ACCUP**（不确定性感知原型 + 对比聚类）与 **OFTTA**（无反传、优化开销极低的测试时自适应）进行集成。无需改动训练阶段，仅在测试阶段对目标域的未标注数据流进行在线适配。已在 **HAR / EEG / FD** 等时间序列任务中验证。

---

## 目录结构

```
ACCUP-OFTTA/
├─ algorithms/
│  ├─ accup.py                 # ACCUP 主体
│  ├─ base_tta_algorithm.py    # TTA 抽象基类/公共工具
│  ├─ get_tta_class.py         # 通过 --method 选择 TTA 实现
│  ├─ oftta_edtn.py            # OFTTA: EDTN（测试时指数衰减归一化）
│  ├─ oftta_proto.py           # OFTTA: 原型/支持集侧的预测调整
│  └─ revin.py                 # 可逆实例归一化 RevIN（可选）
├─ configs/
│  ├─ data_model_configs.py    # 各数据集/模型的配置
│  └─ tta_hparams_new.py       # 超参数（ACCUP / OFTTA）
├─ dataloader/
│  ├─ augmentations.py         # 时序增强（多视图）
│  ├─ dataloader.py            # 常规加载器
│  └─ demo_dataloader.py       # TTA 流式加载器（raw + aug）
├─ loss/
│  └─ sup_contrast_loss.py     # SupCon 对比损失
├─ models/
│  ├─ da_models.py             # 主干网络（CNN/ResNet/…）
│  └─ loss.py
├─ optim/
│  └─ optimizer.py             # 构建优化器（如需）
├─ pre_train_model/
│  ├─ build.py
│  └─ pre_train_model.py       # 源模型预训练封装
├─ trainers/
│  ├─ tta_abstract_trainer.py  # Trainer 抽象
│  └─ tta_trainer.py           # 训练/测试时自适应入口
├─ utils/
│  └─ utils.py
├─ data/                       # 放置数据集
├─ results/                    # 日志与结果输出
└─ README.md
```

---

## 环境与安装

* Python ≥ 3.9
* PyTorch ≥ 2.0（建议 CUDA 版本）
* 其他依赖：`numpy, scipy, scikit-learn, pandas, einops, tqdm, matplotlib`

```bash
# 推荐使用 Conda
conda create -y -n accup_oftta python=3.9
conda activate accup_oftta

# 若仓库提供 requirements.txt：
pip install -r requirements.txt
# 或手动安装：
pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn pandas einops tqdm matplotlib
```

---

## 数据准备

将数据放入 `./data/`。示例（按你本地命名习惯调整）：

```
data/
├─ HAR/            # 或 Dataset/HAR 等
├─ EEG/
└─ FD/
```

> 若需滑窗/标准化/主体划分等，请沿用仓库现有的处理脚本与默认参数；也可通过命令行参数覆盖。

---

## 快速开始

### A. 预训练源模型（ACCUP）

保持你现有 ACCUP 训练流程，无需为 TTA 改动目标函数：

```bash
python -m trainers.tta_trainer \
  --dataset HAR \
  --data_path ./data \
  --save_dir ./results/accup_pretrain \
  --device cuda:0 \
  --stage pretrain
```

> 一些实现会将「预训练」和「TTA」写在同一入口里，通过 `--stage` 或内部开关区分；如你使用单独脚本，请替换为对应命令。

### B. 测试时自适应（ACCUP + OFTTA）

```bash
# Linux / macOS
python -m trainers.tta_trainer \
  --dataset HAR \
  --data_path ./data \
  --save_dir ./results/tta_experiments_logs \
  --device cuda:0 \
  --method oftta \                # 使用 algorithms/get_tta_class.py 选择 OFTTA
  --enable_accup \                # 启用 ACCUP 的原型/聚类增强
  --revin \                       # 可选：RevIN 稳定非平稳序列
  --num_runs 1

# Windows PowerShell
python -m trainers.tta_trainer `
  --dataset HAR `
  --data_path .\data `
  --save_dir .\results\tta_experiments_logs `
  --device cpu `
  --method oftta `
  --enable_accup `
  --revin `
  --num_runs 1
```

常用开关（与你的 `configs/tta_hparams_new.py` 一一对应）：

* `--method oftta`：启用 **EDTN + 原型调整** 的 OFTTA（见 `oftta_edtn.py` / `oftta_proto.py`）
* `--enable_accup`：接入 **不确定性感知原型** 与 **对比聚类**（ACCUP）以稳健伪标签/支持集
* `--revin`：输入端使用 **RevIN**
* 其他诸如 `--target_domain / --continual` 等，请按你的脚本支持情况添加

---

## 评价协议与指标

* **Leave-One-Out（跨主体）**：不同主体/设备作为不同域，留一做目标域，汇报 Macro-F1 / Acc 平均
* **Continual TTA（流式单遍）**：顺序适配测试流，关注遗忘与稳定性

默认在 `./results/tta_experiments_logs/` 生成日志与 CSV；末行可包含 mean/std 聚合。

---

## 方法要点（一段话）

* **OFTTA** 不进行反向传播，通过 **EDTN** 在不同层次上权衡 TBN/CBN 统计，并利用**支持集-原型距离**对分类器输出进行调整；
* **ACCUP** 提供更稳的伪标签与支持集更新（不确定性感知原型 + 对比聚类），降低伪标签噪声对原型的污染。

---

## 关键超参数（在 `configs/tta_hparams_new.py`）

* **ACCUP**：`filter_K`（每类支持集 Top-K 低熵样本）、`temperature`（SupCon）、`tau`（相似度温度）、`bn_only` 等
* **OFTTA**：EDTN 衰减系数/深度权重、支持集规模、滑动窗口/遗忘因子等（与你实现的字段保持一致）
* 建议：小批量/流式场景下可 **减小 LR**，或使用 **BN-only** 更新，必要时加 **grad clip**

---

## 常见问题

* **`torch.load(..., weights_only=False)` 警告**：若加载文件不完全可信，建议设为 `weights_only=True` 并通过 `torch.serialization.add_safe_globals` 白名单自定义对象；若文件完全来自本地/自建，则可忽略。
* **小批量导致 BN 不稳**：降低 LR、仅更新 BN、缩小支持集/增大 Top-K 稳定度，或启用 RevIN。
* **结果不提升**：确认 `--method oftta` 已生效；检查 `get_tta_class.py` 的分发、以及 `oftta_edtn.py / oftta_proto.py` 中开关是否与超参数一致。

---

## 引用

```bibtex
@article{Gong2025ACCUParXiv,
  author  = {Peiliang Gong and Mohamed Ragab and Min Wu and Zhenghua Chen and Yongyi Su and Xiaoli Li and Daoqiang Zhang},
  title   = {Augmented Contrastive Clustering with Uncertainty-Aware Prototyping for Time Series Test Time Adaptation},
  journal = {arXiv preprint arXiv:2501.01472},
  year    = {2025}
}

@inproceedings{OFTTA2023,
  title     = {Optimization-Free Test-Time Adaptation for Cross-Person Activity Recognition},
  author    = {Shuoyuan Wang and Jindong Wang and Huajun Xi and Bob Zhang and Lei Zhang and Hongxin Wei},
  booktitle = {Proc. ACM IMWUT/UbiComp},
  year      = {2023}
}
```

---

## 许可

本项目建议使用 **MIT** 或 **Apache-2.0**。如复用第三方代码（如 RevIN / 数据处理脚本等），请保留其 LICENSE 与致谢。

---

