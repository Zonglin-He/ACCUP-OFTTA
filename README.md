

# ACCUP + OFTTA: Test-Time Adaptation for Time-Series (HAR / EEG / FD)

**Augmented Contrastive Clustering with Uncertainty-Aware Prototyping + Optimization-Free TTA (PyTorch)**

> This repo integrates **ACCUP** with **OFTTA** for **source-free** test-time adaptation on time-series classification. Training remains unchanged; adaptation happens online on unlabeled target streams.

---

## Repository Layout

```
ACCUP-OFTTA/
├─ algorithms/
│  ├─ accup.py
│  ├─ base_tta_algorithm.py
│  ├─ get_tta_class.py       # selects TTA via --method
│  ├─ oftta_edtn.py          # EDTN
│  ├─ oftta_proto.py         # prototype-side adjustment
│  └─ revin.py               # RevIN (optional)
├─ configs/
│  ├─ data_model_configs.py
│  └─ tta_hparams_new.py
├─ dataloader/
│  ├─ augmentations.py
│  ├─ dataloader.py
│  └─ demo_dataloader.py     # stream loader (raw + aug)
├─ loss/sup_contrast_loss.py
├─ models/da_models.py
├─ optim/optimizer.py
├─ pre_train_model/{build.py, pre_train_model.py}
├─ trainers/{tta_abstract_trainer.py, tta_trainer.py}
├─ utils/utils.py
├─ data/      # datasets here
└─ results/   # logs & CSVs
```

---

## Environment

```bash
conda create -y -n accup_oftta python=3.9
conda activate accup_oftta
# if provided:
pip install -r requirements.txt
# otherwise:
pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn pandas einops tqdm matplotlib
```

---

## Data

Place datasets under `./data/` (e.g., `data/HAR`, `data/EEG`, `data/FD`). Keep your existing preprocessing (windowing, normalization, subject splits). CLI flags can override defaults if implemented.

---

## Quick Start

### A. Source Pretraining (ACCUP)

```bash
python -m trainers.tta_trainer \
  --dataset HAR \
  --data_path ./data \
  --save_dir ./results/accup_pretrain \
  --device cuda:0 \
  --stage pretrain
```

> If you use a separate pretraining script, replace this with your command.

### B. Test-Time Adaptation (ACCUP + OFTTA)

```bash
# Linux / macOS
python -m trainers.tta_trainer \
  --dataset HAR \
  --data_path ./data \
  --save_dir ./results/tta_experiments_logs \
  --device cuda:0 \
  --method oftta \
  --enable_accup \
  --revin \
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

**Key flags** (mirroring `configs/tta_hparams_new.py`):

* `--method oftta`: selects OFTTA via `algorithms/get_tta_class.py`
* `--enable_accup`: plug in ACCUP’s uncertainty-aware prototypes / contrastive clustering
* `--revin`: RevIN at the input
* `--target_domain`, `--continual`, etc., if supported by your trainer

---

## Protocols & Metrics

* **Leave-One-Out (cross-subject/device)** and **Continual TTA (single-pass streaming)**
* Report **Macro-F1** (robust to imbalance) and **Accuracy**; logs/CSVs are written to `./results/tta_experiments_logs/`.

---

## Method in a Nutshell

* **OFTTA** performs **EDTN** (exponential-decay test-time normalization) and **prototype-guided** prediction adjustment **without backprop**.
* **ACCUP** supplies **uncertainty-aware prototypes** and **contrastive clustering**, stabilizing pseudo-labels and support-set updates.

---

## Hyperparameters (see `configs/tta_hparams_new.py`)

* **ACCUP**: `filter_K` (per-class top-K low-entropy supports), `temperature` (SupCon), `tau` (similarity temp), `bn_only`, etc.
* **OFTTA**: EDTN decay/weights across depth, support-set size, sliding window/forgetting factor, etc.
* Tips: for tiny batches/streaming, consider **smaller LR**, **BN-only** updates, and **grad clipping**.

---

## FAQ

* **`torch.load(..., weights_only=False)` warning**: if files aren’t fully trusted, set `weights_only=True` and whitelist custom objects via `torch.serialization.add_safe_globals`. If all files are your own, it’s safe to ignore.
* **BN instability with small batches**: lower LR, enable BN-only, tune Top-K/support size, and/or enable RevIN.
* **No gain after enabling OFTTA**: confirm `--method oftta` dispatch in `get_tta_class.py`, and that switches in `oftta_edtn.py` / `oftta_proto.py` match your hparams.

---

## References

```bibtex
@article{ACCUP2024,
  title   = {Augmented Contrastive Clustering with Uncertainty-Aware Prototyping for Time Series Test-Time Adaptation},
  author  = {…},
  journal = {…},
  year    = {2024}
}

@inproceedings{OFTTA2023,
  title     = {Optimization-Free Test-Time Adaptation for Cross-Person Activity Recognition},
  author    = {Wang, Shuoyuan and Wang, Jindong and Xi, Huajun and Zhang, Bob and Zhang, Lei and Wei, Hongxin},
  booktitle = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)},
  year      = {2023}
}
```

---

## License

Recommend **MIT** or **Apache-2.0**. Keep third-party LICENSES and acknowledgments (e.g., RevIN, dataset tools).

---

如果你愿意，我也可以把两份 README **合并成一个带语言切换目录的单文件**，或把你的实际超参数（`tta_hparams_new.py` 里的默认值）自动展开成表格版参数说明。只要把你当前用的主要运行命令贴给我，我就把它们也补进“复现实验”小节。
