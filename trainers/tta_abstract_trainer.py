import sys

sys.path.append('../../ADATIME/')
import torch
import torch.nn.functional as F
import os
import wandb
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
from pathlib import Path
import datetime as dt
import os

from torchmetrics import Accuracy, AUROC, F1Score
from dataloader.dataloader import data_generator, whole_targe_data_generator
from dataloader.demo_dataloader import data_generator_demo, whole_targe_data_generator_demo
from configs.data_model_configs import get_dataset_class
from configs.tta_hparams_new import get_hparams_class
from algorithms.get_tta_class import get_algorithm_class

from models.da_models import get_backbone_class
from pre_train_model.pre_train_model import PreTrainModel
from pre_train_model.build import pre_train_model
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


class TTAAbstractTrainer(object):
    """
    主要功能：实现不同的训练算法
    1. 初始化模型和数据
    2. 预训练模型
    3. 评估模型
    4. 计算指标和风险
    5. 保存和加载检查点
    6. 日志记录和结果保存
    7. 辅助功能：创建保存目录，获取配置等
    8. 处理不同的数据加载器
    """
    def __init__(self, args): #初始化
        self.da_method = args.da_method 
        self.dataset = args.dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)

        self.run_description = f"{args.da_method}_{args.exp_name}"
        self.experiment_description = args.dataset

        self.home_path = os.path.dirname(os.getcwd())
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.num_runs = args.num_runs
        self.dataset_configs, self.hparams_class = self.get_configs()
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels
        self.hparams = {**self.hparams_class.alg_hparams[self.da_method], **self.hparams_class.train_params}

        self.num_classes = self.dataset_configs.num_classes
        # 准备评估指标：Accuracy（多分类），宏F1，AUROC（多分类）
        self.ACC = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.F1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.AUROC = AUROC(task="multiclass", num_classes=self.num_classes)

        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_log_dir = str((Path(self.save_dir).resolve() / self.dataset / f"{self.da_method}_{ts}"))
        Path(self.exp_log_dir).mkdir(parents=True, exist_ok=True)

    def sweep(self):
        pass

    def initialize_pretrained_model(self): #初始化预训练模型
        backbone_fe = get_backbone_class(self.backbone) #获取backbone
        pretrained_model = PreTrainModel(backbone_fe, self.dataset_configs, self.hparams) #预训练模型
        pretrained_model = pretrained_model.to(self.device)

        return pretrained_model

    def pre_train(self): #预训练
        backbone_fe = get_backbone_class(self.backbone)
        # pretraining step
        self.logger.debug(f'Pretraining stage..........')
        self.logger.debug("=" * 45)
        non_adapted_model_state, pre_trained_model = pre_train_model(backbone_fe, self.dataset_configs, self.hparams, self.src_train_dl, self.pre_loss_avg_meters, self.logger, self.device)

        return non_adapted_model_state, pre_trained_model

    def evaluate(self, test_loader, tta_model):
        total_loss, preds_list, labels_list = [], [], []
        tta_model.eval()  # 评估态
        with torch.no_grad():  # 关梯度
            for data, labels, trg_idx in test_loader:
                if isinstance(data, list):
                    data = [d.float().to(self.device) for d in data]
                else:
                    data = data.float().to(self.device)
                labels = labels.view(-1).long().to(self.device)

                if hasattr(self, "_preprocess_inputs") and callable(getattr(self, "_preprocess_inputs")):
                    data = self._preprocess_inputs(data)

                predictions = tta_model(data)
                loss = F.cross_entropy(predictions, labels)
                total_loss.append(loss.item())
                preds_list.append(predictions)
                labels_list.append(labels)

        self.loss = torch.tensor(total_loss).mean()
        self.full_preds = torch.cat(preds_list)
        self.full_labels = torch.cat(labels_list)

    def get_configs(self): #获取当前数据集对应的配置类实例和超参数类实例
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    def get_tta_model_class(self): #获取指定的 TTA 模型类
        tta_model_class = get_algorithm_class(self.da_method)

        return tta_model_class

    def load_data(self, src_id, trg_id): # 加载数据集（不带增强版本）
        self.src_train_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "train")
        self.src_test_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "test")

        self.trg_train_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "train")
        self.trg_test_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "test")

        self.trg_whole_dl = whole_targe_data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams)

    def load_data_demo(self, src_id, trg_id, run_id = 0): #加载数据集（带增强版本）
        self.src_train_dl = data_generator_demo(self.data_path, src_id, self.dataset_configs, self.hparams, "train")
        self.src_test_dl = data_generator_demo(self.data_path, src_id, self.dataset_configs, self.hparams, "test")
        self.trg_whole_dl = whole_targe_data_generator_demo(self.data_path, trg_id, self.dataset_configs, self.hparams, seed_id = run_id)

    def create_save_dir(self, save_dir): #创建保存目录
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def calculate_metrics_risks(self, tta_model):
        # 源/目标测试集风险
        self.evaluate(self.src_test_dl, tta_model)
        src_risk = self.loss.item()
        self.evaluate(self.trg_test_dl, tta_model)
        trg_risk = self.loss.item()

        # 统一用 evaluate 后缓存的预测/标签计算指标
        acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()

        return (src_risk, trg_risk), (acc, f1, auroc)

    def save_tables_to_file(self, table_results: pd.DataFrame, name: str):
        out = Path(self.exp_log_dir) / name
        csv_path = out.with_suffix(".csv")

        # 一定能成功的 CSV
        table_results.to_csv(csv_path, index=False, encoding="utf-8-sig")

        print(f"[SAVE] 结果已保存：\n  CSV : {csv_path}")

    def save_checkpoint(self, home_path, log_dir, non_adapted):
        save_dict = {
            "non_adapted": non_adapted
        }
        # 保存模型checkpoint
        save_path = os.path.join(home_path, log_dir, f"checkpoint.pt")
        torch.save(save_dict, save_path)

    def load_checkpoint(self, model_dir): # 从指定目录加载 checkpoint.pt，提取non_adapted模型参数并返回
        checkpoint = torch.load(os.path.join(self.home_path, model_dir, 'checkpoint.pt'))
        pretrained_model = checkpoint['non_adapted']

        return pretrained_model

    def calculate_avg_std_wandb_table(self, results): #计算平均值和标准差，并将其添加到结果表中
        avg_metrics = [np.mean(results.get_column(metric)) for metric in results.columns[2:]]
        std_metrics = [np.std(results.get_column(metric)) for metric in results.columns[2:]]
        summary_metrics = {metric: np.mean(results.get_column(metric)) for metric in results.columns[2:]}

        results.add_data('mean', '-', *avg_metrics)
        results.add_data('std', '-', *std_metrics)

        return results, summary_metrics

    def log_summary_metrics_wandb(self, results, risks): # 计算表格（wandb.Table）的各指标列的平均值和标准差，

        # Calculate average and standard deviation for metrics
        avg_metrics = [np.mean(results.get_column(metric)) for metric in results.columns[2:]]
        std_metrics = [np.std(results.get_column(metric)) for metric in results.columns[2:]]

        avg_risks = [np.mean(risks.get_column(risk)) for risk in risks.columns[2:]]
        std_risks = [np.std(risks.get_column(risk)) for risk in risks.columns[2:]]

        # Estimate summary metrics
        summary_metrics = {metric: np.mean(results.get_column(metric)) for metric in results.columns[2:]}
        summary_risks = {risk: np.mean(risks.get_column(risk)) for risk in risks.columns[2:]}

        # append avg and std values to metrics
        results.add_data('mean', '-', *avg_metrics)
        results.add_data('std', '-', *std_metrics)

        # append avg and std values to risks
        results.add_data('mean', '-', *avg_risks)
        risks.add_data('std', '-', *std_risks)
        # 将mean和std作为新行添加到表末尾，并返回更新后的表和summary_metrics字典

    def wandb_logging(self, total_results, total_risks, summary_metrics, summary_risks):
        # 使用Weights & Biases记录结果表、风险表、超参数表和汇总指标
        wandb.log({'results': total_results})
        wandb.log({'risks': total_risks})
        wandb.log({'hparams': wandb.Table(
            dataframe=pd.DataFrame(dict(self.hparams).items(), columns=['parameter', 'value']),
            allow_mixed_types=True)})
        wandb.log(summary_metrics)
        wandb.log(summary_risks)

    def calculate_metrics(self, tta_model):
        # 计算适应后模型在整个目标域数据上的指标
        self.evaluate(self.trg_whole_dl, tta_model)
        acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()
        trg_risk = self.loss.item()

        return acc, f1, auroc, trg_risk

    def calculate_risks(self): #计算源测试集和目标测试集上的风险（分类误差）
        self.evaluate(self.src_test_dl)
        src_risk = self.loss.item()
        self.evaluate(self.trg_test_dl)
        trg_risk = self.loss.item()

        return src_risk, trg_risk

    def append_results_to_tables(self, table, scenario, run_id, metrics):
        # 将新的结果添加到表中
        if isinstance(metrics, float):
            results_row = [scenario, run_id, metrics]
        elif isinstance(metrics, tuple):
            results_row = [scenario, run_id, *metrics]

        # Create new dataframes for each row
        results_df = pd.DataFrame([results_row], columns=table.columns)

        # Concatenate new dataframes with original dataframes
        table = pd.concat([table, results_df], ignore_index=True)

        return table

    def add_mean_std_table(self, table, columns): #计算表格的各指标列的平均值和标准差，并将其添加到结果表中
        # Calculate average and standard deviation for metrics
        avg_metrics = [table[metric].mean() for metric in columns[2:]]
        std_metrics = [table[metric].std() for metric in columns[2:]]

        # Create dataframes for mean and std values
        mean_metrics_df = pd.DataFrame([['mean', '-', *avg_metrics]], columns=columns)
        std_metrics_df = pd.DataFrame([['std', '-', *std_metrics]], columns=columns)

        # Concatenate original dataframes with mean and std dataframes
        table = pd.concat([table, mean_metrics_df, std_metrics_df], ignore_index=True)

        # Create a formatting function to format each element in the tables
        format_func = lambda x: f"{x:.4f}" if isinstance(x, float) else x

        # Apply the formatting function to each element in the tables
        table = table.applymap(format_func)

        return table

