import sys
sys.path.append('../ADATIME')

import os
import pandas as pd
import torch
import collections
import argparse
import warnings
import sklearn.exceptions
from datetime import datetime
import numpy as np

from utils.utils import fix_randomness, starting_logs, AverageMeter
from .tta_abstract_trainer import TTAAbstractTrainer
from optim.optimizer import build_optimizer
from algorithms.oftta_edtn import patch_bn_to_edtn
from algorithms.oftta_proto import OFTTAPrototypeHead  # 供 TTA 模型内部使用
from algorithms.revin import RevIN

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()


class TTATrainer(TTAAbstractTrainer):
    """
    This class contain the main training functions for our method.
    训练方法
    """
    def __init__(self, args):
        super(TTATrainer, self).__init__(args)

        # ---- 读取/兜底关键超参（优先 hparams -> 命令行）----
        self.edtn_alpha0 = float(self.hparams.get('edtn_alpha0', args.edtn_alpha0))
        self.edtn_lambda = float(self.hparams.get('edtn_lambda', args.edtn_lambda))

        self.proto_K = int(self.hparams.get('filter_K', args.proto_K))
        self.proto_consistency = bool(self.hparams.get('proto_consistency', args.proto_consistency))
        self.proto_ent_q = float(self.hparams.get('proto_ent_q', args.proto_ent_q))

        # 日志目录
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir,
                                        self.experiment_description, f"{self.run_description}")
        self.load_pretrained_checkpoint = os.path.join(
            self.home_path, self.save_dir, self.experiment_description, f"{'NoAdap'}_{'All_Trg'}"
        )
        os.makedirs(self.exp_log_dir, exist_ok=True)
        self.summary_f1_scores = open(self.exp_log_dir + '/summary_f1_scores.txt', 'w')

        # RevIN
        self.use_revin = bool(args.revin)
        self.revin_on_train = False
        if self.use_revin:
            C = self.dataset_configs.input_channels  # 正确拿到通道数
            # 是否 center_only 由这里控制；你当前习惯用 False 就保持，但打印真实值
            self.revin = RevIN(center_only=False, eps=1e-5, affine=False, num_channels=C).to(self.device)
            self.revin.eval()
            for p in self.revin.parameters():
                p.requires_grad = False
            print(f"[RevIN] enabled. center_only={self.revin.center_only}")

    # 父类会在前向前调用这个钩子
    def _preprocess_inputs(self, data):
        if self.use_revin and not hasattr(self, "_revin_log_once"):
            shapes = [tuple(d.shape) for d in data] if isinstance(data, (list, tuple)) else tuple(data.shape)
            print(f"[RevIN] enabled. input shapes: {shapes}")
            self._revin_log_once = True

        if not self.use_revin:
            return data

        def _norm_one(x: torch.Tensor):
            if not isinstance(x, torch.Tensor):
                return x
            if x.dim() == 2:          # [B,T] 兜底
                x = x.unsqueeze(1)    # -> [B,1,T]
            if x.dim() == 3:
                B, d1, d2 = x.shape
                C = self.dataset_configs.input_channels
                if d1 == C:           # [B,C,T]
                    return self.revin(x)
                elif d2 == C:         # [B,T,C]
                    return self.revin(x.permute(0, 2, 1)).permute(0, 2, 1)
                else:                 # 兜底：按 [B,C,T] 处理（C==1 常见）
                    return self.revin(x)
            return x

        if isinstance(data, (list, tuple)):  # 兼容双视图等
            out = [_norm_one(t) for t in data]
            return type(data)(out) if isinstance(data, tuple) else out
        return _norm_one(data)

    def test_time_adaptation(self):
        results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        table_results = pd.DataFrame(columns=results_columns)
        risks_columns = ["scenario", "run", "trg_risk"]
        table_risks = pd.DataFrame(columns=risks_columns)

        for src_id, trg_id in self.dataset_configs.scenarios:
            cur_scenario_f1_ret = []
            for run_id in range(self.num_runs):
                self.run_id = run_id
                fix_randomness(run_id)
                print(run_id)
                self.logger, self.scenario_log_dir = starting_logs(
                    self.dataset, self.da_method, self.exp_log_dir, src_id, trg_id, run_id
                )
                self.pre_loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                if self.da_method == "NoAdap":
                    self.load_data(src_id, trg_id)
                else:
                    self.load_data_demo(src_id, trg_id, run_id)

                print('Total test datasize:', len(self.trg_whole_dl.dataset))
                all_labels = torch.zeros(self.dataset_configs.num_classes)
                for _, (_, target, _) in enumerate(self.trg_whole_dl):
                    for i in range(target.shape[0]):
                        all_labels[target[i]] += 1
                print('trg whole labels:', all_labels)

                # 预训练源模型
                non_adapted_model_state, pre_trained_model = self.pre_train()

                # 仅 TTA 时给 BN 打补丁（EDTN）
                if self.da_method != "NoAdap":
                    pre_trained_model = patch_bn_to_edtn(
                        pre_trained_model,
                        alpha0=float(self.hparams.get('edtn_alpha0', self.edtn_alpha0)),
                        lam=float(self.hparams.get('edtn_lambda', self.edtn_lambda)),  # 修正键名
                        verbose=True
                    )

                self.save_checkpoint(self.home_path, self.scenario_log_dir, non_adapted_model_state)

                self.hparams.setdefault('steps', 2)  # 你之前跑得好的日志是 steps=2
                self.hparams.setdefault('tau', 0.98)
                self.hparams.setdefault('proto_scale', 11.0)

                optimizer = build_optimizer(self.hparams)

                if self.da_method == 'NoAdap':
                    tta_model = pre_trained_model
                    tta_model.eval()
                else:
                    tta_model_class = self.get_tta_model_class()
                    tta_model = tta_model_class(self.dataset_configs, self.hparams, pre_trained_model, optimizer)

                tta_model.to(self.device)
                pre_trained_model.eval()

                metrics = self.calculate_metrics(tta_model)
                cur_scenario_f1_ret.append(metrics[1])
                scenario = f"{src_id}_to_{trg_id}"
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics[:3])
                table_risks = self.append_results_to_tables(table_risks, scenario, run_id, metrics[-1])

            cur_avg_f1_scores = 100. * np.mean(cur_scenario_f1_ret)
            cur_std_f1_scores = 100. * np.std(cur_scenario_f1_ret)
            print('Average current f1_scores::', cur_avg_f1_scores, 'Std:', cur_std_f1_scores)
            print(scenario, ' : ', np.around(cur_avg_f1_scores, 2), '/',
                  np.around(cur_std_f1_scores, 2), sep='', file=self.summary_f1_scores)

        # 汇总存盘
        table_results = self.add_mean_std_table(table_results, results_columns)
        table_risks = self.add_mean_std_table(table_risks, risks_columns)
        now = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
        self.save_tables_to_file(table_results, now + '_results')
        self.save_tables_to_file(table_risks,   now + '_risks')
        self.summary_f1_scores.close()


if __name__ == "__main__":
    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='results/tta_experiments_logs', type=str)
    parser.add_argument('--exp_name', default='All_Trg', type=str)
    # ========= Select the DA methods ============
    parser.add_argument('--da_method', default='ACCUP', type=str, help='ACCUP, NoAdap')
    # ========= Select the DATASET ==============
    parser.add_argument('--data_path', default=r'E:\Dataset', type=str)
    parser.add_argument('--dataset', default='EEG', type=str, help='(WISDM - EEG - HAR - HHAR_SA)')
    # ========= Select the BACKBONE =============
    parser.add_argument('--backbone', default='CNN', type=str, help='(CNN - RESNET18 - TCN)')
    # ========= Experiment settings =============
    parser.add_argument('--num_runs', default=3, type=int)
    parser.add_argument('--device', default="cuda", type=str)

    # ---- 这里给更稳的默认值（你要保持之前也行）----
    parser.add_argument("--edtn_alpha0", type=float, default=0.70)
    parser.add_argument("--edtn_lambda", type=float, default=0.65)  # 3 层 BN 时能把 prior[-1] 推到 ~0.8
    parser.add_argument("--proto_K", type=int, default=10)
    parser.add_argument("--proto_consistency", action="store_true", help="开启双视图一致性门控")
    parser.add_argument("--proto_ent_q", type=float, default=0.0, help="熵阈的分位数(0表示动态)")
    parser.add_argument("--revin", action="store_true", help="Enable RevIN at TTA")

    args = parser.parse_args()
    trainer = TTATrainer(args)
    trainer.test_time_adaptation()
