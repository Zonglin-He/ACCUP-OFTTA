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
from algorithms.oftta_proto import OFTTAPrototypeHead
from algorithms.revin import RevIN
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()



class TTATrainer(TTAAbstractTrainer):
    """
   This class contain the main training functions for our method.
   训练方法
    """
    def __init__(self, args): # TTATrainer 初始化
        super(TTATrainer, self).__init__(args) #调用父类的初始化方法
        self.edtn_alpha0 = float(self.hparams.get('edtn_alpha0', args.edtn_alpha0))
        self.edtn_lambda = float(self.hparams.get('edtn_lambda', args.edtn_lambda))
        self.proto_K = int(self.hparams.get('filter_K', args.proto_K))
        self.proto_consistency = bool(self.hparams.get('proto_consistency', args.proto_consistency))
        self.proto_ent_q = float(self.hparams.get('proto_ent_q', args.proto_ent_q))
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description, f"{self.run_description}") #实验日志目录
        self.load_pretrained_checkpoint = os.path.join(self.home_path, self.save_dir, self.experiment_description, f"{'NoAdap'}_{'All_Trg'}") #预训练检查点路径
        os.makedirs(self.exp_log_dir, exist_ok=True) #创建实验日志目录
        self.summary_f1_scores = open(self.exp_log_dir + '/summary_f1_scores.txt', 'w') #用于记录汇总F1分数的文件
        self.use_revin = bool(args.revin)
        self.revin_on_train = False

        if self.use_revin:
            C = self.dataset_configs.input_channels  # 正确拿到通道数
            self.revin = RevIN(center_only=True, eps=1e-4, affine=False, num_channels=C).to(self.device)
            self.revin.eval()  # 纯预处理，不参与训练
            for p in self.revin.parameters():
                p.requires_grad = False
            print("[RevIN] enabled (center-only).")

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
            if x.dim() == 2:  # [B,T] 兜底
                x = x.unsqueeze(1)  # -> [B,1,T]

            if x.dim() == 3:
                B, d1, d2 = x.shape
                C = self.dataset_configs.input_channels
                if d1 == C:  # [B,C,T]
                    return self.revin(x)
                elif d2 == C:  # [B,T,C]
                    return self.revin(x.permute(0, 2, 1)).permute(0, 2, 1)
                else:
                    # 兜底：按 [B,C,T] 处理（C==1 常见）
                    return self.revin(x)
            return x

        if isinstance(data, (list, tuple)):  # 兼容双视图等
            out = [_norm_one(t) for t in data]
            return type(data)(out) if isinstance(data, tuple) else out
        return _norm_one(data)

    def test_time_adaptation(self): #测试时间适应主函数
        results_columns = ["scenario", "run", "acc", "f1_score", "auroc"] #初始化结果表和风险表
        table_results = pd.DataFrame(columns=results_columns)
        risks_columns = ["scenario", "run", "trg_risk"]
        table_risks = pd.DataFrame(columns=risks_columns)

        for src_id, trg_id in self.dataset_configs.scenarios: #遍历所有源目标域对
            cur_scenario_f1_ret = []
            for run_id in range(self.num_runs): #对于每个源-目标域对，进行多次运行以计算平均性能
                self.run_id = run_id
                fix_randomness(run_id)
                print(run_id)
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir, src_id, trg_id, run_id)
                self.pre_loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                if self.da_method == "NoAdap":
                    self.load_data(src_id, trg_id)
                else:
                    self.load_data_demo(src_id, trg_id, run_id)

                print('Total test datasize:', len(self.trg_whole_dl.dataset)) #打印目标域数据集的总大小
                # 统计目标域数据集中的每个类别的样本数量
                all_labels = torch.zeros(self.dataset_configs.num_classes)
                for batch_idx, (inputs, target, _) in enumerate(self.trg_whole_dl):
                    for id in range(target.shape[0]):
                        all_labels[target[id]] += 1
                print('trg whole labels:', all_labels)

                non_adapted_model_state, pre_trained_model = self.pre_train()
                # 只有做 TTA 时才打补丁
                if self.da_method != "NoAdap":
                    pre_trained_model = patch_bn_to_edtn(
                        pre_trained_model, alpha0=self.edtn_alpha0, lam=self.edtn_lambda
                    )

                self.save_checkpoint(self.home_path, self.scenario_log_dir, non_adapted_model_state) #保存预训练模型检查点

                ## if finished pre_train. we can directly load pretrainModel from checkpoint ###
                # load_pretrained_checkpoint_path = os.path.join(self.load_pretrained_checkpoint, src_id + "_to_" + trg_id + "_run_" + str(run_id))
                # pre_trained_model = self.initialize_pretrained_model()
                # pre_trained_model_chk = self.load_checkpoint(load_pretrained_checkpoint_path)  # all method load same pretrained model.
                # pre_trained_model.network.load_state_dict(pre_trained_model_chk)

                # 以上注释掉的代码表明：
                # 可以直接加载预训练模型而不每次都训；特别地，作者考虑过统一用NoAdap_All_Trg实验的预训练模型用于所有方法的对比，以公平比较。
                # 但这里仍然每次run都调用 pre_train() 从零开始训练源模型。

                optimizer = build_optimizer(self.hparams) #构建优化器
                self.hparams.setdefault('filter_K', self.proto_K)
                self.hparams.setdefault('proto_consistency', self.proto_consistency)
                self.hparams.setdefault('proto_ent_q', self.proto_ent_q)
                if self.da_method == 'NoAdap': #如果不进行适应，直接使用预训练模型
                    tta_model = pre_trained_model
                    tta_model.eval()
                else: #否则，构建TTA模型并进行适应
                    tta_model_class = self.get_tta_model_class() 
                    tta_model = tta_model_class(self.dataset_configs, self.hparams, pre_trained_model, optimizer)
                tta_model.to(self.device) #将模型移动到指定设备（CPU或GPU）
                pre_trained_model.eval() #将预训练模型设置为评估模式

                metrics = self.calculate_metrics(tta_model) #计算适应后模型在整个目标域数据上的指标
                cur_scenario_f1_ret.append(metrics[1]) #记录当前场景的F1分数
                scenario = f"{src_id}_to_{trg_id}" #场景名称
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics[:3]) #将新的结果添加到结果表中
                table_risks = self.append_results_to_tables(table_risks, scenario, run_id, metrics[-1]) #将新的风险添加到风险表中

            cur_avg_f1_scores, cur_std_f1_scores = 100. * np.mean(cur_scenario_f1_ret), 100. * np.std(cur_scenario_f1_ret)
            print('Average current f1_scores::', cur_avg_f1_scores, 'Std:', cur_std_f1_scores)
            print(scenario, ' : ', np.around(cur_avg_f1_scores, 2), '/', np.around(cur_std_f1_scores, 2), sep='', file=self.summary_f1_scores)

        #将均值和方差加入表中
        table_results = self.add_mean_std_table(table_results, results_columns)
        table_risks = self.add_mean_std_table(table_risks, risks_columns)
        self.save_tables_to_file(table_results, datetime.now().strftime('%d_%m_%Y_%H_%M_%S') +'_results')
        self.save_tables_to_file(table_risks, datetime.now().strftime('%d_%m_%Y_%H_%M_%S') +'_risks')

        self.summary_f1_scores.close()

if __name__ == "__main__":
    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='results/tta_experiments_logs', type=str, help='Directory containing all experiments')
    parser.add_argument('--exp_name', default='All_Trg', type=str, help='experiment name')
    # ========= Select the DA methods ============
    parser.add_argument('--da_method', default='ACCUP', type=str, help='ACCUP, NoAdap')
    # ========= Select the DATASET ==============
    parser.add_argument('--data_path', default=r'E:\Dataset', type=str, help='Path containing dataset')
    parser.add_argument('--dataset', default='EEG', type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')
    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='CNN', type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')
    # ========= Experiment settings ===============
    parser.add_argument('--num_runs', default=3, type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')

    parser.add_argument("--edtn_alpha0", type=float, default=0.7)
    parser.add_argument("--edtn_lambda", type=float, default=0.7)
    parser.add_argument("--proto_K", type=int, default=10)
    parser.add_argument("--proto_consistency", action="store_true", help="开启双视图一致性门控")
    parser.add_argument("--proto_ent_q", type=float, default=0.0, help="熵阈的分位数(0表示禁用)")
    parser.add_argument("--revin", action="store_true", help="Enable RevIN (channel-wise instance norm) at TTA")

    args = parser.parse_args() #解析命令行参数
    trainer = TTATrainer(args) #创建TTATrainer实例
    trainer.test_time_adaptation() #运行测试时间适应

