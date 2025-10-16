import torch
import torch.nn.functional as F
from torch import nn as nn

import random
import os
import sys
import logging
import numpy as np
import pandas as pd
from shutil import copy
from datetime import datetime
from einops import rearrange

from skorch import NeuralNetClassifier  # for DIV Risk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


class AverageMeter(object): #计算并存储平均值和当前值

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def fix_randomness(SEED): #设置随机种子以确保结果可复现
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _logger(logger_name, level=logging.DEBUG): #创建日志记录器，返回日志
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def starting_logs(data_type,da_method, exp_log_dir, src_id, tgt_id, run_id): #初始化日志记录器和日志目录
    log_dir = os.path.join(exp_log_dir, src_id + "_to_" + tgt_id + "_run_" + str(run_id))
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    return logger, log_dir


def save_checkpoint(home_path, algorithm, selected_scenarios, dataset_configs, log_dir, hparams): #保存模型检查点
    save_dict = {
        "x-domains": selected_scenarios,
        "configs": dataset_configs.__dict__,
        "hparams":  dict(hparams),
        "model_dict": algorithm.state_dict()
    }
    # save classification report
    save_path = os.path.join(home_path, log_dir, "checkpoint.pt")

    torch.save(save_dict, save_path)


def weights_init(m): #初始化神经网络层的权重
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)

def _calc_metrics(pred_labels, true_labels, log_dir, home_path, target_names): #计算分类指标并保存分类报告
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    r = classification_report(true_labels, pred_labels, target_names=target_names, digits=6, output_dict=True)

    df = pd.DataFrame(r)
    accuracy = accuracy_score(true_labels, pred_labels)
    df["accuracy"] = accuracy
    df = df * 100

    # save classification report
    file_name = "classification_report.xlsx"
    report_Save_path = os.path.join(home_path, log_dir, file_name)
    df.to_excel(report_Save_path)

    return accuracy * 100, r["macro avg"]["f1-score"] * 100

def _calc_metrics_pretrain(src_pred, src_true,trg_pred, trg_true,  log_dir, home_path, target_names):  #计算预训练模型在源域和目标域上的分类指标

    src_pred_labels = np.array(src_pred).astype(int)
    src_true_labels = np.array(src_true).astype(int)
    trg_pred_labels = np.array(trg_pred).astype(int)
    trg_true_labels = np.array(trg_true).astype(int)

    src_rep = classification_report(src_true_labels, src_pred_labels, target_names=target_names, digits=6, output_dict=True)
    trg_rep = classification_report(trg_true_labels, trg_pred_labels, target_names=target_names, digits=6, output_dict=True)

    src_df = pd.DataFrame(src_rep)
    trg_df = pd.DataFrame(trg_rep)

    src_acc = accuracy_score(src_true_labels, src_pred_labels)
    trg_acc = accuracy_score(trg_true_labels, trg_pred_labels)

    return src_acc * 100, src_df["macro avg"]["f1-score"] * 100, trg_acc *100,  trg_df["macro avg"]["f1-score"] *100

import collections
def to_device(input, device): #将输入数据移动到指定设备（CPU或GPU）
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError("Input must contain tensor, dict or list, found {type(input)}")

def copy_Files(destination): #备份关键代码文件到指定目录
    destination_dir = os.path.join(destination, "MODEL_BACKUP_FILES")
    os.makedirs(destination_dir, exist_ok=True)
    copy("main.py", os.path.join(destination_dir, "main.py"))
    copy("algorithms/algorithms.py", os.path.join(destination_dir, "algorithms.py"))
    copy(f"configs/data_model_configs.py", os.path.join(destination_dir, f"data_model_configs.py"))
    copy(f"configs/hparams.py", os.path.join(destination_dir, f"hparams.py"))
    copy(f"configs/sweep_params.py", os.path.join(destination_dir, f"sweep_params.py"))
    copy("utils.py", os.path.join(destination_dir, "utils.py"))

def get_iwcv_value(weight, error): #计算加权交叉验证风险
    N, d = weight.shape
    _N, _d = error.shape
    assert N == _N and d == _d, 'dimension mismatch!'
    weighted_error = weight * error
    return np.mean(weighted_error)

def get_dev_value(weight, error): #计算加权验证风险
    """
    :param weight: shape [N, 1], the importance weight for N source samples in the validation set
    :param error: shape [N, 1], the error value for each source sample in the validation set
    (typically 0 for correct classification and 1 for wrong classification)
    """
    N, d = weight.shape
    _N, _d = error.shape
    assert N == _N and d == _d, 'dimension mismatch!'
    weighted_error = weight * error
    cov = np.cov(np.concatenate((weighted_error, weight), axis=1), rowvar=False)[0][1]
    var_w = np.var(weight, ddof=1)
    eta = - cov / var_w
    return np.mean(weighted_error) + eta * np.mean(weight) - eta

class simple_MLP(nn.Module): #简单的多层感知机，用于域分类器
    def __init__(self, inp_units, out_units=2):
        super(simple_MLP, self).__init__()

        self.dense0 = nn.Linear(inp_units, inp_units//2)
        self.nonlin = nn.ReLU()
        self.output = nn.Linear(inp_units//2, out_units)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, **kwargs):
        x = self.nonlin(self.dense0(x))
        x = self.softmax(self.output(x))
        return x   
    
def get_weight_gpu(source_feature, target_feature, validation_feature, configs, device): #计算源域和目标域样本的权重
    """
    :param source_feature: shape [N_tr, d], features from training set
    :param target_feature: shape [N_te, d], features from test set
    :param validation_feature: shape [N_v, d], features from validation set
    :return:
    """
    import copy
    N_s, d = source_feature.shape
    N_t, _d = target_feature.shape
    source_feature = copy.deepcopy(source_feature.detach().cpu()) #source_feature.clone()
    target_feature = copy.deepcopy(target_feature.detach().cpu()) #target_feature.clone()
    source_feature = source_feature.to(device)
    target_feature = target_feature.to(device)
    all_feature = torch.cat((source_feature, target_feature), dim=0)
    all_label = torch.from_numpy(np.asarray([1] * N_s + [0] * N_t, dtype=np.int32)).long()
    
    feature_for_train, feature_for_test, label_for_train, label_for_test = train_test_split(all_feature, all_label,
                                                                                            train_size=0.8)
    learning_rates = [1e-1, 5e-2, 1e-2]
    val_acc = []
    domain_classifiers = []
    
    for lr in learning_rates:    
        domain_classifier = NeuralNetClassifier(
            simple_MLP,
            module__inp_units = configs.final_out_channels * configs.features_len,
            max_epochs=30,
            lr=lr,
            device=device,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            callbacks="disable"
        )
        domain_classifier.fit(feature_for_train.float(), label_for_train.long())
        output = domain_classifier.predict(feature_for_test)
        acc = np.mean((label_for_test.numpy() == output).astype(np.float32))
        val_acc.append(acc)
        domain_classifiers.append(domain_classifier)
    
    index = val_acc.index(max(val_acc))
    domain_classifier = domain_classifiers[index]

    domain_out = domain_classifier.predict_proba(validation_feature.to(device).float())
    return domain_out[:, :1] / domain_out[:, 1:] * N_s * 1.0 / N_t


def calc_dev_risk(target_model, src_train_dl, tgt_train_dl, src_valid_dl, configs, device): #计算验证风险
    src_train_feats, _ = target_model.feature_extractor(src_train_dl.dataset.x_data.float().to(device))
    tgt_train_feats, _ = target_model.feature_extractor(tgt_train_dl.dataset.x_data.float().to(device))
    src_valid_feats, _ = target_model.feature_extractor(src_valid_dl.dataset.x_data.float().to(device))
    src_valid_pred = target_model.classifier(src_valid_feats)

    dev_weights = get_weight_gpu(src_train_feats.to(device), tgt_train_feats.to(device),
                                 src_valid_feats.to(device), configs, device)
    dev_error = F.cross_entropy(src_valid_pred, src_valid_dl.dataset.y_data.long().to(device), reduction='none')
    dev_risk = get_dev_value(dev_weights, dev_error.unsqueeze(1).detach().cpu().numpy())
    # iwcv_risk = get_iwcv_value(dev_weights, dev_error.unsqueeze(1).detach().cpu().numpy())
    return dev_risk


def calculate_risk(target_model, risk_dataloader, device): #计算给定数据集上的分类损失作为风险

    x_data = risk_dataloader.dataset.x_data
    y_data = risk_dataloader.dataset.y_data

    feat, _ = target_model.feature_extractor(x_data.float().to(device))
    pred = target_model.classifier(feat)
    cls_loss = F.cross_entropy(pred, y_data.long().to(device))
    return cls_loss.item()

def get_named_submodule(model, sub_name: str): #根据子模块名称获取子模块
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)

    return module


def set_named_submodule(model, sub_name, value): #根据子模块名称设置子模块的值
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])

        else:
            setattr(module, names[i], value)