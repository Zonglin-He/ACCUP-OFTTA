import math

def get_hparams_class(dataset_name): # 根据给定数据集名称字符串返回对应的数据集配置类
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class FD(): 
    def __init__(self):
        super(FD, self).__init__()
        self.train_params = {
            'num_epochs': 20,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5,
            'steps': 3,
            'optim_method': 'adam',
            'momentum': 0.9
        }
        self.alg_hparams = {
            'ACCUP': {
                    'pre_learning_rate': 1e-3,
                    'learning_rate': 2e-5,
                    'tau': 0.98,
                    'temperature': 0.8,   # 若无用可删
                    'filter_K': 20,       # 40 太大易引入噪声
                    'proto_consistency': True,
                    'proto_ent_q': 0.75,   # 关键：关闭固定 0.5，交给 Head 内部动态阈
                    'edtn_alpha0': 0.70,
                    'edtn_lambda0': 0.65,  # 目标 prior[-1]≈0.8
                    'proto_scale': 11.0,
                    'revin': True,
                },
            'NoAdap': {'pre_learning_rate': 1e-3}
        }

class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 64,
            'weight_decay': 5e-4,
            'step_size': 20,
            'lr_decay': 0.5,
            'steps': 1,
            'optim_method': 'adam',
            'momentum': 0.9
        }
        self.alg_hparams = {
            'ACCUP': {
                'pre_learning_rate': 1e-3,
                'learning_rate': 1e-4,
                'tau': 1.0,
                'temperature': 0.9,
                'filter_K': 15,
                'proto_consistency': True,
                'proto_ent_q': 0.0,
                'edtn_alpha0': 0.5,
                'edtn_lambda': 0.9,
                'proto_scale': 7.0,
                'revin': True
            },
            'NoAdap': {'pre_learning_rate': 1e-3}
        }


class HAR():
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
            'num_epochs': 20,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5,
            'steps': 2,
            'optim_method':'adam',
            'momentum':0.9
        }
        self.alg_hparams = {
            'ACCUP': {
        'pre_learning_rate': 1e-3,
        'learning_rate': 5e-4,
        'tau': 1.0,
        'temperature': 0.7,
        'filter_K': 20,
        'proto_consistency': True,
        'proto_ent_q': 0.0,
        'edtn_alpha0': 0.7,
        'edtn_lambda': 0.7,
        'proto_scale': 10.0
    },
    'NoAdap': {'pre_learning_rate': 1e-3}
}
