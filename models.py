#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import nn
from sincnet import SincNet, MLP
from sts.baseline import STSBaselineNet, STSForwardMode


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SimNet(nn.Module):

    def __init__(self, loss_module=None):
        super(SimNet, self).__init__()
        self.loss_module = loss_module

    def layers(self):
        raise NotImplementedError

    def common_state_dict(self):
        raise NotImplementedError

    def load_common_state_dict(self, checkpoint):
        raise NotImplementedError

    def to_prediction_model(self):
        return PredictionModel(self)

    def forward(self, x, y):
        for layer in self.layers():
            x = layer(x)
        logits = self.loss_module(x, y) if self.loss_module is not None else None
        return x, logits

    def all_params(self):
        params = [layer.parameters() for layer in self.layers()]
        if self.loss_module is not None:
            params.append(self.loss_module.parameters())
        return params


class PredictionModel(nn.Module):

    def __init__(self, model: SimNet):
        super(PredictionModel, self).__init__()
        self.model = model

    def forward(self, x):
        for layer in self.model.layers():
            x = layer(x)
        return x


class MNISTNet(SimNet):

    def __init__(self, nfeat, loss_module=None):
        super(MNISTNet, self).__init__(loss_module)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(128 * 3 * 3, nfeat),
            nn.PReLU()
        )

    def layers(self):
        return [self.net]

    def common_state_dict(self):
        return self.net.state_dict()

    def load_common_state_dict(self, checkpoint):
        self.net.load_state_dict(checkpoint)


class SpeakerNet(SimNet):

    def __init__(self, nfeat, sample_rate, window, loss_module=None):
        super(SpeakerNet, self).__init__(loss_module)
        wlen = int(sample_rate * window / 1000)
        self.cnn = SincNet({'input_dim': wlen,
                            'fs': sample_rate,
                            'cnn_N_filt': [80, 60, 60],
                            'cnn_len_filt': [251, 5, 5],
                            'cnn_max_pool_len': [3, 3, 3],
                            'cnn_use_laynorm_inp': True,
                            'cnn_use_batchnorm_inp': False,
                            'cnn_use_laynorm': [True, True, True],
                            'cnn_use_batchnorm': [False, False, False],
                            'cnn_act': ['leaky_relu', 'leaky_relu', 'leaky_relu'],
                            'cnn_drop': [0., 0., 0.],
                            })
        self.dnn = MLP({'input_dim': self.cnn.out_dim,
                        'fc_lay': [2048, 2048, nfeat],
                        'fc_drop': [0., 0., 0.],
                        'fc_use_batchnorm': [True, True, True],
                        'fc_use_laynorm': [False, False, False],
                        'fc_use_laynorm_inp': True,
                        'fc_use_batchnorm_inp': False,
                        'fc_act': ['leaky_relu', 'leaky_relu', 'leaky_relu'],
                        })

    def layers(self):
        return [self.cnn, self.dnn]

    def common_state_dict(self):
        return {
            'cnn': self.cnn.state_dict(),
            'dnn': self.dnn.state_dict()
        }

    def load_common_state_dict(self, checkpoint):
        self.cnn.load_state_dict(checkpoint['cnn'])
        self.dnn.load_state_dict(checkpoint['dnn'])


class SemanticNet(SimNet):

    def __init__(self, device: str, nfeat: int, vector_vocab: dict, mode: STSForwardMode, loss_module: nn.Module = None):
        super().__init__(loss_module)
        self.base_model = STSBaselineNet(device, nfeat_word=300, nfeat_sent=nfeat, vec_vocab=vector_vocab, mode=mode)

    def layers(self):
        return [self.base_model]

    def common_state_dict(self):
        return self.base_model.state_dict()

    def load_common_state_dict(self, checkpoint):
        self.base_model.load_state_dict(checkpoint)
