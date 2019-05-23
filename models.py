#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn
from sincnet import SincNet, MLP
from losses.arcface import ArcLinear


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class MNISTNet(nn.Module):

    def __init__(self, nfeat, loss_module=None):
        super(MNISTNet, self).__init__()
        self.loss_module = loss_module
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

    def forward(self, x, y):
        feat = self.net(x)
        logits = self.loss_module(feat, y) if self.loss_module is not None else None
        return feat, logits

    def net_params(self):
        return self.net.parameters()

    def loss_params(self):
        return self.loss_module.parameters()


class SpeakerNet(nn.Module):

    def __init__(self, nfeat, sample_rate, window, loss_module=None):
        super(SpeakerNet, self).__init__()
        self.loss_module = loss_module
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
        self.loss_module = loss_module

    def forward(self, x, y):
        feat = self.dnn(self.cnn(x))
        logits = self.loss_module(feat, y) if self.loss_module is not None else None
        return feat, logits

    def all_params(self):
        return [self.cnn.parameters(), self.dnn.parameters(), self.loss_module.parameters()]


if __name__ == '__main__':
    x = torch.rand(50, 3200)
    y = torch.randint(0, 1251, (50,))
    loss_module = ArcLinear(2048, 1251, margin=0.2, s=7.)
    net = SpeakerNet(2048, sample_rate=16000, window=200, loss_module=loss_module)
    feat, logits = net(x, y)
    print(f"feat size = {feat.size()}")
    print(f"logits size = {logits.size() if logits is not None else None}")
