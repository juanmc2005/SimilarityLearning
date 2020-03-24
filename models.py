#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import common
from torch import nn
from sts.baseline import STSBaselineNet
from sts.modes import STSForwardMode, ConcatSTSForwardMode
from losses.wrappers import SNLIClassifier


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class MetricNet(nn.Module):

    def __init__(self, encoder: nn.Module, classifier: nn.Module = None):
        super(MetricNet, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def encoder_state_dict(self):
        return self.encoder.state_dict()

    def load_encoder_state_dict(self, checkpoint):
        self.encoder.load_state_dict(checkpoint)

    def to_prediction_model(self):
        return PredictionModel(self)

    def forward(self, x, y):
        x = self.encoder(x)
        logits = self.classifier(x, y) if self.classifier is not None else None
        return x, logits

    def all_params(self):
        params = [self.encoder.parameters()]
        if self.classifier is not None:
            params.append(self.classifier.parameters())
        return params


class PredictionModel(nn.Module):

    def __init__(self, model: MetricNet):
        super(PredictionModel, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model.encoder(x)


def MNISTNet(nfeat: int, classifier: nn.Module = None) -> MetricNet:
    encoder = nn.Sequential(
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
        nn.PReLU())
    return MetricNet(encoder=encoder, classifier=classifier)


def SemanticNet(nfeat: int, nlayers: int, word_list: list, vector_vocab: dict,
                mode: STSForwardMode, classifier: nn.Module = None) -> MetricNet:
    encoder = STSBaselineNet(common.DEVICE, nfeat_word=300, nfeat_sent=nfeat,
                             nlayers=nlayers, word_list=word_list,
                             vec_vocab=vector_vocab, mode=mode)
    return MetricNet(encoder=encoder, classifier=classifier)


# TODO this hasn't been updated for a while, it's not working
def SNLIClassifierNet(encoder_loader, nfeat_sent: int,
                      nclass: int, nlayers: int, vector_vocab: dict) -> MetricNet:
    model = SemanticNet(nfeat_sent, nlayers, vector_vocab, ConcatSTSForwardMode())
    # Load encoder
    encoder_loader.load(model, encoder_loader.get_trained_loss())
    encoder = model.base_model
    # Put encoder in evaluation mode so it won't be learned
    encoder.eval()
    return MetricNet(encoder=encoder, classifier=SNLIClassifier(nfeat_sent, nclass))
