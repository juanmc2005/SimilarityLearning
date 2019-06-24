#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from pyannote.audio.features.utils import RawAudio
from pyannote.audio.embedding.generators import SpeechSegmentGenerator
from pyannote.database import get_protocol
from pyannote.database import FileFinder
from datasets.base import SimDataset, SimDatasetPartition
from metrics import SpeakerValidationConfig


class VoxCelebPartition(SimDatasetPartition):

    def __init__(self, generator, segment_size):
        self.generator = generator()
        self.batches_per_epoch = generator.batches_per_epoch
        self.segment_size = segment_size
        self.nclass = len(generator.specifications['y']['classes'])

    def nbatches(self):
        return self.batches_per_epoch

    def __next__(self):
        dic = next(self.generator)
        x = dic['X']
        batch = []
        for i in range(len(x)):
            actual_size = len(x[i])
            segment = torch.Tensor(x[i]).squeeze()
            if actual_size < self.segment_size:
                # Due to a pyannote crop bug, pad the tensor to fit the expected segment size
                batch.append(F.pad(segment, pad=[0, self.segment_size - actual_size], mode='constant', value=0))
            else:
                batch.append(segment)
        batch = torch.stack(batch, dim=0)
        return batch, torch.Tensor(dic['y']).long()


class VoxCeleb1(SimDataset):
    sample_rate = 16000

    @staticmethod
    def config(segment_size_s: float):
        return SpeakerValidationConfig(protocol_name='VoxCeleb.SpeakerVerification.VoxCeleb1_X',
                                       feature_extraction=RawAudio(sample_rate=VoxCeleb1.sample_rate),
                                       preprocessors={'audio': FileFinder()},
                                       duration=segment_size_s)

    def __init__(self, batch_size: int, segment_size_millis: int):
        self.batch_size = batch_size
        self.segment_size_s = segment_size_millis / 1000
        self.nfeat = VoxCeleb1.sample_rate * segment_size_millis // 1000
        self.config = VoxCeleb1.config(self.segment_size_s)
        self.protocol = get_protocol(self.config.protocol_name, preprocessors=self.config.preprocessors)
        self.train_gen, self.dev_gen, self.test_gen = None, None, None
        print(f"Segment Size = {self.segment_size_s}s")
        print(f"Embedding Size = {self.nfeat}")

    def training_partition(self) -> SimDatasetPartition:
        if self.train_gen is None:
            self.train_gen = SpeechSegmentGenerator(self.config.feature_extraction, self.protocol,
                                                    subset='train', per_label=1, per_fold=self.batch_size,
                                                    duration=self.segment_size_s, parallel=3)
        return VoxCelebPartition(self.train_gen, self.nfeat)

    def dev_partition(self) -> SimDatasetPartition:
        if self.dev_gen is None:
            self.dev_gen = SpeechSegmentGenerator(self.config.feature_extraction, self.protocol,
                                                  subset='development', per_label=1, per_fold=self.batch_size,
                                                  duration=self.segment_size_s, parallel=2)
        return VoxCelebPartition(self.dev_gen, self.nfeat)

    def test_partition(self) -> SimDatasetPartition:
        if self.test_gen is None:
            self.test_gen = SpeechSegmentGenerator(self.config.feature_extraction, self.protocol,
                                                   subset='test', per_label=1, per_fold=self.batch_size,
                                                   duration=self.segment_size_s, parallel=2)
        return VoxCelebPartition(self.test_gen, self.nfeat)
