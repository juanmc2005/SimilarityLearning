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

    def nbatches(self):
        return self.batches_per_epoch

    def __next__(self):
        dic = next(self.generator)
        x = dic['X']
        batch = []
        for i in range(len(x)):
            actual_size = len(x[i])
            if actual_size < self.segment_size:
                # Due to a pyannote crop bug, pad the tensor to fit the expected segment size
                batch.append(F.pad(torch.Tensor(x[i]).squeeze(), pad=[0, self.segment_size - actual_size],
                                   mode='constant', value=0))
            else:
                batch.append(torch.Tensor(x[i]))
        batch = torch.stack(batch, dim=0)
        return batch, torch.Tensor(dic['y']).long()


class VoxCeleb1(SimDataset):

    def __init__(self, batch_size, segment_size_millis):
        sample_rate = 16000
        segment_size_s = segment_size_millis / 1000
        print(f"Segment Size = {segment_size_s}s")
        self.nfeat = sample_rate * segment_size_millis // 1000
        print(f"Embedding Size = {self.nfeat}")
        self.config = SpeakerValidationConfig(protocol_name='VoxCeleb.SpeakerVerification.VoxCeleb1_X',
                                              feature_extraction=RawAudio(sample_rate=sample_rate),
                                              preprocessors={'audio': FileFinder()},
                                              duration=segment_size_s)
        protocol = get_protocol(self.config.protocol_name, preprocessors=self.config.preprocessors)
        self.train_gen = SpeechSegmentGenerator(self.config.feature_extraction, protocol,
                                                subset='train', per_label=1, per_fold=batch_size,
                                                duration=segment_size_s, parallel=3)
        self.dev_gen = SpeechSegmentGenerator(self.config.feature_extraction, protocol,
                                              subset='development', per_label=1, per_fold=batch_size,
                                              duration=segment_size_s, parallel=2)

    def training_partition(self):
        return VoxCelebPartition(self.train_gen, self.nfeat)

    def dev_partition(self):
        return VoxCelebPartition(self.dev_gen, self.nfeat)