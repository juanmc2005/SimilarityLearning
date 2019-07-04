#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from pyannote.audio.features.utils import RawAudio
from pyannote.audio.embedding.generators import SpeechSegmentGenerator
from pyannote.database import get_protocol
from pyannote.database import FileFinder
from datasets.base import SimDataset, SimDatasetPartition
import metrics


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


class VoxCelebDataset(SimDataset):

    def __init__(self, batch_size: int, segment_size_millis: int, segments_per_speaker: int = 1):
        self.sample_rate = 16000
        self.batch_size = batch_size
        self.segments_per_speaker = segments_per_speaker
        self.segment_size_s = segment_size_millis / 1000
        self.nfeat = self.sample_rate * segment_size_millis // 1000
        self.config = self._create_config(self.segment_size_s)
        self.protocol = get_protocol(self.config.protocol_name, preprocessors=self.config.preprocessors)
        self.train_gen, self.dev_gen, self.test_gen = None, None, None
        print(f"[Segment Size: {self.segment_size_s}s]")
        print(f"[Embedding Size: {self.nfeat}]")

    def _create_config(self, segment_size_sec: float):
        raise NotImplementedError

    def training_partition(self) -> VoxCelebPartition:
        if self.train_gen is None:
            self.train_gen = SpeechSegmentGenerator(self.config.feature_extraction, self.protocol,
                                                    subset='train', per_label=self.segments_per_speaker,
                                                    per_fold=self.batch_size // self.segments_per_speaker,
                                                    duration=self.segment_size_s, parallel=3, per_epoch=2)
        return VoxCelebPartition(self.train_gen, self.nfeat)

    def dev_partition(self) -> VoxCelebPartition:
        if self.dev_gen is None:
            self.dev_gen = SpeechSegmentGenerator(self.config.feature_extraction, self.protocol,
                                                  subset='development', per_label=1, per_fold=self.batch_size,
                                                  duration=self.segment_size_s, parallel=2, per_epoch=2)
        return VoxCelebPartition(self.dev_gen, self.nfeat)

    def test_partition(self) -> VoxCelebPartition:
        if self.test_gen is None:
            self.test_gen = SpeechSegmentGenerator(self.config.feature_extraction, self.protocol,
                                                   subset='test', per_label=1, per_fold=self.batch_size,
                                                   duration=self.segment_size_s, parallel=2)
        return VoxCelebPartition(self.test_gen, self.nfeat)


class VoxCeleb1(VoxCelebDataset):

    @staticmethod
    def _config(sample_rate: int, segment_size_sec: float):
        return metrics.SpeakerValidationConfig(protocol_name='VoxCeleb.SpeakerVerification.VoxCeleb1_X',
                                               feature_extraction=RawAudio(sample_rate=sample_rate),
                                               preprocessors={'audio': FileFinder()},
                                               duration=segment_size_sec)

    def _create_config(self, segment_size_sec: float):
        return self._config(self.sample_rate, segment_size_sec)


class VoxCeleb2(VoxCelebDataset):

    def _create_config(self, segment_size_sec: float):
        return metrics.SpeakerValidationConfig(protocol_name='VoxCeleb.SpeakerVerification.VoxCeleb2',
                                               feature_extraction=RawAudio(sample_rate=self.sample_rate),
                                               preprocessors={'audio': FileFinder()},
                                               duration=segment_size_sec)
