from models import SemanticNet
from distances import Distance
from losses.base import ModelLoader, DeviceMapperTransform, TestLogger
from losses.wrappers import STSBaselineClassifier
from datasets.semeval import SemEval, SemEvalPartitionFactory
from sts.augmentation import NoAugmentation
from sts.modes import STSForwardMode, PairSTSForwardMode, ConcatSTSForwardMode
from metrics import STSEmbeddingEvaluator, STSBaselineEvaluator, DistanceSpearmanMetric, LogitsSpearmanMetric
from experiments.base import ModelEvaluationExperiment
from common import DEVICE


class SemEvalEvaluationExperiment(ModelEvaluationExperiment):

    def __init__(self, model_loader: ModelLoader, nfeat: int, data_path: str, word2vec_path: str,
                 vocab_path: str, distance: Distance, log_interval: int, batch_size: int):

        loss_name = model_loader.get_trained_loss()

        # The augmentation is only done for the training set, so it doesn't matter which one we choose.
        # Here I'm choosing NoAugmentation allowing redundancy because it's the cheapest to compute
        augmentation = NoAugmentation(allow_redundancy=True)
        # All partition types (should!) format the data in the same manner when it comes to dev or test
        # so it doesn't matter what loss we choose here
        partition_factory = SemEvalPartitionFactory(loss=loss_name, batch_size=batch_size)
        self.dataset = SemEval(data_path, word2vec_path, vocab_path, augmentation, partition_factory)

        self.model = SemanticNet(DEVICE, nfeat, self.dataset.vocab,
                                 mode=self._get_model_mode(),
                                 loss_module=self._get_loss_module())
        model_loader.load(self.model, loss_name)
        self.model = self._transform_model(self.model)
        self.model = self.model.to(DEVICE)

        self.dev_evaluator, self.test_evaluator = None, None
        self.log_interval = log_interval
        self.distance = distance
        self.nfeat = nfeat

    def evaluate_on_dev(self) -> float:
        if self.dev_evaluator is None:
            self.dev_evaluator = self._build_evaluator(self.dataset.dev_partition())
        self.dev_evaluator.eval(self.model)
        return self.dev_evaluator.metric.get()

    def evaluate_on_test(self) -> float:
        if self.test_evaluator is None:
            self.test_evaluator = self._build_evaluator(self.dataset.test_partition())
        self.test_evaluator.eval(self.model)
        return self.test_evaluator.metric.get()

    def _build_evaluator(self, partition):
        raise NotImplementedError

    def _get_model_mode(self) -> STSForwardMode:
        raise NotImplementedError

    def _get_loss_module(self):
        raise NotImplementedError

    def _transform_model(self, model):
        return model


class SemEvalEmbeddingEvaluationExperiment(SemEvalEvaluationExperiment):

    def _build_evaluator(self, partition):
        return STSEmbeddingEvaluator(DEVICE, partition, DistanceSpearmanMetric(self.distance),
                                     batch_transforms=[DeviceMapperTransform(DEVICE)],
                                     callbacks=[TestLogger(self.log_interval, partition.nbatches())])

    def _get_model_mode(self) -> STSForwardMode:
        return PairSTSForwardMode()

    def _get_loss_module(self):
        return None

    def _transform_model(self, model):
        return self.model.to_prediction_model()


class SemEvalBaselineModelEvaluationExperiment(SemEvalEvaluationExperiment):

    def _build_evaluator(self, partition):
        return STSBaselineEvaluator(DEVICE, partition, LogitsSpearmanMetric(),
                                    batch_transforms=[DeviceMapperTransform(DEVICE)],
                                    callbacks=[TestLogger(self.log_interval, partition.nbatches())])

    def _get_model_mode(self) -> STSForwardMode:
        return ConcatSTSForwardMode()

    def _get_loss_module(self):
        return STSBaselineClassifier(self.nfeat)
