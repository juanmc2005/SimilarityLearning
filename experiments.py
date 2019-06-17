from models import SpeakerNet, SemanticNet
from distances import Distance
from losses.base import ModelLoader, DeviceMapperTransform, TestLogger
from losses.wrappers import STSBaselineClassifier
from datasets.voxceleb import VoxCeleb1
from datasets.semeval import SemEval, SemEvalPartitionFactory
from sts.augmentation import NoAugmentation
from sts.modes import PairSTSForwardMode, ConcatSTSForwardMode
from metrics import SpeakerVerificationEvaluator, STSEmbeddingEvaluator,\
    STSBaselineEvaluator, DistanceSpearmanMetric, LogitsSpearmanMetric
from common import DEVICE


class ModelEvaluationExperiment:

    def evaluate_on_dev(self) -> float:
        raise NotImplementedError

    def evaluate_on_test(self) -> float:
        raise NotImplementedError


class VoxCeleb1ModelEvaluationExperiment(ModelEvaluationExperiment):

    def __init__(self, model_path: str, nfeat: int, distance: Distance, batch_size: int):
        self.model = SpeakerNet(nfeat, sample_rate=16000, window=200)
        model_loader = ModelLoader(model_path)
        loss_name = model_loader.get_trained_loss()
        model_loader.load(self.model, loss_name)
        self.model = self.model.to_prediction_model().to(DEVICE)
        dataset = VoxCeleb1(batch_size, segment_size_millis=200)
        self.evaluator = SpeakerVerificationEvaluator(DEVICE, batch_size, distance,
                                                      eval_interval=0, config=dataset.config)

    def evaluate_on_dev(self) -> float:
        inverse_eer = self.evaluator.eval(self.model, partition='development')
        return 1 - inverse_eer

    def evaluate_on_test(self) -> float:
        inverse_eer = self.evaluator.eval(self.model, partition='test')
        return 1 - inverse_eer


class SemEvalModelEvaluationExperiment(ModelEvaluationExperiment):

    def __init__(self, model_path: str, nfeat: int, data_path: str, word2vec_path: str,
                 vocab_path: str, distance: Distance, log_interval: int, batch_size: int, baseline=False):

        model_loader = ModelLoader(model_path)
        loss_name = model_loader.get_trained_loss()

        # The augmentation is only done for the training set, so it doesn't matter which one we choose.
        # Here I'm choosing NoAugmentation allowing redundancy because it's the cheapest to compute
        augmentation = NoAugmentation(allow_redundancy=True)
        # All partition types (should!) format the data in the same manner when it comes to dev or test
        # so it doesn't matter what loss we choose here
        partition_factory = SemEvalPartitionFactory(loss=loss_name, batch_size=batch_size)
        self.dataset = SemEval(data_path, word2vec_path, vocab_path, augmentation, partition_factory)

        # We can use any mode but ConcatMode, because we don't want to concatenate our 2 embeddings when evaluating
        # Similarly, since we're just testing, we don't care about the loss module
        loss_module = STSBaselineClassifier(nfeat) if baseline else None
        mode = ConcatSTSForwardMode() if baseline else PairSTSForwardMode()
        self.model = SemanticNet(DEVICE, nfeat, self.dataset.vocab, mode=mode, loss_module=loss_module)
        model_loader.load(self.model, loss_name)
        if not baseline:
            self.model = self.model.to_prediction_model()
        self.model = self.model.to(DEVICE)

        self.dev_evaluator, self.test_evaluator = None, None
        self.log_interval = log_interval
        self.distance = distance
        self.baseline = baseline

    def _build_evaluator(self, partition):
        if self.baseline:
            return STSBaselineEvaluator(DEVICE, partition, LogitsSpearmanMetric(),
                                        batch_transforms=[DeviceMapperTransform(DEVICE)],
                                        callbacks=[TestLogger(self.log_interval, partition.nbatches())])
        else:
            return STSEmbeddingEvaluator(DEVICE, partition, DistanceSpearmanMetric(self.distance),
                                         batch_transforms=[DeviceMapperTransform(DEVICE)],
                                         callbacks=[TestLogger(self.log_interval, partition.nbatches())])

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
