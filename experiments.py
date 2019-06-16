from models import SpeakerNet, SemanticNet
from distances import Distance
from losses.base import ModelLoader, DeviceMapperTransform, TestLogger
from datasets.voxceleb import VoxCeleb1
from datasets.semeval import SemEval
from metrics import SpeakerVerificationEvaluator, STSEmbeddingEvaluator, DistanceSpearmanMetric
from utils import DEVICE


class ModelEvaluationExperiment:

    def evaluate_on_dev(self) -> float:
        raise NotImplementedError

    def evaluate_on_test(self) -> float:
        raise NotImplementedError


class VoxCeleb1ModelEvaluationExperiment(ModelEvaluationExperiment):

    def __init__(self, model_path: str, nfeat: int, distance: Distance, batch_size: int):
        self.model = SpeakerNet(nfeat, sample_rate=16000, window=200)
        ModelLoader(None).load(self.model, model_path)
        self.model = self.model.to_prediction_model().to(DEVICE)
        dataset = VoxCeleb1(batch_size, segment_size_millis=200)
        self.evaluator = SpeakerVerificationEvaluator(DEVICE, batch_size, distance,
                                                      eval_interval=0, config=dataset.config)

    def evaluate_on_dev(self) -> float:
        inverse_eer = self.evaluator.eval(self.model, partition='dev')
        return 1 - inverse_eer

    def evaluate_on_test(self) -> float:
        inverse_eer = self.evaluator.eval(self.model, partition='test')
        return 1 - inverse_eer


class SemEvalModelEvaluationExperiment(ModelEvaluationExperiment):

    def __init__(self, model_path: str, nfeat: int, data_path: str, word2vec_path: str,
                 vocab_path: str, distance: Distance, log_interval: int, batch_size: int):
        # The mode and threshold are not important since we won't use any of this for training
        self.dataset = SemEval(data_path, word2vec_path, vocab_path, batch_size)
        # We can pass any mode but 'baseline' here, because we don't want to concatenate both embeddings in the end
        self.model = SemanticNet(DEVICE, nfeat, self.dataset.vocab, mode='pairs')
        ModelLoader(None).load(self.model, model_path)
        self.model = self.model.to_prediction_model().to(DEVICE)
        self.dev_evaluator, self.test_evaluator = None, None
        self.log_interval = log_interval
        self.distance = distance

    def _build_evaluator(self, partition):
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
