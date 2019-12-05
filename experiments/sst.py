from models import HateNet
from distances import Distance
from core.plugins.storage import ModelLoader
from torch import nn
from datasets.sst import BinarySST
from metrics import ClassAccuracyEvaluator, LogitsAccuracyMetric, KNNAccuracyMetric
from experiments.base import ModelEvaluationExperiment
from common import DEVICE
from gensim.models import Word2Vec
from losses.wrappers import PredictLossModuleWrapper


class BinarySSTEvaluationExperiment(ModelEvaluationExperiment):

    def __init__(self, model_loader: ModelLoader, nfeat: int, data_path: str, vocab_path: str,
                 word2vec_model_path: str, distance: Distance, log_interval: int, batch_size: int,
                 base_dir: str, loss_module: nn.Module):

        self.loss_name = model_loader.get_trained_loss()
        self.dev_evaluator, self.test_evaluator = None, None
        self.log_interval = log_interval
        self.distance = distance
        self.nfeat = nfeat
        self.base_dir = base_dir

        self.dataset = BinarySST(data_path, batch_size, vocab_path, None)
        vocab_vec = Word2Vec.load(word2vec_model_path).wv
        self.model = HateNet(DEVICE, nfeat, self.dataset.vocab, vocab_vec, loss_module)
        model_loader.load(self.model, self.loss_name)
        self.model = self._transform_model(self.model)
        self.model = self.model.to(DEVICE)

    def get_dev_evaluator(self):
        if self.dev_evaluator is None:
            self.dev_evaluator = self._build_evaluator(self.dataset.dev_partition())
        return self.dev_evaluator

    def evaluate_on_dev(self, plot: bool) -> float:
        self.get_dev_evaluator().eval(self.model)
        return self.dev_evaluator.metric.get()

    def evaluate_on_test(self) -> float:
        if self.test_evaluator is None:
            self.test_evaluator = self._build_evaluator(self.dataset.test_partition())
        self.test_evaluator.eval(self.model)
        return self.test_evaluator.metric.get()

    def _build_evaluator(self, partition):
        raise NotImplementedError

    def _transform_model(self, model):
        return model


class BinarySSTEmbeddingEvaluationExperiment(BinarySSTEvaluationExperiment):

    def _build_evaluator(self, partition):
        return ClassAccuracyEvaluator(DEVICE, partition, KNNAccuracyMetric(self.distance), str(partition))


class BinarySSTClassicEvaluationExperiment(BinarySSTEvaluationExperiment):

    def _build_evaluator(self, partition):
        return ClassAccuracyEvaluator(DEVICE, partition, LogitsAccuracyMetric(), str(partition))

    def _transform_model(self, model):
        self.model.loss_module = PredictLossModuleWrapper(self.model.loss_module)
        return self.model
