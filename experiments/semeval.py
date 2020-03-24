from scipy.stats import spearmanr
from models import SemanticNet
from distances import Distance
from core.plugins.storage import ModelLoader
from core.plugins.logging import TestLogger
from losses.wrappers import STSBaselineClassifier
from datasets.semeval import SemEval, SemEvalPartitionFactory
from sts.augmentation import NoAugmentation
from sts.modes import STSForwardMode, PairSTSForwardMode, ConcatSTSForwardMode
from metrics import STSEmbeddingEvaluator, STSBaselineEvaluator, DistanceSpearmanMetric, LogitsSpearmanMetric
from experiments.base import ModelEvaluationExperiment
from common import DEVICE
from gensim.models import Word2Vec
import visual_utils


class SemEvalEvaluationExperiment(ModelEvaluationExperiment):

    def __init__(self, model_loader: ModelLoader, nfeat: int, data_path: str, word2vec_path: str,
                 word2vec_model_path: str, vocab_path: str, distance: Distance, log_interval: int,
                 batch_size: int, base_dir: str):

        self.loss_name = model_loader.get_trained_loss()
        self.dev_evaluator, self.test_evaluator = None, None
        self.log_interval = log_interval
        self.distance = distance
        self.nfeat = nfeat
        self.base_dir = base_dir

        # The augmentation is only done for the training set, so it doesn't matter which one we choose.
        # Here I'm choosing NoAugmentation allowing redundancy because it's the cheapest to compute
        augmentation = NoAugmentation(allow_redundancy=True)
        # All partition types (should!) format the data in the same manner when it comes to dev or test
        # so it doesn't matter what loss we choose here
        partition_factory = SemEvalPartitionFactory(loss=self.loss_name, batch_size=batch_size)
        self.dataset = SemEval(data_path, word2vec_path, vocab_path, augmentation, partition_factory)

        vocab_vec = self.dataset.vocab_vec if word2vec_path is not None else Word2Vec.load(word2vec_model_path).wv

        self.model = SemanticNet(nfeat, 1, self.dataset.vocab, vocab_vec,
                                 mode=self._get_model_mode(),
                                 classifier=self._get_loss_module())
        model_loader.load(self.model, self.loss_name)
        self.model = self._transform_model(self.model)
        self.model = self.model.to(DEVICE)

    def get_dev_evaluator(self):
        if self.dev_evaluator is None:
            self.dev_evaluator = self._build_evaluator(self.dataset.dev_partition())
        return self.dev_evaluator

    def evaluate_on_dev(self, plot: bool) -> float:
        self.get_dev_evaluator()
        phrases, feat_test, y_test = self.dev_evaluator.eval(self.model)
        if plot:
            plot_name = f"embeddings-{self.loss_name}"
            plot_title = f"{self.loss_name.capitalize()} Embeddings"
            visual_utils.visualize_tsne_neighbors(feat_test, phrases, self.distance, plot_title, self.base_dir, plot_name)
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
        return STSEmbeddingEvaluator(DEVICE, partition, DistanceSpearmanMetric(self.distance), str(partition),
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
                                    callbacks=[TestLogger(self.log_interval, partition.nbatches())])

    def _get_model_mode(self) -> STSForwardMode:
        return ConcatSTSForwardMode()

    def _get_loss_module(self):
        return STSBaselineClassifier(self.nfeat)


class SemEvalPredictionsSpearmanExperiment:

    def __init__(self, baseline_model: str, other_model: str, nfeat: int, distance: Distance, log_interval: int,
                 batch_size: int, sem_eval_path: str, vocab_path: str, word2vec_path: str):
        baseline_model_loader = ModelLoader(baseline_model, restore_optimizer=False)
        other_model_loader = ModelLoader(other_model, restore_optimizer=False)
        self.exp_baseline = SemEvalBaselineModelEvaluationExperiment(model_loader=baseline_model_loader,
                                                                     nfeat=nfeat,
                                                                     data_path=sem_eval_path,
                                                                     word2vec_path=word2vec_path,
                                                                     vocab_path=vocab_path,
                                                                     distance=distance,
                                                                     log_interval=log_interval,
                                                                     batch_size=batch_size,
                                                                     base_dir='tmp')
        self.exp_other = SemEvalEmbeddingEvaluationExperiment(model_loader=other_model_loader,
                                                              nfeat=nfeat,
                                                              data_path=sem_eval_path,
                                                              word2vec_path=word2vec_path,
                                                              vocab_path=vocab_path,
                                                              distance=distance,
                                                              log_interval=log_interval,
                                                              batch_size=batch_size,
                                                              base_dir='tmp')

    def compare_dev(self):
        baseline_evaluator = self.exp_baseline.get_dev_evaluator()
        other_evaluator = self.exp_other.get_dev_evaluator()
        _, feat_baseline, _ = baseline_evaluator.eval(self.exp_baseline.model)
        _, feat_other, _ = other_evaluator.eval(self.exp_other.model)
        score, pvalue = spearmanr(baseline_evaluator.metric.predictions, other_evaluator.metric.similarity)
        return score, pvalue

