import random
import numpy as np
from experiments.base import ModelEvaluationExperiment
from datasets.voxceleb import VoxCeleb1
from models import SpeakerNet
from core.plugins.storage import ModelLoader
from metrics import SpeakerVerificationEvaluator
from distances import Distance
import common
import visual_utils as vis


class VoxCeleb1ModelEvaluationExperiment(ModelEvaluationExperiment):

    def __init__(self, model_path: str, nfeat: int, distance: Distance, batch_size: int):
        self.model = SpeakerNet(nfeat, sample_rate=16000, window=200)
        model_loader = ModelLoader(model_path)
        loss_name = model_loader.get_trained_loss()
        model_loader.load(self.model, loss_name)
        self.model = self.model.to_prediction_model().to(common.DEVICE)
        config = VoxCeleb1.config(sample_rate=16000, segment_size_sec=0.2)
        # The partition parameter doesn't matter here because we're passing it at each 'eval' call
        self.evaluator = SpeakerVerificationEvaluator('', batch_size, distance, eval_interval=0, config=config)

    def evaluate_on_dev(self, plot: bool) -> float:
        inverse_eer, _, _ = self.evaluator.eval(self.model, partition='development')
        return 1 - inverse_eer

    def evaluate_on_test(self) -> float:
        inverse_eer, _, _ = self.evaluator.eval(self.model, partition='test')
        return 1 - inverse_eer


class VoxCeleb1TSNEVisualizationExperiment:

    def __init__(self, model_path: str, nfeat: int, distance: Distance):
        self.distance = distance
        self.model = SpeakerNet(nfeat, sample_rate=16000, window=200)
        model_loader = ModelLoader(model_path)
        self.loss_name = model_loader.get_trained_loss()
        model_loader.load(self.model, self.loss_name)
        self.model = self.model.to_prediction_model().to(common.DEVICE)
        self.dataset = VoxCeleb1(batch_size=100, segment_size_millis=200)
        self.partition = self.dataset.dev_partition()

    def visualize_dev(self, nbatches: int, dir_path: str, filename: str):
        self.model.eval()
        all_feat, all_y = [], []
        # Only process first `nbatches` examples in dev
        for batch_idx in range(nbatches):
            x, y = next(self.partition)
            x, y = x.to(common.DEVICE), y.to(common.DEVICE)
            feat = self.model(x)
            all_feat.append(feat.detach().cpu().numpy())
            all_y.append(y.detach().cpu().numpy())
        all_feat, all_y = np.concatenate(all_feat), np.concatenate(all_y)
        print(f"Original embeddings taken from dev: {all_feat.shape[0]}")
        # Sample 10 random fixed classes
        class_sample = [4, 5, 6, 7, 10, 15, 19, 21, 27, 40]
        all_feat_sample, all_y_sample = [], []
        for i in range(all_feat.shape[0]):
            if all_y[i] in class_sample:
                all_feat_sample.append(list(all_feat[i, :]))
                all_y_sample.append(all_y[i])
        all_feat_sample, all_y_sample = np.array(all_feat_sample), np.array(all_y_sample)
        print(f"Remaining embeddings after filtering by class: {all_feat_sample.shape[0]}")
        # Visualize selection
        vis.visualize_tsne_speaker(all_feat_sample, all_y_sample, class_sample, self.distance,
                                   title=f't-SNE Speaker Embeddings - {self.loss_name.capitalize()} - {self.distance}',
                                   dir_path=dir_path,
                                   filename=filename)
