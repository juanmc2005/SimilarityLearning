from experiments.base import ModelEvaluationExperiment
from datasets.voxceleb import VoxCeleb1
from models import SpeakerNet
from losses.base import ModelLoader
from metrics import SpeakerVerificationEvaluator
from distances import Distance
import common


class VoxCeleb1ModelEvaluationExperiment(ModelEvaluationExperiment):

    def __init__(self, model_path: str, nfeat: int, distance: Distance, batch_size: int):
        self.model = SpeakerNet(nfeat, sample_rate=16000, window=200)
        model_loader = ModelLoader(model_path)
        loss_name = model_loader.get_trained_loss()
        model_loader.load(self.model, loss_name)
        self.model = self.model.to_prediction_model().to(common.DEVICE)
        config = VoxCeleb1.config(segment_size_s=0.2)
        self.evaluator = SpeakerVerificationEvaluator(batch_size, distance, eval_interval=0, config=config)

    def evaluate_on_dev(self, plot: bool) -> float:
        inverse_eer, _, _ = self.evaluator.eval(self.model, partition='development')
        return 1 - inverse_eer

    def evaluate_on_test(self) -> float:
        inverse_eer, _, _ = self.evaluator.eval(self.model, partition='test')
        return 1 - inverse_eer
