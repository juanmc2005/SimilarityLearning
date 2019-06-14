from models import SpeakerNet
from distances import Distance
from losses.base import ModelLoader
from datasets.voxceleb import VoxCeleb1
from metrics import SpeakerVerificationEvaluator


class VoxCeleb1ModelEvaluationExperiment:

    def __init__(self, model_path: str, nfeat: int, distance: Distance, device: str, batch_size: int):
        self.model = SpeakerNet(nfeat, sample_rate=16000, window=200)
        ModelLoader(None).load(self.model, model_path)
        self.model = self.model.to_prediction_model()
        self.dataset = VoxCeleb1(batch_size, segment_size_millis=200)
        self.evaluator = SpeakerVerificationEvaluator(device, batch_size, distance,
                                                      eval_interval=0, config=self.dataset.config)

    def evaluate_on_dev(self):
        inverse_eer = self.evaluator.eval(self.model, partition='dev')
        return 1 - inverse_eer

    def evaluate_on_test(self):
        inverse_eer = self.evaluator.eval(self.model, partition='test')
        return 1 - inverse_eer
