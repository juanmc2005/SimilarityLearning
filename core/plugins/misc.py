from core.base import TrainingListener
from metrics import Metric


class TrainingMetricCalculator(TrainingListener):

    def __init__(self, name: str, metric: Metric):
        self.name = name
        self.metric = metric

    def on_after_gradients(self, epoch, ibatch, feat, logits, y, loss):
        self.metric.calculate_batch(feat, logits, y)

    def on_after_epoch(self, epoch, model, loss_fn, optim):
        print(f"[{self.name}: {self.metric.get()}]")
