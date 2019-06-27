from core.base import TrainingListener
from metrics import Metric


class TrainingMetricCalculator(TrainingListener):

    def __init__(self, name: str, metric: Metric, file_path: str = None):
        self.name = name
        self.metric = metric
        self.file_path = file_path
        if file_path is not None:
            open(file_path, 'w').close()

    def on_after_gradients(self, epoch, ibatch, feat, logits, y, loss):
        self.metric.calculate_batch(feat, logits, y)

    def on_after_epoch(self, epoch, model, loss_fn, optim):
        metric = self.metric.get()
        print(f"[{self.name}: {metric:.6f}]")
        if self.file_path is not None:
            with open(self.file_path, 'a') as file:
                file.write(f"{metric}\n")
