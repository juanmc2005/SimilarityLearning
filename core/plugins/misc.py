import numpy as np
from core.base import TrainingListener, TestListener
from metrics import Metric
from distances import Distance
from scipy.spatial.distance import pdist


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


class IntraClassDistanceStatLogger(TestListener):

    def __init__(self, distance: Distance, file_path: str):
        self.distance = distance
        self.file_path = file_path
        open(file_path, 'w').close()

    def on_best_accuracy(self, epoch, model, loss_fn, optim, accuracy, feat, y):
        with open(self.file_path, 'a') as out:
            out.write(f"Epoch {epoch}:\n")
            for i in np.unique(y):
                class_dists = pdist(feat[y == i, :], self.distance.to_sklearn_metric())
                out.write(f"Mean distance for class {i}: {np.mean(class_dists)}")
                out.write('\n')
