import numpy as np
from core.base import TestListener
from distances import Distance
import datasets.ami as ami
import visual_utils


class MNISTVisualizer(TestListener):

    def __init__(self, base_dir, loss_name, param_desc=None):
        super(MNISTVisualizer, self).__init__()
        self.base_dir = base_dir
        self.loss_name = loss_name
        self.param_desc = param_desc

    def on_best_accuracy(self, epoch, model, loss_fn, optim, accuracy, feat, y):
        plot_name = f"embeddings-epoch-{epoch}"
        plot_title = f"{self.loss_name} (Epoch {epoch}) - {accuracy * 100:.2f} Accuracy"
        if self.param_desc is not None:
            plot_title += f" - {self.param_desc}"
        print(f"Saving plot as {plot_name}")
        visual_utils.visualize(feat, y, plot_title,
                               legend=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                               dir_path=self.base_dir,
                               filename=plot_name)


class AMIVisualizer(TestListener):

    def __init__(self, base_dir, loss_name, param_desc=None):
        super(AMIVisualizer, self).__init__()
        self.base_dir = base_dir
        self.loss_name = loss_name
        self.param_desc = param_desc

    def on_best_accuracy(self, epoch, model, loss_fn, optim, accuracy, feat, y):
        plot_name = f"embeddings-epoch-{epoch}"
        plot_title = f"{self.loss_name} (Epoch {epoch}) - {accuracy:.2f} Macro F1"
        if self.param_desc is not None:
            plot_title += f" - {self.param_desc}"
        print(f"Saving plot as {plot_name}")
        visual_utils.visualize(feat, y, plot_title,
                               legend=[ami.id2label[i] for i in range(5)],
                               dir_path=self.base_dir,
                               filename=plot_name)


class TSNEVisualizer(TestListener):

    def __init__(self, base_dir, loss: str, distance: Distance, param_desc: str = None):
        self.base_dir = base_dir
        self.loss = loss
        self.distance = distance
        self.param_desc = param_desc

    def on_after_test(self, epoch, feat_test, y_test, metric_value):
        plot_name = f"embeddings-{self.loss}"
        plot_title = f"{self.loss.capitalize()} Embeddings"
        if self.param_desc is not None:
            plot_title += f" - {self.param_desc}"
        print(f"Saving TSNE plot to {plot_name}")
        unique_feat = np.unique(feat_test, axis=0)
        visual_utils.visualize_tsne_neighbors(unique_feat, None, self.distance, plot_title, self.base_dir, plot_name)


class SpeakerDistanceVisualizer(TestListener):

    def __init__(self, base_dir):
        self.base_dir = base_dir

    def on_after_test(self, epoch, feat_test, y_test, metric_value):
        title = f'Distance distribution for dev speakers (Epoch {epoch}) - EER {metric_value:.3f}'
        filename = f'speaker-dists-epoch={epoch}'
        print(f"Saving speaker distances plot as {filename}")
        visual_utils.plot_pred_hists(feat_test, y_test, title, self.base_dir, filename)