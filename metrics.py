import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix
from scipy.stats import spearmanr
from distances import Distance
import core.base as base
import common
from collections import Counter


class Metric:

    def __str__(self):
        raise NotImplementedError("A metric must implement the method '__str__'")

    def fit(self, embeddings, y):
        raise NotImplementedError("A metric must implement the method 'fit'")

    def calculate_batch(self, embeddings, logits, y):
        raise NotImplementedError("A metric must implement the method 'calculate_batch'")

    def get(self):
        raise NotImplementedError("A metric must implement the method 'get'")


class KNNAccuracyMetric(Metric):
    """ TODO update docs
    Abstracts the accuracy calculation strategy. It uses a K Nearest Neighbors
        classifier fit with the embeddings produced for the training set,
        to determine to which class a given test embedding is assigned to.
    :param train_embeddings: a tensor of shape (N, d), where
            N = training set size
            d = embedding dimension
    :param train_y: a non one-hot encoded tensor of labels for the train embeddings
    :param distance: a Distance object for the KNN classifier
    """

    def __init__(self, distance):
        self.knn = KNeighborsClassifier(n_neighbors=1, metric=distance.to_sklearn_metric())
        self.correct, self.total = 0, 0
        self.preds, self.y = [], []

    def __str__(self):
        return 'KNN Accuracy'

    def fit(self, embeddings, y):
        self.knn.fit(embeddings, y)

    def calculate_batch(self, embeddings, logits, y):
        predicted = self.knn.predict(embeddings)
        self.correct += (predicted == y).sum()
        self.total += y.shape[0]
        self.preds.extend(predicted)
        self.y.extend(y)

    def get(self):
        print(f"Confusion Matrix:\n{confusion_matrix(self.y, self.preds)}")
        metric = self.correct / self.total
        self.correct, self.total = 0, 0
        return metric


class KNNF1ScoreMetric(Metric):

    def __init__(self, distance, neighbors: int = 1):
        self.neighbors = neighbors
        self.knn = KNeighborsClassifier(n_neighbors=neighbors, metric=distance.to_sklearn_metric())
        self.preds, self.y = [], []
        self.train_class_counter = None
        self.train_y = None

    def __str__(self):
        return 'KNN Macro F1-Score'

    def fit(self, embeddings, y):
        self.knn.fit(embeddings, y)
        self.train_y = y
        self.train_class_counter = Counter(y)

    def calculate_batch(self, embeddings, logits, y):
        if self.neighbors == 1:
            predicted = self.knn.predict(embeddings)
        else:
            _, idx = self.knn.kneighbors(embeddings)
            predicted = []
            for neigh_labels in self.train_y[idx]:
                counter = Counter(neigh_labels)
                max_vote, max_label = 0, -1
                for label in counter:
                    vote = counter[label] / self.train_class_counter[label]
                    if vote > max_vote:
                        max_vote = vote
                        max_label = label
                predicted.append(max_label)
            predicted = np.array(predicted)
        self.preds.extend(predicted)
        self.y.extend(y)

    def get(self):
        metric = f1_score(self.y, self.preds, average='macro')
        print(f"Confusion Matrix:\n{confusion_matrix(self.y, self.preds)}")
        self.preds, self.y = [], []
        return metric


class LogitsAccuracyMetric(Metric):

    def __init__(self):
        self.correct, self.total = 0, 0

    def __str__(self):
        return 'Softmax Accuracy'

    def fit(self, embeddings, y):
        pass

    def calculate_batch(self, embeddings, logits, y):
        pred = logits.argmax(axis=1)
        self.correct += (pred == y).sum()
        self.total += logits.shape[0]

    def get(self):
        metric = self.correct / self.total
        self.correct, self.total = 0, 0
        return metric


class LogitsF1ScoreMetric(Metric):

    def __init__(self):
        self.preds, self.y = [], []

    def __str__(self):
        return 'Macro F1-Score'

    def fit(self, embeddings, y):
        pass

    def calculate_batch(self, embeddings, logits, y):
        predicted = logits.argmax(axis=1)
        self.preds.extend(predicted)
        self.y.extend(y)

    def get(self):
        metric = f1_score(self.y, self.preds, average='macro')
        print(f"Confusion Matrix:\n{confusion_matrix(self.y, self.preds)}")
        self.preds, self.y = [], []
        return metric


class LogitsSpearmanMetric(Metric):

    def __init__(self):
        self.predictions, self.targets = [], []

    def __str__(self):
        return 'Logit Spearman'

    def fit(self, embeddings, y):
        pass

    def calculate_batch(self, embeddings, logits, y):
        output = np.exp(logits)
        predicted = []
        for i in range(output.shape[0]):
            predicted.append(0 * output[i, 0] +
                             1 * output[i, 1] +
                             2 * output[i, 2] +
                             3 * output[i, 3] +
                             4 * output[i, 4] +
                             5 * output[i, 5])
        self.predictions.extend(predicted)
        self.targets.extend(list(y))

    def get(self):
        metric = spearmanr(self.predictions, self.targets)[0]
        self.predictions, self.targets = [], []
        return metric


class DistanceSpearmanMetric(Metric):

    def __init__(self, distance: Distance):
        self.distance = distance
        self.similarity, self.targets = [], []

    def __str__(self):
        return 'Distance Spearman'

    def fit(self, embeddings, y):
        pass

    def calculate_batch(self, embeddings, logits, y):
        embeddings1, embeddings2 = embeddings
        self.similarity.extend([-self.distance.dist(embeddings1[i, :].unsqueeze(0), embeddings2[i, :].unsqueeze(0))
                                for i in range(embeddings1.size(0))])
        self.targets.extend(list(y))

    def get(self):
        metric = spearmanr(self.similarity, self.targets)[0]
        self.similarity, self.targets = [], []
        return metric


class SNLIFixedThresholdAccuracyMetric(Metric):

    def __init__(self, distance: Distance, label2id: dict):
        self.distance = distance
        self.label2id = label2id
        self.distances, self.targets = np.array([]), []

    def _norm_dist_to_label(self, d: float) -> str:
        if d < 0.33:
            return self.label2id['entailment']
        elif d > 0.66:
            return self.label2id['contradiction']
        else:
            return self.label2id['neutral']

    def __str__(self):
        return 'Distance Accuracy'

    def fit(self, embeddings, y):
        pass

    def calculate_batch(self, embeddings, logits, y):
        embeddings1, embeddings2 = embeddings
        dists = self.distance.dist(embeddings1, embeddings2).detach().cpu().numpy()
        self.distances = np.concatenate((self.distances, dists), axis=None)
        self.targets.extend(list(y))

    def get(self):
        # Min Max Normalization (minimum is always 0)
        preds = self.distances / self.distance.max if self.distance.max is not None else 1
        # Use distances to make predictions
        preds = np.array([self._norm_dist_to_label(d) for d in preds])
        y = np.array(self.targets)
        # Calculate accuracy
        correct = (preds == y).sum()
        total = y.shape[0]
        metric = correct / total
        # Reset internal state and print confusion matrix
        self.distances, self.targets = np.array([]), []
        print(f"Confusion Matrix:\n{confusion_matrix(y, preds)}")
        return metric


class SNLIGridSearchAccuracyMetric(Metric):

    def __init__(self, distance: Distance, label2id: dict, t_lows, t_highs):
        self.distance = distance
        self.label2id = label2id
        self.t_lows = t_lows
        self.t_highs = t_highs
        self.distances, self.targets = np.array([]), []

    def _norm_dist_to_label(self, d: float, t_low: float, t_high: float) -> str:
        if d < t_low:
            return self.label2id['entailment']
        elif d > t_high:
            return self.label2id['contradiction']
        else:
            return self.label2id['neutral']

    def __str__(self):
        return 'Distance Accuracy'

    def fit(self, embeddings, y):
        pass

    def calculate_batch(self, embeddings, logits, y):
        embeddings1, embeddings2 = embeddings
        dists = self.distance.dist(embeddings1, embeddings2).detach().cpu().numpy()
        self.distances = np.concatenate((self.distances, dists), axis=None)
        self.targets.extend(list(y))

    def get(self):
        best = None
        dropped = 0
        for t_low in self.t_lows:
            for t_high in self.t_highs:
                # Fail safe for invalid threshold values
                if t_high <= t_low:
                    dropped += 1
                    break
                # Use distances to make predictions
                preds = np.array([self._norm_dist_to_label(d, t_low, t_high) for d in self.distances])
                y = np.array(self.targets)
                # Calculate accuracy
                correct = (preds == y).sum()
                total = y.shape[0]
                metric = correct / total
                if best is None or metric > best[0]:
                    best = (metric, t_low, t_high, y, preds)
        # Reset internal state and print confusion matrix
        self.distances, self.targets = np.array([]), []
        metric, t_low, t_high, y, preds = best
        print(f"Optimal Thresholds: low={t_low} high={t_high}")
        print(f"Confusion Matrix:\n{confusion_matrix(y, preds)}")
        print(f"Invalid Combinations: {dropped}")
        return metric


class SpeakerValidationConfig:

    def __init__(self, protocol_name, preprocessors, duration):
        self.protocol_name = protocol_name
        self.preprocessors = preprocessors
        self.duration = duration


class VerificationTestCallback:

    def on_evaluation_finished(self, epoch, eer, distances, y_true, fpr, fnr, partition: str):
        raise NotImplementedError


# TODO These evaluator classes need to be refactored, they share a lot of code

class ClassAccuracyEvaluator(base.TrainingListener):

    def __init__(self, device, loader, metric, partition_name, callbacks=None):
        super(ClassAccuracyEvaluator, self).__init__()
        self.device = device
        self.loader = loader
        self.metric = metric
        self.partition_name = partition_name
        self.callbacks = callbacks if callbacks is not None else []
        self.feat_train, self.y_train = None, None
        self.results = []
        self.best_metric, self.best_epoch, self.last_metric = 0, -1, 0

    def eval(self, model):
        model.eval()
        feat_test, logits_test, y_test = [], [], []
        for cb in self.callbacks:
            cb.on_before_test()
        with torch.no_grad():
            for i in range(self.loader.nbatches()):
                x, y = next(self.loader)

                if isinstance(x, torch.Tensor):
                    x = x.to(common.DEVICE)
                if isinstance(y, torch.Tensor):
                    y = y.to(common.DEVICE)

                # Feed Forward
                feat, logits = model(x, y)
                feat = feat.detach().cpu().numpy()
                if logits is not None:
                    logits = logits.detach().cpu().numpy()
                y = y.detach().cpu().numpy()

                # Track accuracy
                feat_test.append(feat)
                if logits is not None:
                    logits_test.append(logits)
                y_test.append(y)
                self.metric.calculate_batch(feat, logits, y)

                for cb in self.callbacks:
                    cb.on_batch_tested(i, feat)

        feat_test, y_test = np.concatenate(feat_test), np.concatenate(y_test)
        return feat_test, y_test

    def on_before_epoch(self, epoch):
        self.feat_train, self.y_train = [], []

    def on_after_gradients(self, epoch, ibatch, feat, logits, y, loss):
        self.feat_train.append(feat.detach().cpu().numpy())
        self.y_train.append(y.detach().cpu().numpy())

    def on_after_epoch(self, epoch, model, loss_fn, optim):
        feat_train = np.concatenate(self.feat_train)
        y_train = np.concatenate(self.y_train)
        self.metric.fit(feat_train, y_train)
        feat_test, y_test = self.eval(model)
        self.last_metric = self.metric.get()
        self.results.append(self.last_metric)
        for cb in self.callbacks:
            cb.on_after_test(epoch, feat_test, y_test, self.last_metric)
        print(f"[{self.partition_name.capitalize()} {self.metric}: {self.last_metric:.6f}]")
        if self.best_epoch != -1:
            print(f"Best until now: {self.best_metric:.6f}, at epoch {self.best_epoch}")
        if self.last_metric > self.best_metric:
            self.best_metric = self.last_metric
            self.best_epoch = epoch
            print(f'New Best {self.partition_name.capitalize()} {self.metric}!')
            for cb in self.callbacks:
                cb.on_best_accuracy(epoch, model, loss_fn, optim, self.last_metric, feat_test, y_test)


class STSEmbeddingEvaluator(base.TrainingListener):

    def __init__(self, device, loader, metric, partition_name, callbacks=None):
        super(STSEmbeddingEvaluator, self).__init__()
        self.device = device
        self.loader = loader
        self.metric = metric
        self.partition_name = partition_name
        self.callbacks = callbacks if callbacks is not None else []
        self.best_metric, self.best_epoch, self.last_metric = 0, -1, 0

    def eval(self, model):
        model.eval()
        phrases, feat_test, y_test = [], [], []
        for cb in self.callbacks:
            cb.on_before_test()
        with torch.no_grad():
            for i in range(self.loader.nbatches()):
                x, y = next(self.loader)

                for pair in x:
                    phrases.append(' '.join([word for word in pair[0] if word != 'null']))
                for pair in x:
                    phrases.append(' '.join([word for word in pair[1] if word != 'null']))

                if isinstance(y, torch.Tensor):
                    y = y.to(common.DEVICE)

                # Feed Forward
                feat = model(x)

                # In evaluation mode, we always receive 2 phrases and no logits
                feat1 = feat[0].detach().cpu().numpy()
                feat2 = feat[1].detach().cpu().numpy()
                y = y.detach().cpu().numpy()

                # Track accuracy
                feat_test.append(feat1)
                feat_test.append(feat2)
                y_test.append(y)
                self.metric.calculate_batch(feat, None, y)

                for cb in self.callbacks:
                    cb.on_batch_tested(i, feat)

        feat_test = np.concatenate(feat_test)
        y_test = np.concatenate(y_test)
        return phrases, feat_test, y_test

    def on_after_epoch(self, epoch, model, loss_fn, optim):
        _, feat_test, y_test = self.eval(model.to_prediction_model())
        self.last_metric = self.metric.get()
        for cb in self.callbacks:
            cb.on_after_test(epoch, feat_test, y_test, self.last_metric)
        print(f"--------------- Epoch {epoch:02d} Results ---------------")
        print(f"{self.partition_name.capitalize()} {self.metric}: {self.last_metric:.6f}")
        if self.best_epoch != -1:
            print(f"Best until now: {self.best_metric:.6f}, at epoch {self.best_epoch}")
        print("------------------------------------------------")
        if self.last_metric > self.best_metric:
            self.best_metric = self.last_metric
            self.best_epoch = epoch
            print(f'New Best {self.partition_name.capitalize()} {self.metric}!')
            for cb in self.callbacks:
                cb.on_best_accuracy(epoch, model, loss_fn, optim, self.last_metric, feat_test, y_test)


class STSBaselineEvaluator(base.TrainingListener):

    def __init__(self, device, loader, metric, partition_name: str = 'dev', callbacks=None):
        super(STSBaselineEvaluator, self).__init__()
        self.partition_name = partition_name
        self.device = device
        self.loader = loader
        self.metric = metric
        self.callbacks = callbacks if callbacks is not None else []
        self.best_metric, self.best_epoch, self.last_metric = 0, -1, 0

    def eval(self, model):
        model.eval()
        phrases, feat_test, logits_test, y_test = [], [], [], []
        for cb in self.callbacks:
            cb.on_before_test()
        with torch.no_grad():
            for i in range(self.loader.nbatches()):
                x, y = next(self.loader)

                for pair in x:
                    phrases.append(' '.join([word for word in pair[0] if word != 'null']))
                for pair in x:
                    phrases.append(' '.join([word for word in pair[1] if word != 'null']))

                if isinstance(y, torch.Tensor):
                    y = y.to(common.DEVICE)

                # Feed Forward
                feat, logits = model(x, y)

                feat1, feat2 = torch.split(feat, feat.size(1) // 2, dim=1)
                feat1, feat2 = feat1.detach().cpu().numpy(), feat2.detach().cpu().numpy()
                logits = logits.detach().cpu().numpy()
                y = y.detach().cpu().numpy()

                # Track accuracy
                feat_test.append(feat1)
                feat_test.append(feat2)
                logits_test.append(logits)
                y_test.append(y)
                self.metric.calculate_batch(feat, logits, y)

                for cb in self.callbacks:
                    cb.on_batch_tested(i, feat)

        feat_test, y_test = np.concatenate(feat_test), np.concatenate(y_test)
        return phrases, feat_test, y_test

    def on_after_epoch(self, epoch, model, loss_fn, optim):
        phrases, feat_test, y_test = self.eval(model)
        self.last_metric = self.metric.get()
        for cb in self.callbacks:
            cb.on_after_test(epoch, feat_test, y_test, self.last_metric)
        print(f"--------------- Epoch {epoch:02d} Results ---------------")
        print(f"{self.partition_name.capitalize()} {self.metric}: {self.last_metric:.6f}")
        if self.best_epoch != -1:
            print(f"Best until now: {self.best_metric:.6f}, at epoch {self.best_epoch}")
        print("------------------------------------------------")
        if self.last_metric > self.best_metric:
            self.best_metric = self.last_metric
            self.best_epoch = epoch
            print(f'New Best {self.partition_name.capitalize()} {self.metric}!')
            for cb in self.callbacks:
                cb.on_best_accuracy(epoch, model, loss_fn, optim, self.last_metric, feat_test, y_test)
