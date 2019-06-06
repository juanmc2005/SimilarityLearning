import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import spearmanr
from pyannote.audio.embedding.extraction import SequenceEmbedding
from pyannote.database import get_protocol, get_unique_identifier
from pyannote.metrics.binary_classification import det_curve
from pyannote.core.utils.distance import cdist
from pyannote.core import Timeline
from distances import Distance


class KNNAccuracyMetric:
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

    def fit(self, embeddings, y):
        self.knn.fit(embeddings, y)

    def calculate_batch(self, embeddings, logits, y):
        predicted = self.knn.predict(embeddings)
        self.correct += (predicted == y).sum()
        self.total += y.shape[0]

    def get(self):
        metric = self.correct / self.total
        self.correct, self.total = 0, 0
        return metric


class SpearmanMetric:
    def __init__(self):
        self.predictions, self.targets = [], []

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


class SpeakerValidationConfig:

    def __init__(self, protocol_name, feature_extraction, preprocessors, duration):
        self.protocol_name = protocol_name
        self.feature_extraction = feature_extraction
        self.preprocessors = preprocessors
        self.duration = duration


class EERMetric:

    @staticmethod
    def get_hash(trial_file):
        uri = get_unique_identifier(trial_file)
        try_with = trial_file['try_with']
        if isinstance(try_with, Timeline):
            segments = tuple(try_with)
        else:
            segments = (try_with,)
        return hash((uri, segments))

    def __init__(self, model, device: str, batch_size: int, distance: Distance, config: SpeakerValidationConfig):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.distance = distance
        self.config = config

    def fit(self, embeddings, y):
        pass

    def calculate_batch(self, embeddings, logits, y):
        pass

    def get(self):
        # initialize embedding extraction
        sequence_embedding = SequenceEmbedding(model=self.model,
                                               feature_extraction=self.config.feature_extraction,
                                               duration=self.config.duration,
                                               step=.5 * self.config.duration,
                                               batch_size=self.batch_size,
                                               device=self.device)
        protocol = get_protocol(self.config.protocol_name, progress=False, preprocessors=self.config.preprocessors)

        y_true, y_pred, cache = [], [], {}

        for trial in protocol.development_trial():

            # compute embedding for file1
            file1 = trial['file1']
            hash1 = self.get_hash(file1)
            if hash1 in cache:
                emb1 = cache[hash1]
            else:
                emb1 = sequence_embedding.crop(file1, file1['try_with'])
                emb1 = np.mean(np.stack(emb1), axis=0, keepdims=True)
                cache[hash1] = emb1

            # compute embedding for file2
            file2 = trial['file2']
            hash2 = self.get_hash(file2)
            if hash2 in cache:
                emb2 = cache[hash2]
            else:
                emb2 = sequence_embedding.crop(file2, file2['try_with'])
                emb2 = np.mean(np.stack(emb2), axis=0, keepdims=True)
                cache[hash2] = emb2

            # compare embeddings
            dist = cdist(emb1, emb2, metric=self.distance.to_sklearn_metric())[0, 0]
            y_pred.append(dist)
            y_true.append(trial['reference'])

        _, _, _, eer = det_curve(np.array(y_true), np.array(y_pred), distances=True)
        return eer