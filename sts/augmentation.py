import numpy as np
import math
from sts import utils


class SemEvalAugmentationStrategy:

    def nclass(self):
        return None

    def augment(self, train_sents_a: list, train_sents_b: list, train_scores: list) -> np.ndarray:
        raise NotImplementedError


class ScoreFormatter:

    def format(self, scores):
        raise NotImplementedError


class ProbabilitiesScoreFormatter(ScoreFormatter):

    def format(self, scores):
        labels = []
        for s in scores:
            ceil = int(math.ceil(s))
            floor = int(math.floor(s))
            tmp = [0, 0, 0, 0, 0, 0]
            if floor != ceil:
                tmp[ceil] = s - floor
                tmp[floor] = ceil - s
            else:
                tmp[floor] = 1
            labels.append(tmp)
        return labels


class BinaryScoreFormatter(ScoreFormatter):

    def __init__(self, threshold: float):
        self.threshold = threshold

    def format(self, scores):
        return [0 if s >= self.threshold else 1 for s in scores]


class NoAugmentation(SemEvalAugmentationStrategy):

    @staticmethod
    def remove_pairs_with_score(a: list, b: list, sim: list, targets: tuple):
        anew, bnew, simnew = [], [], []
        for i in range(len(a)):
            if math.floor(sim[i]) not in targets:
                anew.append(a[i])
                bnew.append(b[i])
                simnew.append(sim[i])
        return anew, bnew, simnew

    def __init__(self, allow_redundancy: bool = False, remove_scores: tuple = (), formatter: ScoreFormatter = None):
        self.allow_redundancy = allow_redundancy
        self.remove_scores = remove_scores
        self.formatter = formatter

    def augment(self, train_sents_a: list, train_sents_b: list, train_scores: list) -> np.ndarray:
        atrain, btrain, simtrain = self.remove_pairs_with_score(train_sents_a, train_sents_b,
                                                                train_scores, self.remove_scores)
        print(f"Total Train Pairs: {len(atrain)}")
        if self.allow_redundancy:
            pairs = zip(atrain, btrain)
            sim = simtrain
        else:
            unique_train_data = list(set(zip(atrain, btrain, simtrain)))
            pairs = [(x1, x2) for x1, x2, _ in unique_train_data]
            sim = [y for _, _, y in unique_train_data]
            print(f"Unique Train Pairs: {len(unique_train_data)}")

        sim = self.formatter.format(sim) if self.formatter is not None else sim
        print(f"Redundancy in the training set: {'YES' if self.allow_redundancy else 'NO'}")
        return np.array(list(zip(pairs, sim)))


class ClusterAugmentation(SemEvalAugmentationStrategy):

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.classes = None

    def _clusterize(self, sents_a, sents_b, scores):
        segment_a = utils.SemEvalSegment(sents_a)
        segment_b = utils.SemEvalSegment(sents_b)
        return segment_a.clusters(segment_b, scores, self.threshold)

    def nclass(self):
        return self.classes

    def augment(self, train_sents_a: list, train_sents_b: list, train_scores: list) -> np.ndarray:
        sents_a, sents_b, scores = utils.unique_pairs(train_sents_a, train_sents_b, train_scores)

        clusters = self._clusterize(sents_a, sents_b, scores)
        self.classes = len(clusters)

        train_sents, train_sents_raw = [], []
        for i, cluster in enumerate(clusters):
            for sent in cluster:
                if sent in train_sents_a or sent in train_sents_b:
                    train_sents.append((sent.split(' '), i))
                    train_sents_raw.append(sent)
        train_sents = np.array(train_sents)

        print(f"Unique sentences used for clustering: {len(set(sents_a + sents_b))}")
        print(f"Total Train Sentences: {len(set(train_sents_a + train_sents_b))}")
        print(f"Train Sentences Kept: {len(set(train_sents_raw))}")
        print(f"N Clusters: {self.classes}")
        print(f"Max Cluster Size: {max([len(cluster) for cluster in clusters])}")
        print(f"Mean Cluster Size: {np.mean([len(cluster) for cluster in clusters])}")

        return train_sents


class PairAugmentation(SemEvalAugmentationStrategy):

    def __init__(self, threshold):
        # Threshold can be a pair (low, high) or a float, which is the same as (value, value)
        self.threshold = threshold

    def _pairs(self, sents_a, sents_b, scores, threshold):
        segment_a = utils.SemEvalSegment(sents_a)
        segment_b = utils.SemEvalSegment(sents_b)
        pos, neg = utils.pairs(segment_a, segment_b, scores, threshold)
        data = [((s1, s2), 0) for s1, s2 in pos] + [((s1, s2), 1) for s1, s2 in neg]
        return np.array([((s1.split(' '), s2.split(' ')), y) for (s1, s2), y in data])

    def augment(self, train_sents_a: list, train_sents_b: list, train_scores: list) -> np.ndarray:
        train_sents = self._pairs(train_sents_a, train_sents_b, train_scores, self.threshold)
        print(f"Original Train Pairs: {len(train_sents_a)}")
        print(f"Original Unique Train Pairs: {len(set(zip(train_sents_a, train_sents_b)))}")
        print(f"Total Train Pairs: {len(train_sents)}")
        print(f"+ Train Pairs: {len([y for _, y in train_sents if y == 0])}")
        print(f"- Train Pairs: {len([y for _, y in train_sents if y == 1])}")
        return train_sents


class TripletAugmentation(SemEvalAugmentationStrategy):

    def __init__(self, threshold):
        # Threshold can be a pair (low, high) or a float, which is the same as (value, value)
        self.threshold = threshold

    def _triplets(self, sents_a, sents_b, scores):
        segment_a = utils.SemEvalSegment(sents_a)
        segment_b = utils.SemEvalSegment(sents_b)
        unique_sents = set(sents_a + sents_b)
        pos, neg = utils.pairs(segment_a, segment_b, scores, self.threshold)
        anchors, positives, negatives = utils.triplets(unique_sents, pos, neg)
        return np.array([(a.split(' '), p.split(' '), n.split(' '))
                         for a, p, n in zip(anchors, positives, negatives)])

    def augment(self, train_sents_a: list, train_sents_b: list, train_scores: list):
        return self._triplets(train_sents_a, train_sents_b, train_scores)


class SemEvalAugmentationStrategyFactory:

    def __init__(self, loss: str, threshold=2.5, allow_redundancy: bool = False, remove_scores: tuple = ()):
        self.loss = loss
        self.threshold = threshold
        self.allow_redundancy = allow_redundancy
        self.remove_scores = remove_scores

    def new(self):
        if self.loss == 'kldiv':
            return NoAugmentation(self.allow_redundancy, self.remove_scores, ProbabilitiesScoreFormatter())
        elif self.loss == 'contrastive':
            # return PairAugmentation(self.threshold)
            return NoAugmentation(self.allow_redundancy, self.remove_scores, BinaryScoreFormatter(self.threshold))
        elif self.loss == 'triplet':
            return TripletAugmentation(self.threshold)
        else:
            # Softmax based loss
            return ClusterAugmentation(self.threshold)
