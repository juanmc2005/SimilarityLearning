import torch
import array
from tqdm import tqdm
from collections import deque


def unique_pairs(xs, ys, scores):
    seen = set()
    xunique, yunique, sunique = [], [], []
    for x, y, score in zip(xs, ys, scores):
        if (x, y) not in seen:
            seen.add((x, y))
            seen.add((y, x))
            xunique.append(x)
            yunique.append(y)
            sunique.append(score)
    return xunique, yunique, sunique


def _anchor_related_sents(anchor, pairs):
    anchor_pairs = [(x, y) for x, y in pairs if x == anchor or y == anchor]
    related = []
    for pos1, pos2 in anchor_pairs:
        if pos1 != anchor:
            related.append(pos1)
        else:
            related.append(pos2)
    return related


def triplets(unique_sents, pos_pairs, neg_pairs):
    anchors, positives, negatives = [], [], []
    for anchor in tqdm(unique_sents, total=len(unique_sents), desc='Generating triplets'):
        for positive in _anchor_related_sents(anchor, pos_pairs):
            for negative in _anchor_related_sents(anchor, neg_pairs):
                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)
    return anchors, positives, negatives


def pairs(segment_a, segment_b, scores, threshold=(2, 3)):
    pos, neg = segment_a.pos_neg_pairs(segment_b, scores, threshold=threshold)
    return set(pos), set(neg)


def vectorized_vocabulary(vocab_path, vec_path):
    tokens = [line.strip() for line in open(vocab_path, 'r')]
    stoi, vectors, dim = load_word_vectors(vec_path)
    vec_vocab = {}
    n_inv, n_oov = 0, 0
    for token in tokens:
        if token in stoi:
            vec_vocab[token] = vectors[stoi[token]]
            n_inv += 1
        else:
            vec_vocab[token] = vectors['oov']
            n_oov += 1
    return vec_vocab, n_inv, n_oov


def load_word_vectors(path):
    itos, vectors, dim = [], array.array(str('d')), None
    with open(path, 'r') as f:
        lines = [line for line in f]
    for line in lines:
        # Explicitly splitting on " " is important, so we don't
        # get rid of Unicode non-breaking spaces in the vectors.
        entries = line.rstrip().split(" ")
        word, entries = entries[0], entries[1:]
        if dim is None and len(entries) > 1:
            dim = len(entries)
        elif len(entries) == 1:
            continue
        elif dim != len(entries):
            raise RuntimeError(
                f"Vector for token {word} has {len(entries)} dimensions, but previously "
                f"read vectors have {dim} dimensions. All vectors must have "
                "the same number of dimensions.")
        vectors.extend(float(x) for x in entries)
        itos.append(word)
    stoi = {word: i for i, word in enumerate(itos)}
    vectors = torch.Tensor(vectors).view(-1, dim)
    return stoi, vectors, dim


class SemEvalSegment:

    @staticmethod
    def find_cluster(clusters, sentence):
        for i, cluster in enumerate(clusters):
            if sentence in cluster:
                return i
        return None

    def __init__(self, sents):
        self.sents = sents

    def append(self, more_sents):
        self.sents.extend(more_sents)

    def clusters(self, other_segment, scores, threshold=2.5):
        """
        Consider scores as edge weights in a graph of sentences.
        Search for positive relationships and build clusters à la Breadth First Search
        """
        clusters = []
        sent_set = set(self.sents)
        for i, s in tqdm(enumerate(sent_set), total=len(sent_set), desc=f"Generating clusters with threshold={threshold}"):
            if SemEvalSegment.find_cluster(clusters, s) is not None:
                continue
            c = [s]
            added = {(i, self)}
            stack = deque()
            for j, x in enumerate(self.sents):
                if j != i and x == s:
                    stack.append((j, other_segment, self, True))
                    added.add((j, other_segment))
            while stack:  # is not empty
                # Retrieve next sentence from the stack
                j, seg, other_seg, equals_last = stack.popleft()
                other_sent = seg.sents[j]
                # Create the pair
                equals_this = False
                if scores[j] >= threshold:
                    if equals_last:
                        # A = B = C --> A = C (We're putting these 2 in the same cluster)
                        c.append(other_sent)
                        equals_this = True
                # Add dependencies
                for k, x in enumerate(seg.sents):
                    if k != j and (k, other_seg) not in added and x == other_sent:
                        stack.append((k, other_seg, seg, equals_this))
                        added.add((k, other_seg))
            if len(c) > 1:
                clusters.append(c)
        return clusters

    def pos_neg_pairs(self, other_segment, scores, threshold=2.5):
        """
        Consider scores as edge weights in a graph of sentences.
        Search for positive and negative pairs Breadth First Search
        """
        if isinstance(threshold, tuple):
            tlow, thigh = threshold
        else:
            tlow = threshold
            thigh = threshold
        pos, neg = [], []
        sent_set = set(self.sents)
        for i, s in tqdm(enumerate(sent_set), total=len(sent_set),
                         desc=f"Generating positive and negative pairs with threshold={threshold}"):
            added = {(i, self)}
            stack = deque()
            for j, x in enumerate(self.sents):
                if j != i and x == s:
                    stack.append((j, other_segment, self, True))
                    added.add((j, other_segment))
            while stack:  # is not empty
                # Retrieve next sentence from the stack
                j, seg, other_seg, equals_last = stack.popleft()
                other_sent = seg.sents[j]
                # Create the pair
                equals_this = False
                candidate = (s, other_sent)
                if scores[j] >= thigh:
                    if equals_last:
                        # A = B = C --> A = C
                        if candidate not in neg:
                            # We don't want to add a pair that we already counted as negative
                            pos.append(candidate)
                            equals_this = True
                    elif candidate not in pos:
                        # We don't want to add a pair that we already counted as positive
                        # A != B and B = C --> A != C
                        neg.append(candidate)
                elif scores[j] <= tlow and equals_last and candidate not in pos:
                    # We don't want to add a pair that we already counted as positive
                    # A = B and B != C --> A != C
                    neg.append(candidate)
                # Add next sentences to the stack
                for k, x in enumerate(seg.sents):
                    if k != j and (k, other_seg) not in added and x == other_sent:
                        stack.append((k, other_seg, seg, equals_this))
                        added.add((k, other_seg))
        return pos, neg
