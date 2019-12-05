from os.path import join
import numpy as np
from datasets.base import SimDataset, SimDatasetPartition, TextLongLabelPartition
import sts.utils as sts
from collections import Counter


def _read_sst2_data(path: str):
    with open(join(path, 'sents'), 'r') as file:
        data = [line.strip().split('\t') for line in file.readlines()]
        sents = [sent.split(' ') for sent, _ in data]
        labels = [int(label) for _, label in data]
    return sents, labels


class BinarySST(SimDataset):

    def __init__(self, path: str, batch_size: int, vocab_path: str, vector_path: str):
        self.batch_size = batch_size
        if vector_path is not None:
            self.vocab_vec, n_inv, n_oov = sts.vectorized_vocabulary(vocab_path, vector_path)
            self.vocab = list(self.vocab_vec.keys())
            print(f"Created vocabulary with {n_inv} INV and {n_oov} OOV ({int(100 * n_inv / (n_inv + n_oov))}% coverage)")
        else:
            self.vocab_vec = None
            self.vocab = [line.strip() for line in open(vocab_path, 'r')]
        x_train, y_train = _read_sst2_data(join(path, 'train'))
        x_dev, y_dev = _read_sst2_data(join(path, 'dev'))
        x_test, y_test = _read_sst2_data(join(path, 'test'))
        train_counter = Counter(y_train)
        dev_counter = Counter(y_dev)
        test_counter = Counter(y_test)
        print(f"Train Sentences: {len(x_train)}")
        print(f"Train class distribution: {train_counter}")
        print(f"Dev Sentences: {len(x_dev)}")
        print(f"Dev class distribution: {dev_counter}")
        print(f"Test Sentences: {len(x_test)}")
        print(f"Test class distribution: {test_counter}")
        self.train_data = np.array(list(zip(x_train, y_train)))
        self.dev_data = np.array(list(zip(x_dev, y_dev)))
        self.test_data = np.array(list(zip(x_test, y_test)))

    def training_partition(self) -> SimDatasetPartition:
        return TextLongLabelPartition(self.train_data, self.batch_size, train=True)

    def dev_partition(self) -> SimDatasetPartition:
        return TextLongLabelPartition(self.dev_data, self.batch_size, train=False)

    def test_partition(self) -> SimDatasetPartition:
        return TextLongLabelPartition(self.test_data, self.batch_size, train=False)
