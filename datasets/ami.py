from os.path import join
import numpy as np
from datasets.base import SimDataset, SimDatasetPartition, TextLongLabelPartition, ClassBalancedTextLongLabelPartition
import sts.utils as sts
from collections import Counter


label2id = {'derailing': 0, 'discredit': 1, 'dominance': 2, 'sexual_harassment': 3, 'stereotype': 4, '0': 5}
id2label = {0: 'derailing', 1: 'discredit', 2: 'dominance', 3: 'sexual_harassment', 4: 'stereotype', 5: '0'}


def _read_ami_data(path: str):
    with open(join(path, 'text.toks'), 'r') as text_file, open(join(path, 'class.txt'), 'r') as label_file:
        sents = [line.strip().split(' ') for line in text_file.readlines()]
        labels = [label2id[line.strip()] for line in label_file.readlines()]
    return sents, labels


class AMI(SimDataset):

    def __init__(self, path: str, batch_size: int, vocab_path: str, vector_path: str,
                 lang: str = 'en', balance_train: bool = False):
        self.batch_size = batch_size
        self.balance_train = balance_train
        if vector_path is not None:
            self.vocab_vec, n_inv, n_oov = sts.vectorized_vocabulary(vocab_path, vector_path)
            self.vocab = list(self.vocab_vec.keys())
            print(f"Created vocabulary with {n_inv} INV and {n_oov} OOV ({int(100 * n_inv / (n_inv + n_oov))}% coverage)")
        else:
            self.vocab_vec = None
            self.vocab = [line.strip() for line in open(vocab_path, 'r')]
        x_train, y_train = _read_ami_data(join(path, lang, 'train'))
        x_dev, y_dev = _read_ami_data(join(path, lang, 'dev'))
        x_test, y_test = _read_ami_data(join(path, lang, 'test'))
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
        if self.balance_train:
            nclass = len(label2id.keys())
            return ClassBalancedTextLongLabelPartition(self.train_data, self.batch_size // nclass, nclass)
        else:
            return TextLongLabelPartition(self.train_data, self.batch_size, train=True)

    def dev_partition(self) -> SimDatasetPartition:
        return TextLongLabelPartition(self.dev_data, self.batch_size, train=False)

    def test_partition(self) -> SimDatasetPartition:
        return TextLongLabelPartition(self.test_data, self.batch_size, train=False)
