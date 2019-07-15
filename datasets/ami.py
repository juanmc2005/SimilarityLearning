from os.path import join
import numpy as np
from datasets.base import SimDataset, SimDatasetPartition, TextLongLabelPartition


label2id = {'derailing': 0, 'discredit': 1, 'dominance': 2, 'sexual_harassment': 3, 'stereotype': 4}


def _read_ami_data(path: str):
    with open(join(path, 'text.toks'), 'r') as text_file, open(join(path, 'class.txt'), 'r') as label_file:
        sents = [line.strip().split(' ') for line in text_file.readlines()]
        labels = [label2id[line.strip()] for line in label_file.readlines()]
    return np.array(list(zip(sents, labels)))


class AMI(SimDataset):

    def __init__(self, path: str, batch_size: int, lang: str = 'en'):
        self.batch_size = batch_size
        self.train_data = _read_ami_data(join(path, lang, 'train'))
        self.dev_data = _read_ami_data(join(path, lang, 'dev'))
        self.test_data = _read_ami_data(join(path, lang, 'test'))

    def training_partition(self) -> SimDatasetPartition:
        return TextLongLabelPartition(self.train_data, self.batch_size, train=True)

    def dev_partition(self) -> SimDatasetPartition:
        return TextLongLabelPartition(self.dev_data, self.batch_size, train=False)

    def test_partition(self) -> SimDatasetPartition:
        return TextLongLabelPartition(self.test_data, self.batch_size, train=False)
