from os.path import join
import numpy as np
from datasets.base import SimDataset, SimDatasetPartition, TextLongLabelPartition
import sts.utils as sts


label2id = {'derailing': 0, 'discredit': 1, 'dominance': 2, 'sexual_harassment': 3, 'stereotype': 4}
id2label = {0: 'derailing', 1: 'discredit', 2: 'dominance', 3: 'sexual_harassment', 4: 'stereotype'}


def _read_ami_data(path: str):
    with open(join(path, 'text.toks'), 'r') as text_file, open(join(path, 'class.txt'), 'r') as label_file:
        sents = [line.strip().split(' ') for line in text_file.readlines()]
        labels = [label2id[line.strip()] for line in label_file.readlines()]
    return np.array(list(zip(sents, labels)))


class AMI(SimDataset):

    def __init__(self, path: str, batch_size: int, vocab_path: str, vector_path: str, lang: str = 'en'):
        self.batch_size = batch_size
        self.vocab, n_inv, n_oov = sts.vectorized_vocabulary(vocab_path, vector_path)
        print(f"Created vocabulary with {n_inv} INV and {n_oov} OOV ({int(100 * n_inv / (n_inv + n_oov))}% coverage)")
        self.train_data = _read_ami_data(join(path, lang, 'train'))
        self.dev_data = _read_ami_data(join(path, lang, 'dev'))
        self.test_data = _read_ami_data(join(path, lang, 'test'))

    def training_partition(self) -> SimDatasetPartition:
        return TextLongLabelPartition(self.train_data, self.batch_size, train=True)

    def dev_partition(self) -> SimDatasetPartition:
        return TextLongLabelPartition(self.dev_data, self.batch_size, train=False)

    def test_partition(self) -> SimDatasetPartition:
        return TextLongLabelPartition(self.test_data, self.batch_size, train=False)
