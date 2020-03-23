from os.path import join
import numpy as np
from datasets.base import SimDataset, SimDatasetPartition, TextLongLabelPartition, ClassBalancedTextLongLabelPartition


label2id = {'derailing': 0, 'discredit': 1, 'dominance': 2, 'sexual_harassment': 3, 'stereotype': 4, '0': 5}
id2label = {0: 'derailing', 1: 'discredit', 2: 'dominance', 3: 'sexual_harassment', 4: 'stereotype', 5: '0'}


class AMIGenericLoader(SimDataset):

    def __init__(self, train: np.array, dev: np.array, test: np.array,
                 batch_size: int, balance_train: bool):
        self.train, self.dev, self.test = train, dev, test
        self.batch_size = batch_size
        self.balance_train = balance_train

    def training_partition(self) -> SimDatasetPartition:
        if self.balance_train:
            nclass = len(label2id.keys())
            return ClassBalancedTextLongLabelPartition(self.train, self.batch_size // nclass, nclass)
        else:
            return TextLongLabelPartition(self.train, self.batch_size, train=True)

    def dev_partition(self) -> SimDatasetPartition:
        return TextLongLabelPartition(self.dev, self.batch_size, train=False)

    def test_partition(self) -> SimDatasetPartition:
        return TextLongLabelPartition(self.test, self.batch_size, train=False)


class AMI:
    """
    Represents the AMI (Automatic Misoginy Identification) dataset.
    It should support different languages (spanish and italian are also available),
    but it's only used for english in this project.
    :param path: a path to the dataset where language dirs are located
      (e.g. structure assumed is .../AMI/<lang>/<split>/{text.toks & class.txt})
    :param lang: a language code: en, es, it (must be available in `path`)
      Defaults to 'en'
    """

    @staticmethod
    def load_sentences(path: str):
        """
        Load sentences from a directory where
        files `text.toks` and `class.txt` are available
        :param path: the path of the directory
        :return: a pair (sentences: list[string], labels: list[int])
        """
        with open(join(path, 'text.toks'), 'r') as text_file, open(join(path, 'class.txt'), 'r') as label_file:
            sents = [line.strip() for line in text_file.readlines()]
            labels = [label2id[line.strip()] for line in label_file.readlines()]
        return sents, labels

    @staticmethod
    def pad_sentences(input_tokens: list, pad_token) -> list:
        """
        Pad a list of tokenized sentences using `pad_value` as fill
        :param input_tokens: list[list[A]]
        :param pad_token: something of type A
        :return: list[list[A]] where lists of tokens have the same length
        """
        max_len = max([len(sent) for sent in input_tokens])
                # Complete the rest of `tokens` with `pad_value`
        return [tokens + [pad_token] * (max_len - len(tokens))
                # Only if the length is less than `max_len`, otherwise just `tokens`
                if len(tokens) < max_len else tokens
                # For every sentence in the input
                for tokens in input_tokens]

    @staticmethod
    def lstm_tokenize(sentences: list) -> list:
        """
        Tokenize sentences by splitting them on space characters
        :param sentences: a list of sentences (list[string])
        :return: a list of lists of words (list[list[string]])
        """
        # Attention! `[PAD]` tokens are being considered out of vocabulary words here
        return AMI.pad_sentences([sent.split(' ') for sent in sentences], 'oov')


    @staticmethod
    def bert_tokenize(tokenizer, sentences: list):
        """
        Tokenize, add `[CLS]` and `[SEP]` special characters and pad sentences
        :param tokenizer: a tokenizer object with an `encode` method (like in huggingface transformers)
        :param sentences: a list of sentences (list[string])
        :return: a pair of:
            - a list of tokenized sentences for BERT (list[list[long]])
            - a list of attention masks for the sentences (list[list[int]])
        """
        # Special tokens include '[CLS]' and '[SEP]'
        input_ids = [tokenizer.encode(sent, add_special_tokens=True) for sent in sentences]
        # value 0 == `[PAD]`
        input_ids = AMI.pad_sentences(input_ids, pad_token=0)
        # Attention mask is 0 for padding tokens and 1 otherwise
        attention_masks = [[int(token != 0) for token in sent]
                           for sent in input_ids]
        return input_ids, attention_masks

    def __init__(self, path: str, lang: str = 'en'):
        self.train_sents, self.train_labels = self.load_sentences(join(path, lang, 'train'))
        self.dev_sents, self.dev_labels = self.load_sentences(join(path, lang, 'dev'))
        self.test_sents, self.test_labels = self.load_sentences(join(path, lang, 'test'))

    def word2vec_loader(self, batch_size: int, balance_train: bool) -> SimDataset:
        """
        Split sentences on space for tokenization and return a loader instance
        for this type of usage -> ```for batch_sents, batch_labels in loader```
        :param batch_size: the batch size of the loader
        :param balance_train: whether to balance the training w.r.t classes
        :return: a SimDataset that loads batches for usage with word embeddings
        """
        x_train = self.lstm_tokenize(self.train_sents)
        x_dev = self.lstm_tokenize(self.dev_sents)
        x_test = self.lstm_tokenize(self.test_sents)
        return AMIGenericLoader(train=np.array(list(zip(x_train, self.train_labels))),
                                dev=np.array(list(zip(x_dev, self.dev_labels))),
                                test=np.array(list(zip(x_test, self.test_labels))),
                                batch_size=batch_size, balance_train=balance_train)

    def bert_loader(self, tokenizer, batch_size: int, balance_train: bool) -> SimDataset:
        """
        Tokenize sentences according to the given tokenizer and add special tokens
        for BERT: `[CLS]`, `[SEP]` and `[PAD]`. The returned loader will be used in
        this way -> ```for (batch_sents, batch_attention), batch_labels in loader```
        :param tokenizer: a tokenizer with an `encode` method
          (like in huggingface transformers)
        :param batch_size: the batch size of the loader
        :param balance_train: whether to balance the training w.r.t classes
        :return: a SimDataset that loads batches for usage with BERT
        """
        x_train, x_train_attn = self.bert_tokenize(tokenizer, self.train_sents)
        x_dev, x_dev_attn = self.bert_tokenize(tokenizer, self.dev_sents)
        x_test, x_test_attn = self.bert_tokenize(tokenizer, self.test_sents)
        return AMIGenericLoader(train=np.array(list(zip(zip(x_train, x_train_attn), self.train_labels))),
                                dev=np.array(list(zip(zip(x_dev, x_dev_attn), self.dev_labels))),
                                test=np.array(list(zip(zip(x_test, x_test_attn), self.test_labels))),
                                batch_size=batch_size, balance_train=balance_train)
