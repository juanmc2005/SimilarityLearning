import os
from os.path import join
import csv
import argparse
import numpy as np
from nltk.tokenize.casual import TweetTokenizer
from sklearn.model_selection import train_test_split


def read_tsv_data(path: str):
    with open(path, 'r') as tsv_file:
        reader = csv.DictReader(tsv_file, dialect='excel-tab')
        sents, labels = [], []
        for row in reader:
            sents.append(row['text'].lower())
            labels.append(row['misogyny_category'])
    return sents, labels


def create_dev_like(x_train: list, y_train: list, count_dict: dict):
    perm = np.random.permutation(len(y_train))
    x_dev, y_dev = [], []
    new_x_train, new_y_train = [], []
    for i in perm:
        if count_dict[y_train[i]] > 0:
            count_dict[y_train[i]] -= 1
            x_dev.append(x_train[i])
            y_dev.append(y_train[i])
        else:
            new_x_train.append(x_train[i])
            new_y_train.append(y_train[i])
    return new_x_train, x_dev, new_y_train, y_dev


def tokenize(sents: list):
    tokenizer = TweetTokenizer(preserve_case=True, strip_handles=True)
    sents_tok = []
    for sent in sents:
        tokens = [token for token in tokenizer.tokenize(sent) if not token.startswith('http')]
        sents_tok.append(' '.join(tokens))
    return sents_tok


def create_partition(path: str, sentences: list, tokenized_sentences: list, labels: list):
    os.mkdir(path)
    with open(join(path, 'text.toks'), 'w') as tokfile, \
            open(join(path, 'text.txt'), 'w') as textfile, \
            open(join(path, 'class.txt'), 'w') as classfile:
        for sent, toks, label in zip(sentences, tokenized_sentences, labels):
            textfile.write(sent + '\n')
            tokfile.write(toks + '\n')
            classfile.write(label + '\n')


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='Path to AMI dataset')
parser.add_argument('--out', type=str, required=True, help='Path to output directory')
args = parser.parse_args()

train_sents, train_labels = read_tsv_data(join(args.path, 'en_training.tsv'))
x_train, x_dev, y_train, y_dev = train_test_split(train_sents, train_labels,
                                                  test_size=0.2, stratify=train_labels, random_state=124)
x_test, y_test = read_tsv_data(join(args.path, 'en_testing.tsv'))

assert len(x_train) == len(y_train)
assert len(x_dev) == len(y_dev)
assert len(x_test) == len(y_test)

x_train_tok = tokenize(x_train)
x_dev_tok = tokenize(x_dev)
x_test_tok = tokenize(x_test)

assert len(x_train_tok) == len(y_train)
assert len(x_dev_tok) == len(y_dev)
assert len(x_test_tok) == len(y_test)

create_partition(join(args.out, 'train'), x_train, x_train_tok, y_train)
create_partition(join(args.out, 'dev'), x_dev, x_dev_tok, y_dev)
create_partition(join(args.out, 'test'), x_test, x_test_tok, y_test)

# No TEST set when creating vocabulary !
vocab = set()
for partition in [x_train_tok, x_dev_tok]:
    for sent in partition:
        for token in sent.split():
            vocab.add(token)

print(f"Vocabulary size: {len(vocab)}")

with open(join(args.out, 'ami_vocab.txt'), 'w') as vocabfile:
    for token in vocab:
        vocabfile.write(token + '\n')

print('Done')


