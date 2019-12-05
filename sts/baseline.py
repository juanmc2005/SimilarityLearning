import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy
from sts.modes import STSForwardMode


class STSBaselineNet(nn.Module):

    def __init__(self, device: str, nfeat_word: int, nfeat_sent: int, nlayers: int,
                 word_list: list, vec_vocab: dict, mode: STSForwardMode):
        super(STSBaselineNet, self).__init__()
        self.device = device
        self.nfeat_word = nfeat_word
        self.nfeat_sent = nfeat_sent
        self.mode = mode
        if 'oov' not in word_list:
            word_list.append('oov')
        self.word2id = {word: index for index, word in enumerate(word_list)}
        # This loads the pretrained embeddings into the Embedding object which will be learned
        self.word_embedding = nn.Embedding(len(word_list), nfeat_word)
        pretrained_weight = numpy.zeros(shape=(len(word_list), nfeat_word))
        for i, word in enumerate(word_list):
            pretrained_weight[i] = np.array(vec_vocab[word]) if word in vec_vocab else np.zeros(nfeat_word)
        self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.lstm = nn.LSTM(input_size=nfeat_word, hidden_size=nfeat_sent // 2,
                            num_layers=nlayers, bidirectional=True)

    def _to_word_embeddings(self, sent):
        word_ids = [self.word2id[word] if word in self.word2id else self.word2id['oov'] for word in sent]
        indices = Variable(torch.LongTensor(word_ids)).to(self.device)
        embedded_sent = self.word_embedding(indices)
        return embedded_sent.view(-1, 1, self.nfeat_word)

    def encode(self, sent):
        x = self._to_word_embeddings(sent)
        out, _ = self.lstm(x)
        # Max pooling to get the embedding
        return torch.max(out, 0)[0]

    def forward(self, sents):
        return self.mode.forward(self.encode, sents, self.training)
