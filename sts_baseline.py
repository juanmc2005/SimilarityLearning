from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy


class STSBaselineNet(nn.Module):

    def __init__(self, device, nfeat_word, nfeat_sent, vec_vocab, mode='baseline'):
        super(STSBaselineNet, self).__init__()
        self.device = device
        self.nfeat_word = nfeat_word
        self.nfeat_sent = nfeat_sent
        self.mode = mode
        tokens = vec_vocab.keys()
        if 'oov' not in tokens:
            tokens.append('oov')
        self.word2id = {word: index for index, word in enumerate(tokens)}
        # This loads the pretrained embeddings into the Embedding object which will be learned
        self.word_embedding = nn.Embedding(len(tokens), nfeat_word)
        pretrained_weight = numpy.zeros(shape=(len(tokens), nfeat_word))
        for i, word in enumerate(tokens):
            pretrained_weight[i] = vec_vocab[word].numpy()
        self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.word_embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=nfeat_word, hidden_size=nfeat_sent // 2,
                            num_layers=1, bidirectional=True)

    def _to_word_embeddings(self, sent):
        tmp = []
        for word in sent:
            if word in self.word2id:
                tmp.append(self.word2id[word])
            else:
                tmp.append(self.word2id['oov'])
        indices = Variable(torch.LongTensor(tmp)).to(self.device)
        embedded_sent = self.word_embedding(indices)
        return embedded_sent.view(-1, 1, self.nfeat_word)

    def _embed(self, sent):
        x = self._to_word_embeddings(sent)
        out, _ = self.lstm(x)
        # Max pooling to get the embedding
        embedding = torch.max(out, 0)[0]
        return embedding

    def forward(self, sents):
        # FIXME Replace this with subclasses or (preferably) delegation
        if self.mode == 'baseline':
            return self._forward_pair_concat(sents)
        elif not self.training:
            return self._forward_pair(sents)
        elif self.mode == 'pairs':
            return self._forward_pair(sents)
        else:
            return self._forward_single(sents)

    def _forward_single(self, sents):
        embs = [self._embed(sent) for sent in sents]
        return torch.stack(embs, dim=0).squeeze()

    def _forward_pair_concat(self, sents):
        embs = [torch.cat((self._embed(s1), self._embed(s2)), 1) for s1, s2 in sents]
        return torch.stack(embs, dim=0).squeeze()

    def _forward_pair(self, sents):
        embs1, embs2 = [], []
        for s1, s2 in sents:
            embs1.append(self._embed(s1))
            embs2.append(self._embed(s2))
        return torch.stack(embs1, dim=0).squeeze(), torch.stack(embs2, dim=0).squeeze()
