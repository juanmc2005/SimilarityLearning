from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy


class STSBaselineNet(nn.Module):
    def __init__(self, device, nfeat_word, nfeat_sent, vec_vocab, tokens, mode='baseline'):
        super(STSBaselineNet, self).__init__()
        self.device = device
        self.nfeat_word = nfeat_word
        self.nfeat_sent = nfeat_sent
        self.mode = mode
        if 'oov' not in tokens:
            tokens.append('oov')
        self.word2id = {word: index for index, word in enumerate(tokens)}
        # This loads the pretrained embeddings into the Embedding object which will be learned
        self.word_embedding = nn.Embedding(len(tokens), nfeat_word)
        pretrained_weight = numpy.zeros(shape=(len(tokens), nfeat_word))
        for i, word in enumerate(tokens):
            pretrained_weight[i] = vec_vocab[word].numpy()
        self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.lstm = nn.LSTM(input_size=nfeat_word, hidden_size=nfeat_sent // 2,
                            num_layers=1, bidirectional=True)

    def word_layer(self, sent):
        tmp = []
        for word in sent:
            if word in self.word2id:
                tmp.append(self.word2id[word])
            else:
                tmp.append(self.word2id['oov'])
        indices = Variable(torch.LongTensor(tmp)).to(self.device)
        embedded_sent = self.word_embedding(indices)
        return embedded_sent.view(-1, 1, self.nfeat_word)

    def forward(self, sents):
        # FIXME Replace this with subclasses or (preferably) delegation
        if self.mode == 'baseline':
            return self._forward_pair_concat(sents)
        elif self.mode == 'pairs':
            return self._forward_pair(sents)
        else:
            return self._forward_single(sents)

    def _forward_single(self, sents):
        embeds = []
        for sent in sents:
            x = self.word_layer(sent)
            # Encode input
            out, _ = self.lstm(x)
            # Max pooling to get embeddings
            embed = torch.max(out, 0)[0]
            embeds.append(embed)
        return torch.cat(embeds, 0).view(-1, self.nfeat_sent)

    def _forward_pair_concat(self, sents):
        embeds = []
        for s1, s2 in sents:
            x1 = self.word_layer(s1)
            x2 = self.word_layer(s2)
            # Encode input
            out1, _ = self.lstm(x1)
            out2, _ = self.lstm(x2)
            # Max pooling to get embeddings
            embed1 = torch.max(out1, 0)[0]
            embed2 = torch.max(out2, 0)[0]
            embed = torch.cat((embed1, embed2), 1)
            embeds.append(embed)
        return torch.cat(embeds, 0).view(-1, 2*self.nfeat_sent)

    def _forward_pair(self, sents):
        embeds1, embeds2 = [], []
        for s1, s2 in sents:
            x1 = self.word_layer(s1)
            x2 = self.word_layer(s2)
            # Encode input
            out1, _ = self.lstm(x1)
            out2, _ = self.lstm(x2)
            # Max pooling to get embeddings
            embed1 = torch.max(out1, 0)[0]
            embed2 = torch.max(out2, 0)[0]
            embeds1.append(embed1)
            embeds2.append(embed2)
        return torch.cat(embeds1, 0).view(-1, self.nfeat_sent), torch.cat(embeds2, 0).view(-1, self.nfeat_sent)
