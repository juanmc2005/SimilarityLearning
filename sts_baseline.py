from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy


# This function was an attempt to solve a vanishing gradient
# We can remove it if it doesn't prove useful
def set_forget_gate_bias(lstm: nn.LSTM, value: float):
    """
    Set forget bias of the given LSTM module to `value`.
    Pytorch guarantees the following fixed positions for LSTM biases:

    [input_gate | forget_gate | cell_gate | output_gate]
    0          n/4           n/2         3n/4          n
                ^             ^
                |_____________|
                 we want this!

    Reference: https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
    """
    for names in lstm._all_weights:
        for name in filter(lambda n: "bias" in n, names):
            bias = getattr(lstm, name)
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(value)


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
        if not self.training:
            if self.mode == 'baseline':
                return self._forward_pair_concat(sents)
            else:
                return self._forward_pair(sents)
        elif self.mode == 'baseline':
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
        return torch.stack(embeds, dim=0).squeeze()

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
        return torch.stack(embeds, dim=0).squeeze()

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
        return torch.stack(embeds1, dim=0).squeeze(), torch.stack(embeds2, dim=0).squeeze()
