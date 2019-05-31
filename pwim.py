from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy


class PWIM(nn.Module):
    def __init__(self, device, nfeat_word, nfeat_sent, vec_vocab, tokens):
        super(PWIM, self).__init__()
        self.device = device
        self.nfeat_word = nfeat_word
        self.nfeat_sent = nfeat_sent
        if not 'oov' in tokens:
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
        sentA = self.word_embedding(indices)
        return sentA.view(-1, 1, self.nfeat_word)

    def forward(self, sents):
        embeds = []
        for sent in sents:
            x = self.word_layer(sent)
            # Encode input
            out, _ = self.lstm(x)
            # Max pooling to get embeddings
            embed = torch.max(out, 0)[0]
            embeds.append(embed)
        return torch.cat(embeds, 0).view(-1, self.nfeat_sent)
