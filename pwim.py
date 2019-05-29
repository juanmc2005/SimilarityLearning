from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy


# TODO further clean up and inherit from SimNet
class PWIM(nn.Module):
    def __init__(self, nfeat_word, nfeat_sent, nlayers, vec_vocab, tokens):
        super(PWIM, self).__init__()
        self.nfeat_word = nfeat_word
        self.word2id = {word: index for index, word in enumerate(tokens)}
        self.word_embedding = nn.Embedding(len(tokens), nfeat_word) # TODO is this actually used ?
        # This loads the pretrained embeddings into the Embedding object which will be learned
        self.copied_word_embedding = nn.Embedding(len(tokens), nfeat_word)
        pretrained_weight = numpy.zeros(shape=(len(tokens), nfeat_word))
        for i, word in enumerate(tokens):
            pretrained_weight[i] = vec_vocab[word].numpy()
        self.copied_word_embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.lstm = nn.LSTM(nfeat_word, nfeat_sent // 2, nlayers, bidirectional=True)

    def word_layer(self, sent):
        tmp = []
        for word in sent:
            if word in self.word2id:
                tmp.append(self.word2id[word])
            else:
                tmp.append(self.word2id['oov'])
        indices = Variable(torch.LongTensor(tmp))
        if torch.cuda.is_available():
            indices = indices.cuda()
        sentA = self.copied_word_embedding(indices)
        return sentA.view(-1, 1, self.nfeat_word)

    # TODO currently, this only supports a single sentence at a time, modify to accept batches
    def forward(self, input, y):
        input = self.word_layer(input)
        # encode input
        out, (state, _) = self.lstm(input)
        # max pooling to get embeddings
        embed = torch.max(out, 0)[0]
        # TODO utiliser l'embedding pour obtenir des logits (i.e. add loss module)
        return embed, None
