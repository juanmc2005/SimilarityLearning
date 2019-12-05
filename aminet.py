import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


# TODO shares code with STSBaselineNet --> REFACTORING!
class AMINet(nn.Module):

    def __init__(self, device: str, nfeat_word: int, nfeat_sent: int, word_list: list, vec_vocab, dropout: float = 0):
        super(AMINet, self).__init__()
        self.device = device
        self.nfeat_word = nfeat_word
        self.nfeat_sent = nfeat_sent
        if 'oov' not in word_list:
            word_list.append('oov')
        self.word2id = {word: index for index, word in enumerate(word_list)}
        # This loads the pretrained embeddings into the Embedding object which will be learned
        self.word_embedding = nn.Embedding(len(word_list), nfeat_word)
        pretrained_weight = np.zeros(shape=(len(word_list), nfeat_word))
        for i, word in enumerate(word_list):
            pretrained_weight[i] = np.array(vec_vocab[word]) if word in vec_vocab else np.zeros(nfeat_word)
        self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.lstm = nn.LSTM(input_size=nfeat_word, hidden_size=nfeat_sent // 2,
                            num_layers=1, bidirectional=True, dropout=dropout)

    def _to_word_embeddings(self, sent):
        word_ids = [self.word2id[word] if word in self.word2id else self.word2id['oov'] for word in sent]
        indices = Variable(torch.LongTensor(word_ids)).to(self.device)
        embedded_sent = self.word_embedding(indices)
        return embedded_sent.view(-1, 1, self.nfeat_word)

    def _embed(self, sent):
        x = self._to_word_embeddings(sent)
        out, _ = self.lstm(x)
        # Max pooling to get the embedding
        return torch.max(out, 0)[0]

    def forward(self, sents):
        embeddings = [self._embed(sent) for sent in sents]
        return torch.stack(embeddings, dim=0).squeeze()
