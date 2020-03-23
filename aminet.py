import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel
import common


class AMILSTM(nn.Module):

    def __init__(self, nfeat_word: int, nfeat_sent: int, word_list: list, vec_vocab, dropout: float = 0):
        super(AMILSTM, self).__init__()
        self.nfeat_word = nfeat_word
        self.nfeat_sent = nfeat_sent
        if 'oov' not in word_list:
            word_list.append('oov')
        self.word2id = {word: index for index, word in enumerate(word_list)}
        # This loads the pretrained embeddings into the Embedding object which will be learned
        self.word_embedding = nn.Embedding(len(word_list), nfeat_word).to(common.DEVICE)
        pretrained_weight = np.zeros(shape=(len(word_list), nfeat_word))
        for i, word in enumerate(word_list):
            pretrained_weight[i] = np.array(vec_vocab[word]) if word in vec_vocab else np.zeros(nfeat_word)
        self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.lstm = nn.LSTM(input_size=nfeat_word, hidden_size=nfeat_sent // 2,
                            num_layers=1, bidirectional=True,
                            dropout=dropout, batch_first=True).to(common.DEVICE)

    def forward(self, sents: list) -> torch.Tensor:
        """
        Apply the LSTM encoder over the sentence inputs and obtain
        corresponding sentence embeddings
        :param sents: list[list[string]], tokenized and padded sentences
        :return: word embeddings for current batch, a tensor of shape (batch_size, emb_dim)
        """
        # Convert words to word IDs
        word_ids = []
        for sent in sents:
            word_ids.append([self.word2id[word]
                             if word in self.word2id else self.word2id['oov']
                             for word in sent])
        word_ids = torch.LongTensor(word_ids).to(common.DEVICE)
        # Get word embeddings
        emb_sequences = self.word_embedding(word_ids)
        # Encode
        hidden, _ = self.lstm(emb_sequences)
        # Max pooling
        # hidden has shape (batch_size, seq_len, emb_dim)
        return torch.max(hidden, dim=1)[0]

    # def _to_word_embeddings(self, sent):
    #     word_ids = [self.word2id[word] if word in self.word2id else self.word2id['oov'] for word in sent]
    #     indices = Variable(torch.LongTensor(word_ids)).to(self.device)
    #     embedded_sent = self.word_embedding(indices)
    #     return embedded_sent.view(-1, 1, self.nfeat_word)
    #
    # def _embed(self, sent):
    #     x = self._to_word_embeddings(sent)
    #     out, _ = self.lstm(x)
    #     # Max pooling to get the embedding
    #     return torch.max(out, 0)[0]
    #
    # def forward(self, sents):
    #     embeddings = [self._embed(sent) for sent in sents]
    #     return torch.stack(embeddings, dim=0).squeeze()


class AMIBert(nn.Module):

    def __init__(self, pretrained_weights: str, freeze: bool = False):
        super(AMIBert, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_weights)
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, tokens_with_attention_mask: tuple) -> torch.Tensor:
        """
        Apply the BERT encoder over the sentence inputs and obtain
        the corresponding sentence embeddings
        :param tokens_with_attention_mask: a pair (token_ids, attn_mask) with
          - token_ids: list[list[long]], tokenized sentences already mapped to token IDs
          - attn_mask: list[list[int]], attention masks with 0 set for [PAD] tokens
        :return: the batch of sentence embeddings with shape (batch_size, embedding_dim)
        """
        token_ids = [token for token, _ in tokens_with_attention_mask]
        attn_mask = [mask for _, mask in tokens_with_attention_mask]
        # To tensors
        token_ids = torch.tensor(token_ids).to(common.DEVICE)
        attn_mask = torch.tensor(attn_mask).to(common.DEVICE)
        # Get hidden states for each word
        hidden = self.bert(token_ids, attention_mask=attn_mask)[0]
        # Max pooling
        # hidden has shape (batch_size, seq_len, emb_dim)
        return torch.max(hidden, dim=1)[0]
