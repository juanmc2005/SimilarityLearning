import torch


class STSForwardMode:

    @staticmethod
    def stack(embeddings: list):
        return torch.stack(embeddings, dim=0).squeeze()

    def forward(self, embed_fn, sents: list, train: bool):
        raise NotImplementedError


class ConcatSTSForwardMode(STSForwardMode):

    def forward(self, embed_fn, sents: list, train: bool):
        embs = [torch.cat((embed_fn(s1), embed_fn(s2)), 1) for s1, s2 in sents]
        return self.stack(embs)


class PairSTSForwardMode(STSForwardMode):

    def forward(self, embed_fn, sents: list, train: bool):
        embs1, embs2 = [], []
        for s1, s2 in sents:
            embs1.append(embed_fn(s1))
            embs2.append(embed_fn(s2))
        return self.stack(embs1), self.stack(embs2)


class TripletSTSForwardMode(STSForwardMode):

    def __init__(self):
        self.eval_mode = PairSTSForwardMode()

    def forward(self, embed_fn, sents: list, train: bool):
        if train:
            embs_anchor, embs_pos, embs_neg = [], [], []
            for anchor, pos, neg in sents:
                embs_anchor.append(embed_fn(anchor))
                embs_pos.append(embed_fn(pos))
                embs_neg.append(embed_fn(neg))
            return self.stack(embs_anchor), self.stack(embs_pos), self.stack(embs_neg)
        else:
            return self.eval_mode.forward(embed_fn, sents, train)


class SingleSTSForwardMode(STSForwardMode):

    def __init__(self):
        self.eval_mode = PairSTSForwardMode()

    def forward(self, embed_fn, sents: list, train: bool):
        if train:
            embs = [embed_fn(sent) for sent in sents]
            return self.stack(embs)
        else:
            return self.eval_mode.forward(embed_fn, sents, train)


class STSForwardModeFactory:

    def new(self, loss: str) -> STSForwardMode:
        if loss == 'kldiv':
            # Kullback-Leibler Divergence loss, forward pairs and then concatenate embeddings
            return ConcatSTSForwardMode()
        elif loss == 'contrastive':
            # Contrastive loss, forward pairs and don't concatenate
            return PairSTSForwardMode()
        elif loss == 'triplet':
            # Triplet loss, forward triplets and don't concatenate
            return TripletSTSForwardMode()
        else:
            # Softmax based, forward a single sentence at a time
            return SingleSTSForwardMode()
