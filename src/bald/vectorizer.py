from torch import Tensor
from typing import List
from bald.vocab import Vocab

class Vectorizer:

    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def pre_vectorize(self,sequence: List[str]) -> List[int]:
        out = [self.vocab.lookup_id(self.vocab.bos)]
        out.extend([self.vocab.lookup_id(token) for token in sequence])
        out.append(self.vocab.lookup_id(self.vocab.eos))
        return out

    def vectorize(self,sequence: List[str]) -> Tensor:
        pre_vectorized = self.pre_vectorize(sequence)
        return Tensor(pre_vectorized)
