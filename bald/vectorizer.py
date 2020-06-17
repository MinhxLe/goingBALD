from typing import List

import torch
from torch import Tensor
# import torchnlp.word_to_vector.pretrained_word_vectors._PretrainedWordVectors as BaseWordVectorizer

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
        pre_vectorized = [Tensor(token) for token in pre_vectorized]
        return torch.stack(pre_vectorized)


class LabelVectorizer:
    '''
    Class to vectorize the NER tags
    '''
    encoding = {
        'O':0,
        'B-PER':1,
        'I-PER':1,
        'B-ORG':2,
        'I-ORG':2,
        'B-LOC':3,
        'I-LOC':3,
        'B-MISC':4,
        'I-MISC':4,
    }

    @classmethod
    def pre_vectorize(cls,sequence: List[str]) -> List[int]:
        out = [0]
        middle = [cls.encoding[tag] for tag in sequence]
        out.extend(middle)
        out.append(0)
        return out

    @classmethod
    def vectorize(cls,sequence: List[str]) -> Tensor:
        pre_vectorized = cls.pre_vectorize(sequence)
        pre_vectorized = [Tensor(token) for token in pre_vectorized]
        return torch.stack(pre_vectorized)

class WordVectorizer:

    def __init__(self, vectorizer: BaseWordVectorizer):
        self.vectorizer = vectorizer

    def vectorize(self, sequence: List[str]) -> Tensor:
        return self.vectorizer(text)
