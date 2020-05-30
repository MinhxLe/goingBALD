from bald.vectorizer import Vectorizer
from bald.vocab import Vocab

def test_vectorizer():
    vocab = Vocab()
    vocab.add_token("a")
    vocab.add_token("b")

    ve = Vectorizer(vocab)

    seq = ["a","c","b"]
    assert ve.pre_vectorize(seq) == [1,3,0,4,2]
