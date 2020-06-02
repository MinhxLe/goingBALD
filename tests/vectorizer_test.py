from bald.vectorizer import Vectorizer, LabelVectorizer
from bald.vocab import Vocab

def test_char_vectorizer():
    vocab = Vocab()
    vocab.add_token("a")
    vocab.add_token("b")

    ve = Vectorizer(vocab)

    seq = ["a","c","b"]
    assert ve.pre_vectorize(seq) == [1,3,0,4,2]

def test_label_vectorizer():
    sequence = ["O", "I-PER", "B-ORG", "I-ORG"]
    seq_v = LabelVectorizer.pre_vectorize(sequence)
    assert seq_v == [0,0,1,2,2,0]