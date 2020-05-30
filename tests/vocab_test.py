from pathlib import Path
from bald.vocab import Vocab

def test_vocab():
    vocab = Vocab()

    token_to_id = vocab.token_to_id
    id_to_token = vocab.id_to_token
    assert len(token_to_id) == len(id_to_token)
    assert len(token_to_id) == 3

    i = vocab.add_token("godzilla")
    assert token_to_id["godzilla"] == i
    assert vocab.lookup_token(i) == "godzilla"
    assert id_to_token[i] == "godzilla"
    assert vocab.lookup_id("godzilla") == i

    ii = vocab.add_token("godzilla")
    assert ii == i

    for token in token_to_id:
        assert token == id_to_token[token_to_id[token]]

    for j in id_to_token:
        assert j == token_to_id[id_to_token[j]]



def test_json():
    vocab = Vocab()
    vocab.add_token("godzilla")
    vocab.add_token("spiderman")

    path = Path(__file__).resolve().parent
    path = path / Path("minimal.json")

    vocab.to_json(path)
    assert 1 == 1

    bacov = Vocab.from_json(path)

    for token in vocab.token_to_id:
        i = vocab.token_to_id[token]
        j = bacov.token_to_id[token]
        assert i == j

    for token in bacov.token_to_id:
        i = vocab.token_to_id[token]
        j = bacov.token_to_id[token]
        assert i == j
