"""
Index module uses to map string entities (words, characters) to index
The index can be used with an embedding
"""
import string
import pickle
from bald.data.constants import PAD_TOKEN, UNKNOWN_TOKEN


class Indexer:
    def __init__(self):
        self.key2idx = {}
        self.idx2key = []

    def add(self, key):
        if key not in self.key2idx:
            self.key2idx[key] = len(self.idx2key)
            self.idx2key.append(key)
        return self.key2idx[key]

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.key2idx[key]
        if isinstance(key, int):
            return self.idx2key[key]

    def __len__(self):
        return len(self.idx2key)

    def save(self, f):
        with open(f, 'wt', encoding='utf-8') as fout:
            for index, key in enumerate(self.idx2key):
                fout.write(key + '\t' + str(index) + '\n')

    def load(self, f):
        with open(f, 'rt', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                key = line.split()[0]
                self.add(key)


class Vocabulary(Indexer):
    def __init__(self):
        super().__init__()
        self.add(PAD_TOKEN)
        self.add(UNKNOWN_TOKEN)

    def __getitem__(self, key):
        if isinstance(key, str) and key not in self.key2idx:
            return self.key2idx[UNKNOWN_TOKEN]
        return super().__getitem__(key)


class Charset(Indexer):

    def __init__(self):
        super().__init__()
        for char in string.printable[0:-6]:
            self.add(char)
        self.add(PAD_TOKEN)
        self.add(UNKNOWN_TOKEN)

    @staticmethod
    def type(char):
        if char in string.digits:
            return "Digits"
        if char in string.ascii_lowercase:
            return "Lower Case"
        if char in string.ascii_uppercase:
            return "Upper Case"
        if char in string.punctuation:
            return "Punctuation"
        return "Other"

    def __getitem__(self, key):
        if isinstance(key, str) and key not in self.key2idx:
            return self.key2idx[UNKNOWN_TOKEN]
        return super().__getitem__(key)
