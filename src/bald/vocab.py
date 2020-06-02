import json
from typing import Dict

class Vocab:

    def __init__(self, pre_made: Dict[str,int] = None):
        if pre_made:
            self.token_to_id = pre_made
            self.id_to_token = {pre_made[token] for token in pre_made}
        else:
            self.token_to_id = {}
            self.id_to_token = {}

        self.unk = "<UNK>"
        self.add_token(self.unk)
        self.bos = "<BOS>"
        self.add_token(self.bos)
        self.eos = "<EOS>"
        self.add_token(self.eos)

    def __len__(self):
        return len(self.token_to_id)

    def add_token(self,token: str) -> int:

        if token in self.token_to_id:
            index = self.token_to_id[token]
        else:
            index = len(self.token_to_id)
            self.token_to_id[token] = index
            self.id_to_token[index] = token

        return index

    def lookup_id(self,token: str) -> int:

        if token in self.token_to_id:
            return self.token_to_id[token]
        else:
            return self.token_to_id[self.unk]

    def lookup_token(self,j: int) -> str:
        if j in self.id_to_token.keys():
            return self.id_to_token[j]
        else:
            raise KeyError(f"{j} not a valid index.")

    def to_json(self,path: str):
        with open(path, 'w') as file:
            json.dump(self.token_to_id, file)

    @classmethod
    def from_json(cls,path: str):

        with open(path, 'r') as file:
            pre_made = json.load(file)

        return cls(pre_made)

