import json
from typing import Dict

class Vocab:

    def __init__(self, pre_made: Dict[str,int] = None):
        if pre_made:
            self.token_to_i = pre_made
            self.i_to_token = {pre_made[token] for token in pre_made}
        else:
            self.token_to_i = {}
            self.i_to_token = {}

        self.unk = "<UNK>"
        self.add_token(self.unk)
        self.bos = "<BOS>"
        self.add_token(self.bos)
        self.eos = "<EOS>"
        self.add_token(self.eos)

    def add_token(self,token: str) -> int:
        i = self.token_to_i.setdefault(
                token,
                len(self.token_to_i)
            )
        return i

    def lookup_i(self,token: str) -> int:
        if token in self.token_to_i:
            return self.token_to_i[token]
        else:
            return self.token_to_i[self.unk]

    def lookup_token(self,j: int) -> str:
        if j in self.i_to_token.keys():
            return self.i_to_token[j]
        else:
            raise KeyError(f"{j} not a valid index.")

    @classmethod
    def from_json(cls,path: str):
        pre_made = json.loads(path)
        return cls(pre_made)
