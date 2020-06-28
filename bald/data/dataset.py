from torch.utils.data import Dataset
from typing import List, Dict

class CoNLLNERDataset(Dataset):

    def __init__(self, sentence_data):
        self.sentences = sentence_data

    def __getitem__(self, key):
        return self.sentences[key]

    def __len__(self):
        return len(self.sentences)
