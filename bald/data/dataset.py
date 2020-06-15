from torch.utils.data import Dataset
from bald.data.conll2003_utils import load_raw_dataset
from typing import List, Dict

class CoNLLNERDataset(Dataset):

    def __init__(self, fname):
        self.sentences = load_raw_dataset(fname)

    def __getitem__(self, key):
        return self.sentences[key]

    def __len__(self):
        return len(self.sentences)
