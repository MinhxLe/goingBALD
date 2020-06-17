import torch
from torch.utils.data import Dataset
from bald.load_ner_dataset import load_ner_dataset

class ConllData(Dataset):
    def __init__(self,data_path,vectors,emb_dim):
        self.data = load_ner_dataset(data_path)
        self.encoding = {
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
        self.max_seq_len = self.compute_max_seq_len()
        self.num_labels = len(self.encoding)
        self.vectors = vectors
        self.emb_dim = emb_dim

    def compute_max_seq_len(self):
        return max(len(d["tag"]) for d in self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        sample = self.data[i]
        x_seq = sample["text"]
        y_seq = sample["tag"]

        x_seq = [self.vectors[tok] for tok in x_seq]
        rest = self.max_seq_len - len(x_seq)
        assert rest >= 0
        x_seq.extend([torch.zeros(self.emb_dim) for _ in range(rest)])
        x_seq = torch.stack(x_seq)

        y_seq = [self.encoding[tok] for tok in y_seq]
        y_seq.extend([0 for _ in range(rest)])
        assert len(y_seq) == self.max_seq_len
        y_seq = torch.tensor(y_seq)

        return x_seq,y_seq
