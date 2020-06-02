from torch.utils.data import Dataset

from bald.load_ner_dataset import load_ner_dataset

class DatasetConllStyle(Dataset):

    def __init__(self,sents,vectorizer,label_vectorizer):
        self.sentences = sents
        self.vectorizer = vectorizer
        self.label_vectorizer = label_vectorizer

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        
        x = sentence["text"]
        y = sentence["tag"]
        
        x_vect = self.vectorizer(x)
        y_vect = self.label_vectorizer(y)
        return x_vect,y_vect

    @classmethod
    def from_path(
            cls,
            vectorizer,
            label_vectorizer,
            path = "../../datasets/CoNLL2003/eng.testa",
        ):
        sents = load_ner_dataset(path)
        return cls(sents)