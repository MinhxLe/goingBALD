"""
Step 1: train a simple model.
Step 2: train a simple model using active learning.
Step 3: understand something.
Step 4: feel productive.
Step 5: ...
Step 6: profit.
"""
import os
from torchnlp.word_to_vector import GloVe
import torch
from torch import nn
from torch.utils.data import Dataloader
from bald import data_dir,vectors_dir,load_ner_dataset
from bald.dataset import Conll

vectors = GloVe(cache=vectors_dir)
data_path = os.path.join(data_dir,"raw","CoNLL2003","eng.train")
ds = ConllDataset(data_path=data_path,vectors=vectors,emb_dim=300)
dl = Dataloader(ds, batch_size=32, shuffle=True)
model = ConllModel(ds.max_seq_len,ds.num_labels,ds.emb_dim)

num_epochs = 2
objective = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for datum in dl:
        x_raw, y_raw = datum

        


