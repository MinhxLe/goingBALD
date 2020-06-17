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
import torch.nn as nn
from torch.utils.data import Dataset
from bald import data_dir,vectors_dir,load_ner_dataset
from bald.dataset import Conll

vectors = GloVe(cache=vectors_dir)
data_path = os.path.join(data_dir,"raw","CoNLL2003","eng.train")
ds = ConllData(data_path=data_path,vectors=vectors,emb_dim=300)

