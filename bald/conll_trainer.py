"""
Step 1: train a simple model.
Step 2: train a simple model using active learning.
Step 3: understand something.
Step 4: feel productive.
Step 5: ...
Step 6: profit.
"""
import os
import matplotlib.pyplot as plt
from torchnlp.word_to_vector import GloVe
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from loguru import logger
from datetime import datetime

from bald import data_dir,vectors_dir,load_ner_dataset, results_dir
from bald.dataset import ConllDataset
from bald.simple_model import ConllModel
from bald.utils import epoch_run

root_name = str(results_dir / "logs" / f"conll_trainer_{datetime.now()}")
logger.add(root_name+"_.log")
logger.info("Using GloVe 300 embeddings, CoNLL2003 dataset.")

vectors = GloVe(cache=vectors_dir)

train_path = os.path.join(data_dir,"raw","CoNLL2003","eng.train")
train_ds = ConllDataset(data_path=train_path,vectors=vectors,emb_dim=300)

test_path = os.path.join(data_dir,"raw","CoNLL2003","eng.testa")
test_ds = ConllDataset(data_path=test_path,vectors=vectors,emb_dim=300)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

# global stuff
cnns = 10
out_channels = 100
logger.info(f"{cnns} Convolutional layers with {out_channels} filters.")

num_epochs = 10
logger.info(f"Will (try to) train for {num_epochs} epochs.")

# here we go
model = ConllModel(
    num_labels = train_ds.num_labels,
    emb_dim = train_ds.emb_dim,
    out_channels = out_channels,
    cnns = cnns,
    )

def loss_fun(input,target):
    batch_len,seq_len = target.size()
    target = target.view(batch_len*seq_len)
    return F.cross_entropy(input=input,target=target,ignore_index=0)

def score_fun(input,target):
    dims,labels = input.size()
    y_pred = F.softmax(input,dim=1)
    y_pred = torch.argmax(y_pred,dim=1)
    target = target.view(dims)
    return f1_score(
            y_true = target.cpu().data.numpy(),
            y_pred = y_pred.cpu().data.numpy(),
            labels = list(range(1,6)),
            average = "weighted",
        )

optimizer = torch.optim.Adam(model.parameters())


train_losses = []
test_losses = []

for epoch in range(num_epochs):
    logger.info(f"Epoch {epoch+1}")

    print("Training.")
    run_d = epoch_run(
        model = model,
        data_loader = train_dl,
        criterion = loss_fun,
        score_fun = score_fun,
        trainer_mode = True,
        optimizer = optimizer,
        )

    logger.info(f"Train loss is {run_d['loss']}.")
    logger.info(f"Train f1 score is {run_d['score']}.")
    train_losses.append(run_d["loss"])

    print("Evaluating.")
    run_d = epoch_run(
        model = model,
        data_loader = train_dl,
        criterion = loss_fun,
        score_fun = score_fun,
        trainer_mode = False,
        )
    logger.info(f"Eval loss is {run_d['loss']}.")
    logger.info(f"Eval f1 score is {run_d['score']}.")
    test_losses.append(run_d["loss"])

plt.plot(train_losses, label="train")
plt.plot(test_losses, label="test")
plt.legend()
plt.savefig(fname=root_name+"_.png")
