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
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from bald import data_dir,vectors_dir,load_ner_dataset
from bald.dataset import ConllDataset
from bald.simple_model import ConllModel

vectors = GloVe(cache=vectors_dir)

train_path = os.path.join(data_dir,"raw","CoNLL2003","eng.train")
train_ds = ConllDataset(data_path=train_path,vectors=vectors,emb_dim=300)

test_path = os.path.join(data_dir,"raw","CoNLL2003","eng.testa")
test_ds = ConllDataset(data_path=test_path,vectors=vectors,emb_dim=300)

max_seq_len = max(train_ds.max_seq_len,test_ds.max_seq_len)
train_ds.set_max_seq_len(max_seq_len)
test_ds.set_max_seq_len(max_seq_len)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

model = ConllModel(
    max_seq_len = max_seq_len,
    num_labels = train_ds.num_labels,
    emb_dim = train_ds.emb_dim
    )

num_epochs = 5
objective = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters())

epoch_train_losses = []
epoch_test_losses = []
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}.")

    model.train()
    batch_losses = []
    print("Training.")
    with tqdm.tqdm(total=len(train_dl)) as progress_bar:
        for datum in train_dl:

            optimizer.zero_grad()

            x_raw, y_raw = datum
            batch_len,seq_len = y_raw.size()
            y_target = y_raw.view(batch_len*seq_len)
            y_pred = model(x_raw)

            loss = objective(y_pred, y_target)
            batch_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            progress_bar.update(1)

    epoch_loss = sum(batch_losses)/len(batch_losses)
    epoch_train_losses.append(epoch_loss)

    model.eval()
    batch_losses = []
    print("Testing.")
    with tqdm.tqdm(total=len(test_dl)) as progress_bar:
        for datum in test_dl:

            x_raw, y_raw = datum
            batch_len,seq_len = y_raw.size()
            y_target = y_raw.view(batch_len*seq_len)
            y_pred = model(x_raw)

            loss = objective(y_pred, y_target)
            batch_losses.append(loss.item())

            progress_bar.update(1)

    epoch_loss = sum(batch_losses)/len(batch_losses)
    epoch_test_losses.append(epoch_loss)

plt.plot(epoch_train_losses, label="train")
plt.plot(epoch_test_losses, label="test")
plt.legend()
plt.show()


