import argparse
import logging
import numpy as np
import pandas as pd
import os
import sys
import time
import torch
import json
import torch.nn as nn
import torch.optim as optim

from bald.data.conll2003_utils import load_raw_dataset
from bald.data.constants import (
        PAD_TOKEN,
)
from bald.data.dataset import CoNLLNERDataset
from bald.data.indexer import (
        Charset, Indexer, Vocabulary,
)
from bald.data.samplers import (
        ALRandomSampler,
)
from bald.log_utils import time_display
from bald.model.model import Model
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (
        SequentialSampler,
        BatchSampler,
        RandomSampler,
)
from typing import List
from datetime import datetime
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
# model args
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (default: 0.5)')
parser.add_argument('--emb_dropout', type=float, default=0.25,
                    help='dropout applied to the embedded layer (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.35,
                    help='gradient clip, -1 means no clip (default: 0.35)')
parser.add_argument('--char_kernel_size', type=int, default=3,
                    help='character-level kernel size (default: 3)')
parser.add_argument('--word_kernel_size', type=int, default=3,
                    help='word-level kernel size (default: 3)')
parser.add_argument('--emsize', type=int, default=50,
                    help='size of character embeddings (default: 50)')
parser.add_argument('--char_layers', type=int, default=3,
                    help='# of character-level convolution layers (default: 3)')
parser.add_argument('--word_layers', type=int, default=3,
                    help='# of word-level convolution layers (default: 3)')
parser.add_argument('--char_nhid', type=int, default=50,
                    help='number of hidden units per character-level convolution layer (default: 50)')
parser.add_argument('--word_nhid', type=int, default=300,
                    help='number of hidden units per word-level convolution layer (default: 300)')

# training args
parser.add_argument('--train_epochs', type=int, default=3,
                    help='upper training epoch limit (default: 3)')
parser.add_argument('--lr', type=float, default=4,
                    help='initial learning rate (default: 4)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer type (default: SGD)')
parser.add_argument('--weight', type=float, default=10,
                    help='manual rescaling weight given to each tag except "O"')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
# AL args
parser.add_argument('--al_epochs', type=int, default=20,
                    help='# of active learning steps (default: 10)')

# experiment logging/debugging
parser.add_argument('--experiment_name', type=str, default='conll2003_random_sampler',
                    help='experiment name')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='report interval (default: 10)')
parser.add_argument('--debug', type=bool, default=False,
                    help='is debug runs on smaller dataset (defaults to False)')
args = parser.parse_args()

if args.debug:
    args.experiment_name += "_DEBUG"
    args.train_epochs = 1
    args.batch_size = 25
    args.al_epochs = 2


# insert random string to make experiment name unique in between runs
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
EXPERIMENT_DIR = os.path.join("experiments", f"{args.experiment_name}_{timestamp}")
if args.debug:
    EXPERIMENT_DIR = os.path.join("experiments", args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    os.makedirs(EXPERIMENT_DIR)

# dumping experiment args
with open(os.path.join(EXPERIMENT_DIR, "experiment_args.txt"), 'w') as f:
    json.dump(args.__dict__, f, indent=2)


# setting up logger
logger = logging.getLogger("train_logger")
fh = logging.FileHandler(os.path.join(EXPERIMENT_DIR, "experiment.log"))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)
logger.setLevel(logging.DEBUG)


# loading in data
DATA_PROCESSED_DIR = "artifacts/data/processed/CoNLL2003/"

charset = Charset()
vocab_set = Vocabulary()
vocab_set.load(os.path.join(DATA_PROCESSED_DIR, "word2vec_vocab_idx.txt"))
tag_set = Indexer()
tag_set.load(os.path.join(DATA_PROCESSED_DIR, "tags_idx.txt"))


# setting up model
word_embeddings = torch.Tensor(
        np.load(os.path.join(DATA_PROCESSED_DIR, "word2vec.vector.npy")))
word_embedding_size = word_embeddings.size(1)
pad_embedding = torch.empty(1, word_embedding_size).uniform_(-0.5, 0.5)
unk_embedding = torch.empty(1, word_embedding_size).uniform_(-0.5, 0.5)
word_embeddings = torch.cat([pad_embedding, unk_embedding, word_embeddings])

char_channels = [args.emsize] + [args.char_nhid] * args.char_layers
word_channels = [word_embedding_size + args.char_nhid] + [args.word_nhid] * args.word_layers

model = Model(
        charset_size=len(charset),
        char_embedding_size=args.emsize,
        char_channels=char_channels,
        char_padding_idx=charset["<pad>"],
        char_kernel_size=args.char_kernel_size,
        word_embedding=word_embeddings,
        word_channels=word_channels,
        word_kernel_size=args.word_kernel_size,
        num_tag=len(tag_set),
        dropout=args.dropout,
        emb_dropout=args.emb_dropout)


def raw_data_to_train_data(sentences: List[List[str]]):
    """
    convert raw data into model input and label
    """
    # TODO not hardcode this
    # TODO find a better way to do this function w/o so many lists
    record2idx = lambda record: vocab_set[record['word']]
    def record2charidx(record):
        word = record['word']
        return torch.LongTensor([charset[c] for c in word])
    record2tag = lambda record: tag_set[record['NER_tag']]
    batch_word_data = []
    batch_char_data = []
    batch_tag_data = []
    sentence_len = []
    for sentence in sentences:
        sentence_len.append(len(sentence))
        batch_word_data.append(
                torch.LongTensor(list(map(record2idx, sentence))))
        batch_char_data.extend(list(map(record2charidx, sentence)))
        batch_tag_data.append(
                torch.LongTensor(list(map(record2tag, sentence))))

    padded_word_data = pad_sequence(
            batch_word_data,
            batch_first=True,
            padding_value=vocab_set[PAD_TOKEN])

    # first we pad based on word length
    padded_char_data = pad_sequence(batch_char_data,
            padding_value=charset[PAD_TOKEN], batch_first=True)

    # TODO does this make sence
    padded_char_data = torch.split(padded_char_data, sentence_len)
    padded_char_data = pad_sequence(
            padded_char_data,
            batch_first=True,
            padding_value=charset[PAD_TOKEN],
            )

    padded_tag_data = pad_sequence(
            batch_tag_data,
            batch_first=True,
            padding_value = tag_set["O"],
    )
    return padded_word_data, padded_char_data, padded_tag_data

##### loss function, optimizer
# weighing loss differently for none 0
weight = [args.weight] * len(tag_set)
weight[tag_set["O"]] = 1
weight = torch.Tensor(weight)
criterion = nn.NLLLoss(weight, size_average=False)
# TODO reduce is avg by default, is this sensible?
# optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)
optimizer = torch.optim.Adam(model.parameters())


##### Dataset
train_raw_sentences = load_raw_dataset("artifacts/data/raw/CoNLL2003/eng.train")
test_raw_sentences = load_raw_dataset("artifacts/data/raw/CoNLL2003/eng.train")
if args.debug:
    train_raw_sentences = train_raw_sentences[:100]
    test_raw_sentences = test_raw_sentences[:100]
len_sorted_raw_sentences = sorted(train_raw_sentences, key=len, reverse=True)
train_data = CoNLLNERDataset(len_sorted_raw_sentences)
test_data = CoNLLNERDataset(test_raw_sentences)


def train_model(
        model,
        train_data,
        train_data_indices,
        args,
        start_time=None):
    model.train()
    train_losses = []
    total_loss = 0
    count = 0
    train_data_indices = list(train_data_indices)
    sampler = BatchSampler(
            RandomSampler(train_data_indices),
            args.batch_size,
            drop_last=False)

    for idx, indices in enumerate(sampler):
        batch = [train_data[train_data_indices[i]] for i in indices]
        word_data, char_data, tag_data = raw_data_to_train_data(batch)
        optimizer.zero_grad()
        output = model(word_data, char_data)
        # (batch size, seq_len, tag_set)
        output = output.view(-1, len(tag_set))
        target = tag_data.view(-1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += len(target)

        elapsed = time.monotonic() - start_time
        if (idx+1) % args.log_interval == 0:
            cur_loss = total_loss / count
            train_losses.append(cur_loss)
            logger.info(f"Batch: {idx}/{len(sampler)}"
                       f"\tLoss: {cur_loss}"
                       f"\tElapsed Time: {time_display(time.monotonic()-start_time)}")
            total_loss = 0
            count = 0
    if count > 0:
        train_losses.append(total_loss/count)
    return np.mean(train_losses)


def evaluate_model(
        model, test_data, args):
    model.eval()
    count = 0
    f1_scores = []
    losses = []
    total_f1_score = 0
    total_batch_count = 0
    total_loss = 0
    count = 0
    with torch.no_grad():
        # for batch
        for indices in BatchSampler(RandomSampler(test_data),args.batch_size, drop_last=False):
            batch_data = [test_data[i] for i in indices]
            word_data, char_data, tag_data = raw_data_to_train_data(batch_data)
            output = model(word_data, char_data)
            output = output.view(-1, len(tag_set))
            prediction = torch.argmax(output, dim=-1) # getting class
            target = tag_data.view(-1)
            total_f1_score += f1_score(
                prediction,
                target,
                labels=[i for i in range(len(tag_set))],
                average='macro', # TODO is this sensible
            )
            total_loss += criterion(output, target)
            total_batch_count += 1
            count += len(target)
    return total_loss/count, total_f1_score/total_batch_count


# experiment code
start_time = time.monotonic()
labelled_indices = set()
test_f1_scores = []
test_losses = []
labelled_data_counts = []
AL_sampler = ALRandomSampler(len(train_data))
curr_AL_epoch = 1

logger.info(f"{args.experiment_name} experiment")
n_labels = len(train_data)//args.al_epochs
try:
    while len(labelled_indices) < len(train_data):
        # AL step
        AL_sampler.label_n_elements(n_labels)
        labelled_indices = AL_sampler.labelled_idx_set

        labelled_data_counts.append(len(labelled_indices))
        logger.info("-" * 118)
        logger.info(
                f"AL Epoch: {curr_AL_epoch}/{args.al_epochs}"
                f"\tLabelled Data: {len(labelled_indices)}/{len(train_data)}"
                f"\tElapsed Time: {time_display(time.monotonic()-start_time)}")
        # train step
        for epoch in range(1, args.train_epochs+1):
            logger.info(f"Train Epoch: {epoch}/{args.train_epochs}"
                        f"\tElapsed Time: {time_display(time.monotonic()-start_time)}")
            train_loss = train_model(
                    model, train_data, labelled_indices, args, start_time)
        # validation step
        test_loss, test_f1_score = evaluate_model(model, test_data, args)
        test_losses.append(test_loss)
        test_f1_scores.append(test_f1_score)
        logger.info(
                f"AL Epoch:{curr_AL_epoch}/{curr_AL_epoch}"
                f"\tTest F1 Score: {test_f1_score}"
                f"\tTest Loss: {test_loss}"
                f"\tElapsed Time: {time_display(time.monotonic()-start_time)}")
        curr_AL_epoch += 1
        # save model
        model_fpath = os.path.join(EXPERIMENT_DIR, f"model_AL_epoch_{curr_AL_epoch}.pt")
        with open(model_fpath, 'wb') as f:
            torch.save(model, f)
        if max(test_f1_scores) == test_f1_score:
            model_fpath = os.path.join(
                    EXPERIMENT_DIR,
                    f"model_BEST.pt")
            with open(model_fpath, 'wb') as f:
                torch.save(model, f)
except KeyboardInterrupt:
    logger.warning('Exiting from training early!')


# TODO add train losses
n = np.min((len(labelled_data_counts), len(test_f1_scores), len(test_losses)))
results_df = pd.DataFrame.from_dict({
    "labelled_data_counts": labelled_data_counts[:n],
    "test_f1_scores": test_f1_scores[:n],
    "test_losses": test_losses[:n]})
results_df.to_csv(os.path.join(EXPERIMENT_DIR, "test_results"))


# TODO generate plots
plt.plot(labelled_data_counts[:n], test_f1_scores[:n])
plt.legend()
plt.savefig(os.path.join(EXPERIMENT_DIR, "test_n_labelled_f1_scores.png"))
