import argparse
import os
import time
import numpy as np
from typing import List
from bald.data.constants import (
    PAD_TOKEN,
)
from bald.model.model import Model
from bald.data.indexer import (
    Charset,
    Indexer,
    Vocabulary,
)
from bald.data.dataset import CoNLLNERDataset
from bald.data.samplers import (
    BatchRandomSubsetSamplerFactory,
)

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import logging


logger = logging.getLogger()
fh = logging.FileHandler("test.log")
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (default: 0.5)')
parser.add_argument('--emb_dropout', type=float, default=0.25,
                    help='dropout applied to the embedded layer (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.35,
                    help='gradient clip, -1 means no clip (default: 0.35)')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit (default: 30)')
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
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='report interval (default: 10)')
parser.add_argument('--lr', type=float, default=4,
                    help='initial learning rate (default: 4)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer type (default: SGD)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--weight', type=float, default=10.0,
                    help='manual rescaling weight given to each tag except "O"')
args = parser.parse_args()



DATA_PROCESSED_DIR = "data/processed/CoNLL2003/"
MODEL_DIR = "data/model"
MODEL_NAME = "ConNLL_NERModel"

charset = Charset()
vocab_set = Vocabulary()
vocab_set.load(os.path.join(DATA_PROCESSED_DIR, "word2vec_vocab.txt"))
tag_set = Indexer()
tag_set.load(os.path.join(DATA_PROCESSED_DIR, "tags_idx.txt"))


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


def sentences_to_data(sentences: List[List[str]]):
    # TODO not hardcode this
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

# weighing loss differently for none 0
weight = [args.weight] * len(tag_set)
weight[tag_set["O"]] = 1
weight = torch.Tensor(weight)
criterion = nn.NLLLoss(weight, size_average=False)
# TODO reduce is avg by default, is this sensible?
optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)


train_sentences = CoNLLNERDataset("data/raw/CoNLL2003/eng.train")


def train_model(epoch):
    logger.info("Hello")
    model.train()
    total_loss = 0
    count = 0
    sampler_factory = BatchRandomSubsetSamplerFactory(
            train_sentences,
            batch_size=args.batch_size,
            initial_indices=set([i for i in range(len(train_sentences))]))
    sampler = sampler_factory.get_data_sampler()
    for idx, indices in enumerate(sampler):
        batch = [train_sentences[i] for i in indices]
        word_data, char_data, tag_data = sentences_to_data(batch)
        optimizer.zero_grad()
        output = model(word_data, char_data)

        output = output.view(-1, len(tag_set))
        target = tag_data.view(-1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += len(target)
        if (idx+1) % args.log_interval == 0:
            cur_loss = total_loss / count
            # TODO abstract this out
            logger.info(f"Epoch {epoch}/{args.epochs} | Batch {idx}/{len(sampler)} | Loss {cur_loss}")
            total_loss = 0
            count = 0


train_model(0)



