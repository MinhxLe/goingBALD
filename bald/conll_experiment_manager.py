import argparse
import logging
import numpy as np
import pandas as pd
import os
import sys
import time
import torch
import json
import tensorflow as tf
import torch.nn as nn
import torch.optim as optim
from bald.constants import (
        CONLL_RAW_TRAIN_FNAME,
        CONLL_RAW_TESTA_FNAME,
        CONLL_DATA_PROCESSED_DIR,
        EXPERIMENTS_RESULT_DIR,
)
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
        MNLPSampler,
        DropoutBALDSampler,
)
from bald.log_utils import (
    time_display,
    set_up_experiment_logging,
)

from bald.utils import mask_sequence
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




class CoNLL2003ActiveLearningExperimentManager:
    """
    - handles managing data set
    - interface between model and raw data
    - logging
    - model generation
    """
    def __init__(self, args, save_exp=False):
        self.args = args

        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        self.save_exp = save_exp

        experiments_root_path = os.path.join(EXPERIMENTS_RESULT_DIR, self.experiment_name)
        self.logger, self.tf_summary_writer, self.model_snapshot_dir = (
            set_up_experiment_logging(
                self.experiment_name,
                log_fpath=os.path.join(experiments_root_path, "experiment.log"),
                model_snapshot_dir=os.path.join(experiments_root_path, "model_snapshots"),
                metrics_dir=os.path.join(experiments_root_path, "metrics"),
                stdout=args.stdout,
                clear_old_data=True,
            )
        )

        self._init_data()
        self.model = self.get_model()
        self.criterion = self._get_criterion()
        self.optimizer = self._get_optimizer(self.model)
        torch.manual_seed(args.seed)

    @property
    def experiment_name(self):
        return f"{self.args.experiment_name}"

    @property
    def experiment_dir(self):
        return os.path.join(EXPERIMENTS_RESULT_DIR, self.experiment_name)

    def get_model(self) -> Model:
        charset = self.char_set
        tag_set = self.tag_set
        args = self.args
        word_embeddings = torch.Tensor(np.load(args.word_embedding_fname))
        word_embedding_size = word_embeddings.size(1)
        pad_embedding = torch.empty(1, word_embedding_size).uniform_(-0.5, 0.5)
        unk_embedding = torch.empty(1, word_embedding_size).uniform_(-0.5, 0.5)
        word_embeddings = torch.cat([pad_embedding, unk_embedding, word_embeddings])

        char_channels = [args.emsize] + [args.char_nhid] * args.char_layers
        word_channels = [word_embedding_size + args.char_nhid] + [args.word_nhid] * args.word_layers
        return Model(
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

    def _get_criterion(self):
        weight = [self.args.weight] * len(self.tag_set)
        weight[self.tag_set["O"]] = 1
        weight = torch.Tensor(weight)
        return nn.NLLLoss(weight, reduction='none')

    def _get_optimizer(self, model):
       return torch.optim.Adam(model.parameters())

    def _init_data(self) -> None:
        self.char_set = Charset()
        self.vocab_set = Vocabulary()
        self.vocab_set.load(os.path.join(CONLL_DATA_PROCESSED_DIR, "word2vec_vocab_idx.txt"))
        self.tag_set = Indexer()
        self.tag_set.load(os.path.join(CONLL_DATA_PROCESSED_DIR, "tags_idx.txt"))

        train_raw_sentences = load_raw_dataset(CONLL_RAW_TRAIN_FNAME)
        test_raw_sentences = load_raw_dataset(CONLL_RAW_TESTA_FNAME)
        if self.args.debug:
            train_raw_sentences = train_raw_sentences[:100]
            test_raw_sentences = test_raw_sentences[:100]
        len_sorted_raw_sentences = sorted(train_raw_sentences, key=len, reverse=True)
        self.train_data = CoNLLNERDataset(len_sorted_raw_sentences)
        self.test_data = CoNLLNERDataset(test_raw_sentences)


    def _raw_data_to_train_data(self, sentences: List[List[str]]):
        """
        convert raw data into model input, label, and seq len
        """
        # TODO does this belong as part of model
        vocab_set = self.vocab_set
        charset = self.char_set
        tag_set = self.tag_set
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
        return (padded_word_data, padded_char_data), padded_tag_data, torch.Tensor(sentence_len)

    def _get_loss(self, prediction, target, sentence_lens):
        loss = self.criterion(prediction.view(-1, len(self.tag_set)), target.view(-1)).view(target.shape)
        return torch.mean(mask_sequence(loss, sentence_lens))

    def train_model_step(self, train_data_indices, start_time=None):
        # TODO we don't retrain here, is that sensible?
        model = self.model
        train_data = self.train_data
        args = self.args
        logger = self.logger
        optimizer = self.optimizer
        criterion = self.criterion
        tag_set = self.tag_set
        train_data = self.train_data

        model.train()
        train_losses = []
        total_loss = 0
        count = 0
        elapsed = None
        train_data_indices = list(train_data_indices)
        sampler = BatchSampler(
                RandomSampler(train_data_indices),
                args.batch_size,
                drop_last=False)

        for idx, indices in enumerate(sampler):
            batch = [train_data[train_data_indices[i]] for i in indices]
            model_input, tag_data, sentence_lens = self._raw_data_to_train_data(batch)
            optimizer.zero_grad()
            output = model(*model_input)
            # (batch size, seq_len, tag_set)
            target = tag_data
            loss = self._get_loss(output, target, sentence_lens)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += len(target)

            if start_time:
                elapsed = time.monotonic() - start_time
            if (idx+1) % args.log_interval == 0:

                cur_loss = total_loss / count
                train_losses.append(cur_loss)
                logger.info(f"Batch: {idx}/{len(sampler)}"
                        f"\tLoss: {cur_loss}"
                        f"\tElapsed Time: {time_display(elapsed)}")
                total_loss = 0
                count = 0
        if count > 0:
            train_losses.append(total_loss/count)
        self.model = model
        return train_losses

    def evaluate_model_step(self, test_data=None):
        model = self.model
        args = self.args
        logger = self.logger
        optimizer = self.optimizer
        criterion = self.criterion
        tag_set = self.tag_set
        train_data = self.train_data
        if test_data is None:
            test_data = self.test_data
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
            for indices in BatchSampler(SequentialSampler(test_data),args.batch_size, drop_last=False):
                batch_data = [test_data[i] for i in indices]
                model_input, tag_data, sentence_lens = self._raw_data_to_train_data(batch_data)
                output = model(*model_input)
                prediction = torch.argmax(output.view(-1, len(self.tag_set)), dim=-1) # getting class
                target = tag_data
                # TODO should i remove contributino from mask
                total_f1_score += f1_score(
                    prediction.view(-1),
                    target.view(-1),
                    labels=[i for i in range(len(tag_set))],
                    average='macro', # TODO is this sensible
                )
                total_loss += self._get_loss(output, target, sentence_lens)
                total_batch_count += 1
                count += len(target)
        return total_loss/count, total_f1_score/total_batch_count

    def _get_sampler(self):
        raise Exception("pick a sampler")

    def label_n_elements(self, sampler, n_elements):
        raise Exception("using sampler not implemented")

    def run_experiment(self):
        args = self.args
        train_data = self.train_data
        logger = self.logger
        labelled_indices = set()
        test_f1_scores = []
        test_losses = []
        labelled_data_counts = []
        AL_sampler = self._get_sampler()
        curr_AL_epoch = 1
        start_time = time.monotonic()

        logger.info(f"{args.experiment_name} experiment")
        n_labels = len(train_data)//args.al_epochs
        try:
            while len(labelled_indices) < len(train_data):
                # AL step
                self.label_n_elements(AL_sampler, n_labels)
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
                    train_loss = self.train_model_step(labelled_indices, start_time)
                #final train loss
                train_loss, train_f1_score = self.evaluate_model_step(self.train_data)
                with self.tf_summary_writer.as_default():
                    tf.summary.scalar("train f1 score", train_f1_score, step=curr_AL_epoch)
                    tf.summary.scalar("train loss", train_loss, step=curr_AL_epoch)
                # validation step
                test_loss, test_f1_score = self.evaluate_model_step()
                test_losses.append(test_loss)
                test_f1_scores.append(test_f1_score)
                logger.info(
                        f"AL Epoch:{curr_AL_epoch}/{args.al_epochs}"
                        f"\tTest F1 Score: {test_f1_score}"
                        f"\tTest Loss: {test_loss}"
                        f"\tElapsed Time: {time_display(time.monotonic()-start_time)}")
                with self.tf_summary_writer.as_default():
                    tf.summary.scalar("test f1 score", test_f1_score, step=curr_AL_epoch)
                    tf.summary.scalar("test loss", test_loss, step=curr_AL_epoch)
                # save model
                if self.save_exp:
                    model_fpath = os.path.join(self.model_snapshot_dir, f"model_AL_epoch_{curr_AL_epoch}.pt")
                    torch.save(self.model.state_dict(), model_fpath)

                    if max(test_f1_scores) == test_f1_score:
                        model_fpath = os.path.join(self.model_snapshot_dir, f"model_BEST.pt")
                        torch.save(self.model.state_dict(), model_fpath)
                curr_AL_epoch += 1
        except KeyboardInterrupt:
            logger.warning('Exiting from training early!')

        # TODO add train losses
        # TODO extract out
        n = np.min((len(labelled_data_counts), len(test_f1_scores), len(test_losses)))
        results_df = pd.DataFrame.from_dict({
            "labelled_data_counts": labelled_data_counts[:n],
            "test_f1_scores": test_f1_scores[:n],
            "test_losses": test_losses[:n]})
        results_df.to_csv(os.path.join(self.experiment_dir, "test_results.csv"))


class RandomExperimentManager(CoNLL2003ActiveLearningExperimentManager):

    def __init__(self, args, save_exp=False):
        super().__init__(args, save_exp)

    def _get_sampler(self):
        return ALRandomSampler(len(self.train_data))

    def label_n_elements(self, sampler, n_elements):
        sampler.label_n_elements(n_elements)


class MNLPExperimentManager(CoNLL2003ActiveLearningExperimentManager):

    def __init__(self, args, save_exp=False):
        super().__init__(args, save_exp)

    def _get_sampler(self):
        return MNLPSampler(self.train_data)

    def label_n_elements(self, sampler, n_elements):
        return sampler.label_n_elements(n_elements,
                self.model, self._raw_data_to_train_data)


class DropoutBALDExperimentManager(CoNLL2003ActiveLearningExperimentManager):

    def __init__(self, args, save_exp=False):
        super().__init__(args, save_exp)

    def _get_sampler(self):
        return DropoutBALDSampler(self.train_data, self.args.mc_sample_size)

    def label_n_elements(self, sampler, n_elements):
        return sampler.label_n_elements(n_elements,
                self.model, self._raw_data_to_train_data)
