import heapq
import numpy as np
import random
import torch
from bald.utils import mask_sequence
from torch.utils.data.sampler import (
    BatchSampler,
    SequentialSampler,
    SubsetRandomSampler,
    Sampler,

)
from typing import List


class ActiveLearningSamplerT:
    """
    ActiveLearningSampler manages a dataset and what is labeled/unlabled
    """
    def __init__(
            self,
            n_elements):
        self.labelled_idx_set = set()
        self.unlabelled_idx_set = set([i for i in range(n_elements)])

    @property
    def n_labelled(self):
        return len(self.labelled_idx_set)

    def label_n_elements(self, n_elements: int, **kwargs) -> int:
        """
        chooses n labeled indices to labeled
        returns # of new elemnts labelled
        """
        # labels
        assert Exception("not implemented")

    def get_labelled_set(self):
        return self.labelled_idx_set


class ALRandomSampler(ActiveLearningSamplerT):

    def label_n_elements(self, n_elements: int) -> int:
        n_sampled = min(len(self.unlabelled_idx_set), n_elements)
        new_labels = set(random.sample(self.unlabelled_idx_set, n_sampled))
        self.labelled_idx_set |= new_labels
        self.unlabelled_idx_set -= new_labels
        return n_sampled


# class LeastConfidenceSampler(ActiveLearningSamplerT):
#     def label_n_elements(self, n_elements: int) -> int:
#         n_sampled = min(len(self.unlabelled_idx_set), n_elements)
#         new_labels = set(random.sample(self.unlabelled_idx_set, n_sampled))
#         self.labelled_idx_set = self.labelled_idx_set.union(new_labels)
#         self.unlabelled_idx_set = self.unlabelled_idx_set.difference(new_labels)


class FixedHeap:
    def __init__(self, key=lambda x:x):
        # https://stackoverflow.com/questions/8875706/heapq-with-custom-compare-predicate
        self.key = key
        self._heap = []
        self.index = 0

    def __len__(self):
        return len(self._heap)

    def data_to_heap_data(self, data):
        return (self.key(data), self.index, data)

    def push(self, data):
        heapq.heappush(self._heap, self.data_to_heap_data(data))
        self.index += 1

    def top(self):
        return self._heap[0][2]

    def pop(self):
        return heapq.heappop(self._heap)[2]

class MNLPSampler(ActiveLearningSamplerT):
    _batch_sampler_size = 32

    def __init__(self, train_data):
        n_elements = len(train_data)
        super().__init__(n_elements)
        self.train_data = train_data

    def label_n_elements(
            self,
            n_elements: int,
            model,
            data_process_fn,
            ) -> int:
        """
        chooses n labeled indices to labeled
        returns # of new elemnts labelled
        """
        n_to_sample = min(len(self.unlabelled_idx_set), n_elements)
        model.eval()
        unlabelled_indices = list(self.unlabelled_idx_set)
        heap = FixedHeap(key=lambda x : x[0])

        for indices in BatchSampler(SequentialSampler(unlabelled_indices),
                self._batch_sampler_size,
                drop_last=False):
            indices_to_evaluate = [unlabelled_indices[i] for i in indices]
            batch_data = [self.train_data[i] for i in indices_to_evaluate]
            model_input, _, seq_lens = data_process_fn(batch_data)
            # batch size, seq_len, n_tags
            output = model(*model_input)
            nll = output.max(axis=2)[0]
            nll_masked = mask_sequence(nll, seq_lens)
            nll_sentences = nll_masked.sum(axis=1)
            # mnlp = nll_sentences
            mnlp = torch.div(nll_sentences, seq_lens)
            # min heap
            for mnlp, index in zip(mnlp, indices_to_evaluate):
                mnlp = mnlp.item()
                if len(heap) < n_to_sample:
                    heap.push((-mnlp, index))
                else:
                    top_mnlp, _ = heap.top()
                    if mnlp < -top_mnlp:
                        heap.pop()
                        heap.push((-mnlp, index))
        while len(heap) > 0:
            mnlp, idx = heap.pop()
            self.labelled_idx_set.add(idx)
            self.unlabelled_idx_set.remove(idx)
        del heap
        return n_to_sample


class DropoutBALDSampler(ActiveLearningSamplerT):
    _batch_sampler_size = 32

    def __init__(self, train_data, mc_sample_size=100):
        n_elements = len(train_data)
        self.labelled_idx_set = set()
        self.unlabelled_idx_set = set([i for i in range(n_elements)])
        self.train_data = train_data
        self.mc_sample_size = mc_sample_size


    def label_n_elements(self, n_elements, model, data_process_fn):
        n_to_sample = min(len(self.unlabelled_idx_set), n_elements)
        model.train()  # we need drop out here so we keep model in training
        unlabelled_indices = list(self.unlabelled_idx_set)
        heap = FixedHeap(key=lambda x : x[0])

        for indices in BatchSampler(SequentialSampler(unlabelled_indices),
                self._batch_sampler_size,
                drop_last=False):
            indices_to_evaluate = [unlabelled_indices[i] for i in indices]
            batch_data = [self.train_data[i] for i in indices_to_evaluate]
            model_input, _, seq_lens = data_process_fn(batch_data)

            prediction_classes = None
            for _ in range(self.mc_sample_size):

                output = model(*model_input).data.numpy()
                batch_size, seq_len, n_tags = output.shape
                prediction = np.argmax(output, axis=-1)
                predictions_flattened = prediction.flatten()
                labels = np.eye(n_tags)[predictions_flattened]
                labels = labels.reshape(batch_size, seq_len, -1)
                if prediction_classes is None:
                    prediction_classes = labels
                else:
                    prediction_classes += labels

            max_predicted_class_count = np.max(prediction_classes, axis=-1)
            max_predicted_class_percent = max_predicted_class_count/self.mc_sample_size
            disagreement_percent = 1 - max_predicted_class_percent
            sentence_normalized_disagreement_precent = np.sum(disagreement_percent, axis=1)/seq_len
            # min heap
            for score, index in zip(sentence_normalized_disagreement_precent, indices_to_evaluate):
                if len(heap) < n_to_sample:
                    heap.push((score, index))
                else:
                    min_score, _ = heap.top()
                    if score > -min_score:
                        heap.pop()
                        heap.push((score, index))
        while len(heap) > 0:
            _, idx = heap.pop()
            self.labelled_idx_set.add(idx)
            self.unlabelled_idx_set.remove(idx)
        del heap
        return n_to_sample


class EpsilonGreedyBanditSampler(ActiveLearningSamplerT):
    # TODO add TS sampling
    def __init__(self, train_data, epsilon=0.01):
        self.n_elements = len(train_data)
        super().__init__(self.n_elements)
        self.samplers = [
            ALRandomSampler(self.n_elements),
            MNLPSampler(train_data)
        ]
        # we make sure we share the same set when labelling
        for sampler in self.samplers:
            sampler.unlabelled_idx_set = (
                self.unlabelled_idx_set)
            sampler.labelled_idx_set = (
                self.labelled_idx_set)
        self.n_samplers = len(self.samplers)
        self.q_score = np.zeros(self.n_samplers)
        self.arm_count = np.zeros(self.n_samplers)
        self.epsilon = epsilon

    def label_n_elements(
            self,
            n_elements: int,
            model,
            data_process_fn,
            ) -> int:
        sampler_selected = None
        if np.random.random() <= self.epsilon:
            arm = np.random.randint(0, self.n_samplers)
        else:
            arm = np.argmax(self.q_score)
        sampler_selected = self.samplers[arm]
        # TODO add logging of which arm selected
        if isinstance(sampler_selected, ALRandomSampler):
            n_labeled = sampler_selected.label_n_elements(n_elements)
        if isinstance(sampler_selected, MNLPSampler):
            n_labeled = sampler_selected.label_n_elements(n_elements, model, data_process_fn)
        return arm, n_labeled


    def update_q_score(self, action, reward):
        # TODO we can probably can do more aggressive score decay
        self.arm_count[action] += 1
        self.q_score[action] += (reward - self.q_score[action])/self.arm_count[action]

