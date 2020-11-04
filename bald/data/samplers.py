import heapq
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
    def __init__(self, n_elements):
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
        self.labelled_idx_set = self.labelled_idx_set.union(new_labels)
        self.unlabelled_idx_set = self.unlabelled_idx_set.difference(new_labels)
        return n_sampled


class LeastConfidenceSampler(ActiveLearningSamplerT):
    def label_n_elements(self, n_elements: int) -> int:
        n_sampled = min(len(self.unlabelled_idx_set), n_elements)
        new_labels = set(random.sample(self.unlabelled_idx_set, n_sampled))
        self.labelled_idx_set = self.labelled_idx_set.union(new_labels)
        self.unlabelled_idx_set = self.unlabelled_idx_set.difference(new_labels)


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
        self.labelled_idx_set = set()
        self.unlabelled_idx_set = set([i for i in range(n_elements)])
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


# class BALDSampler(ActiveLearningSamplerT):

#     def label_n_elements(self, n_elements, model, data):
#         n_sampled = min(len(self.unlabelled_idx_set), n_elements)
