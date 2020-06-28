import random
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


class BALDSampler(ActiveLearningSamplerT):

    def label_n_elements(self, n_elements, model, data):
        n_sampled = min(len(self.unlabelled_idx_set), n_elements)
