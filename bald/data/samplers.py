import random
from torch.utils.data.sampler import (
    BatchSampler,
    SubsetRandomSampler,
)
from torch.utils.data import (
    Dataset,
)


class BatchRandomSubsetSamplerFactory:
    """
    data sampler given subset of indices to sample from
    """
    def __init__(self, data: Dataset, batch_size, initial_indices):

        self.data = data
        self.indices = initial_indices
        self.batch_size = batch_size

    def get_data_sampler(self) -> BatchSampler:
        """
        returns index batch sampler current set indices
        """
        return BatchSampler(
                SubsetRandomSampler(list(self.indices)),
                batch_size=self.batch_size,
                drop_last=False)

    def update_indices(new_indices) -> None:
        indices.update(new_indices)


class ActiveLearningSamplerT:
    def __init__(self, data: Dataset):
        self.labelled_idx_set = set()
        self.unlabelled_idx_set = set(
                [i for i in range(len(data))])
        self.data = data

    def sample_pool(self, n_elements: int) -> None:
        assert Exception("not implemented")


class RandomALSampler(ActiveLearningSamplerT):
    def sample_pool(self, n_elements: int) -> None:
        pass


