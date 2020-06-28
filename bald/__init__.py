from pathlib import Path
from .load_ner_dataset import load_ner_dataset

data_dir = Path(__file__).parent.parent / "data"
vectors_dir = Path(__file__).parent.parent / ".word_vectors_cache"
results_dir = Path(__file__).parent.parent / "results"

__all__ = [
    load_ner_dataset,
    data_dir,
    vectors_dir,
    results_dir,
]