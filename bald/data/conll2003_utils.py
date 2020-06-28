from typing import Dict, List
from collections import defaultdict
from bald.data.indexer import Vocabulary, Indexer


_NEW_DOC_LINE = "-DOCSTART-"
_NEW_SENTENCE_LINE = '\n'


def _parse_single_line(line: str) -> Dict:
    """
    return parsed cornll DIct
    """
    line_data = line.split()
    return {
        "word": line_data[0],
        "POS_tag": line_data[1],
        "SC_tag": line_data[2],
        "NER_tag": _parse_NER_tag(line_data[3]),
    }


def _parse_NER_tag(raw_tag: str) -> str:
    return raw_tag.split('-')[-1]


def load_raw_dataset(fpath: str) -> List[List[Dict]]:
    """
    loads in raw cornll dataset to return list of
    list of parsed sentences
    """
    with open(fpath, "r") as f:
        sentences = []
        curr_sentence = []

        new_line = f.readline()
        while new_line:
            if new_line == _NEW_SENTENCE_LINE: # of line
                sentences.append(curr_sentence)
                curr_sentence = []
            elif _NEW_DOC_LINE in new_line:
                f.readline() # skip another blankspace
            else:
                parsed_data = _parse_single_line(new_line)
                curr_sentence.append(parsed_data)
            new_line = f.readline()
        return sentences

# def generate_vocab_set(
#         sentences: List[List[Dict]]) -> Vocabulary:
#     # ordering based on frequency
#     word_frequency = defaultdict(int)
#     for sentence in sentences:
#         for word_dict in sentence:
#             word_str = word_dict["word"]
#             word_frequency[word_str] += 1
#     words_sorted = sorted(
#             word_frequency.keys(),
#             reverse=True,
#             key=lambda word : word_frequency[word])
#     vocab = Vocabulary()
#     for word in words_sorted:
#         vocab.add(word)
#     return vocab

def generate_NER_tag_set(
        sentences: List[List[Dict]]) -> Indexer:
    tag_frequency = defaultdict(int)
    for sentence in sentences:
        for word_dict in sentence:
            tag_str = word_dict["NER_tag"]
            tag_frequency[tag_str] += 1
    tags_sorted = sorted(
            tag_frequency.keys(),
            reverse=True,
            key=lambda tag : tag_frequency[tag])
    indexer = Indexer()
    for tag in tags_sorted:
        indexer.add(tag)
    return indexer
