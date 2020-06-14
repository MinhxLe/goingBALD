def parse_single_line(line):
    split = line.split()
    return (split[0], split[-1])

def load_ner_dataset(path):

    with open(path, "r") as f:
        new_line = f.readline()

        sentences = []
        while new_line:

            if new_line == "\n":
                new_line = f.readline()
                continue

            new_sentence = {"text":[], "tag":[]}
            while new_line and new_line != "\n":
                text, tag = parse_single_line(new_line)
                new_sentence["text"].append(text)
                new_sentence["tag"].append(tag)
                new_line = f.readline()

            sentences.append(new_sentence)
            new_line = f.readline()

    return sentences

def load_ner_dataset_old(path):

    with open(path, "r") as f:
        new_line = f.readline()

        sentences = []
        while new_line:

            if new_line == "\n":
                new_line = f.readline()
                continue

            new_sentence = []
            while new_line and new_line != "\n":
                text, tag = parse_single_line(new_line)
                new_entry = {
                    "text": text,
                    "tag": tag,
                }
                new_sentence.append(new_entry)
                new_line = f.readline()

            sentences.append(new_sentence)
            new_line = f.readline()

    return sentences
