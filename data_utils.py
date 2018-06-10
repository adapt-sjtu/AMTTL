# encoding=utf-8
# Project: transfer_cws
# Author: xingjunjie
# Create Time: 17/08/2017 11:55 AM on PyCharm

import os
import numpy as np

UNK = 'UNK_WORD'

"""
Data utils for tf implement
"""


def _pad_sequences(sequences, pad_tok, max_length):
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok):
    max_length = max(map(lambda x: len(x), sequences))
    sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)

    return sequence_padded, sequence_length


def minibatches(data, minibatch_size, circle=False):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)
    Returns:
        list of tuples
    """
    # if mini_padding:
    #     data.sort(key=lambda x: len(x[0]))
    x_batch, y_batch, z_batch = [], [], []

    while True:
        for (x, y, z) in data:
            if len(x_batch) == minibatch_size:
                yield x_batch, y_batch, z_batch
                x_batch, y_batch, z_batch = [], [], []

            if type(x[0]) == tuple:
                x = zip(*x)
            x_batch += [x]
            y_batch += [y]
            z_batch += [z]

        if not circle:
            break

    if len(x_batch) != 0:
        yield x_batch, y_batch, z_batch


def minibatches_evaluate(data, minibatch_size, mini_padding=True):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)
    Returns:
        list of tuples
    """
    # if mini_padding:
    #     data.sort(key=lambda x: len(x[0]))
    x_batch, y_batch, z_batch = [], [], []
    for x, y, z in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch, z_batch
            x_batch, y_batch, z_batch = [], [], []

        x_batch += [x]
        y_batch += [y]
        z_batch += [z]

    if len(x_batch) != 0:
        yield x_batch, y_batch, z_batch


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """
    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk + start of a chunk!
        tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
        if chunk_type is None:
            chunk_type, chunk_start = tok_chunk_type, i
        elif tok_chunk_type != chunk_type or tok_chunk_class == "B" or tok_chunk_class == "S":
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = tok_chunk_type, i

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


class EvaluateSet(object):
    def __init__(self, filename, processing_word, processing_target_word=None):
        self.filename = filename
        self.processing_word = processing_word
        self.processing_target_word = processing_target_word
        self.length = None

    def __iter__(self):
        with open(self.filename, 'r') as infile:
            for line in infile:
                line = line[:-1]
                if line:
                    idx = self.processing_word(line)
                    if self.processing_target_word is not None:
                        target_idx = self.processing_target_word(line)
                        yield (idx, target_idx, list(line))
                    else:
                        yield (idx, [], list(line))

    def __len__(self):
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


class Dataset(object):
    def __init__(self, filename, processing_word, processing_tag, processing_target_word=None,
                 transfer_flag=0):
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.processing_target_word = processing_target_word
        self.transfer_flag = transfer_flag
        self.length = None

    def __iter__(self):
        with open(self.filename, encoding='utf-8') as infile:
            words, tags, target_words = [], [], []
            for line in infile:
                line = line[:-1]
                if len(line) == 0:
                    if len(words):
                        yield words, tags, target_words
                        words, tags, target_words = [], [], []
                else:
                    items = line.split('\t')
                    assert len(items) == 2
                    word, tag = items[0], items[1]

                    word_idx = self.processing_word(word)
                    tag_idx = self.processing_tag(tag)
                    words.append(word_idx)
                    tags.append(tag_idx)

                    if self.transfer_flag:
                        target_idx = self.processing_target_word(word)
                        target_words.append(target_idx)

    def __len__(self):
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def get_vocabs(datasets):
    """
    Args:
        datasets: a list of dataset objects
    Return:
        a set of all the words in the dataset
    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags


def load_pre_train(file_path):
    pre_dictionary = dict()
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
        try:
            num, embedding_size = lines[0].split()
            num = int(num)
            embedding_size = int(embedding_size)
        except Exception as e:
            return None, None
        assert num == len(lines) - 1
        embedding = np.zeros((num, embedding_size), dtype=float)
        for index, line in enumerate(lines[1:]):
            items = line.split()
            assert len(items) == embedding_size + 1, print(index)
            word = items[0]
            pre_dictionary[word] = len(pre_dictionary)
            embed = np.zeros((embedding_size,), dtype=float)
            for i in range(0, embedding_size):
                embed[i] = float(items[i + 1])
            embedding[index] = embed
        return pre_dictionary, embedding


def load_vocab(filename, base_dict=None):
    if base_dict == None:
        base_dict = dict()
    if UNK not in base_dict:
        base_dict[UNK] = len(base_dict)

    word_dict = base_dict
    tag_dict = dict()
    # tag_dict[UNK] = len(tag_dict)

    with open(filename, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line[:-1]
            if len(line) > 0:
                items = line.split('\t')
                assert len(items) == 2
                if items[0] not in word_dict:
                    word_dict[items[0]] = len(word_dict)
                if items[1] not in tag_dict:
                    tag_dict[items[1]] = len(tag_dict)
    return word_dict, tag_dict


def get_processing(vocab=None, default_key=UNK):
    def f(item):
        if len(item) > 1:
            item = list(item)
            return [f(i) for i in item]
        if item in vocab:
            item = vocab[item]
        else:
            item = vocab[default_key]
        return item

    return f
