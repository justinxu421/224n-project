#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
nmt.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def pad_sents_char(sents, char_pad_token):
    """ Pad list of sentences according to the longest sentence in the batch and max_word_length.
    @param sents (list[list[list[int]]]): list of sentences, result of `words2charindices()` 
        from `vocab.py`
    @param char_pad_token (int): index of the character-padding token
    @returns sents_padded (list[list[list[int]]]): list of sentences where sentences/words shorter
        than the max length sentence/word are padded out with the appropriate pad token, such that
        each sentence in the batch now has same number of words and each word has an equal 
        number of characters
        Output shape: (batch_size, max_sentence_length, max_word_length)
    """
    # Words longer than 21 characters should be truncated
    max_word_length = 21 

    ### YOUR CODE HERE for part 1f
    ### TODO:
    ###     Perform necessary padding to the sentences in the batch similar to the pad_sents() 
    ###     method below using the padding character from the arguments. You should ensure all 
    ###     sentences have the same number of words and each word has the same number of 
    ###     characters. 
    ###     Set padding words to a `max_word_length` sized vector of padding characters.  
    ###
    ###     You should NOT use the method `pad_sents()` below because of the way it handles 
    ###     padding and unknown words.

    sents_padded = []

    max_sent_len = max(len(s) for s in sents)
    batch_size = len(sents)

    for s in sents:
        padded_s = []
        for w in s:
            padded_w = []
            if len(w) > max_word_length:
                padded_w = w[:max_word_length]
            else:
                padded_w = [char_pad_token] * max_word_length
                padded_w[:len(w)] = w 
            padded_s.append(padded_w)

        if len(padded_s) < max_sent_len:
            padding_word = [char_pad_token] * max_word_length
            padded_s = padded_s + [padding_word for i in range(max_sent_len - len(padded_s))]
        sents_padded.append(padded_s)

    return sents_padded
    ### END YOUR CODE

    

def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    ### YOUR CODE HERE (~6 Lines)
    def pad(sent,max_len):
        if len(sent) < max_len:
            return sent + [pad_token] * (max_len - len(sent)) 
        elif len(sent) == max_len:
            return sent 

    max_len = max([len(sent) for sent in sents]) # why is list comprehension faster?
    sents_padded = [pad(sent,max_len) for sent in sents]
    ### END YOUR CODE

    return sents_padded



def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        sents_a = [e[0] for e in examples]
        sents_b = [e[1] for e in examples]
        sents_c = [e[2] for e in examples]
        tgt_sents = [e[3] for e in examples]

        yield sents_a, sents_b,sents_c, tgt_sents

