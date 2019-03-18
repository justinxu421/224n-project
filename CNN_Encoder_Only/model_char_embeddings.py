#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
from highway import Highway
from cnn import CNN
# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

# from cnn import CNN
# from highway import Highway

# End "do not change" 

class ModelCharEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab,dropout_rate = 0.3):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelCharEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        #embed_size is e_word
        #padding_idx=vocab.char2id['<pad>']
        #the handout says e_char = 50
        self.embed_size = embed_size
        self.embeddings = nn.Embedding(len(vocab.char2id), 50)
        self.highway = Highway(embed_size)
        self.cnn = CNN(50,embed_size)
        self.dropout =  nn.Dropout(p=dropout_rate)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        embeddings = self.embeddings(input)
        # input = input.permute(1,0,2)

        #now input is ->(batch_size, sentence_length,max_word_length)

        #now embeddings is ->(batch_size, sentence_length,max_word_length, e_char)
        #reshape to -> (batch_size x sentence_length,max_word_length, e_char)
        sentence_length, batch_size, max_word_length, e_char = embeddings.shape
        embeddings = embeddings.view(batch_size * sentence_length,e_char, max_word_length)
        #cnn_embeds -> batch_size x sentence_length, e_word
        cnn_embeds = self.cnn.forward(embeddings)
        output = self.highway.forward(cnn_embeds)
        x_wordemb =  self.dropout(output)
        _ , e_word = x_wordemb.shape
        x_wordemb = x_wordemb.view(sentence_length,batch_size, e_word)
        return x_wordemb
        ### END YOUR CODE

