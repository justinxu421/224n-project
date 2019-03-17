#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self,embedding_size):
        """Initiates the highway function
        @param embedding_size is the size of an output embedding (e_word)
        """
        super(Highway, self).__init__()

        self.w_proj = nn.Linear(embedding_size,embedding_size, bias=True)
        self.w_gate = nn.Linear(embedding_size,embedding_size, bias=True)

    def forward(self,x):
        """forward function through highway
        @param x is of  size (batch, e_word)
        @param output is of shape (batch, e_word)
        """
        x_proj = F.relu(self.w_proj(x))
        x_gate = torch.sigmoid(self.w_gate(x))

        x_highway = torch.add(torch.mul(x_proj,x_gate), torch.mul((1- x_gate), x))
        return x_highway

### END YOUR CODE 

