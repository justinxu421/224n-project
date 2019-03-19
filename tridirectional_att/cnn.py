#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self,e_char,e_word,kernel_size=5):
        """Initiates the CNN function
        """
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(e_char,e_word, kernel_size, bias=True)

    def forward(self,x):
        """ fowward cnn
        @param x -> batch size, e_char
        """
        #pretorch max batch size, e_char , e_word (output channels)
        #post max batch size, e_word (output channels)
        conv = F.relu(self.conv(x))
        return torch.max(conv,2)[0]
### END YOUR CODE

