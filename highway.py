#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__(self,eword):
        super(Highway, self).__init__()
        self.eword = eword
        
        self.proj = nn.Linear(self.eword,self.eword,
                              bias=True)
        self.gate = nn.Linear(self.eword,self.eword,
                              bias=True)
        
        
    def forward(self,X_conv_out):
        X_proj = F.relu(self.proj(X_conv_out))
        X_gate = torch.sigmoid(self.gate(X_conv_out))
        X_highway = X_proj * X_gate + (1 - X_gate) * X_conv_out

        return X_highway



    ### END YOUR CODE

