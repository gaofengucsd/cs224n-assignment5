#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    
    

    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g

    def __init__(self, echar, eword, kernel_size=5,padding=1):
        super(CNN, self).__init__()
        self.echar = echar
        self.eword = eword
        self.conv1d = nn.Conv1d(in_channels=echar,out_channels=eword,kernel_size=kernel_size,padding=padding,bias=True)
        
    def forward(self,X_reshaped):
        #X_reshaped: (batch_size, echar, mword)
        x_conv = self.conv1d(X_reshaped)
        x_conv_out = torch.max(x_conv, dim=2)[0]
        #X_conv_out  (batch_size,eword)
        return x_conv_out


    ### END YOUR CODE

