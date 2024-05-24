#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 09:56:31 2022

@author: valancewang


Newsela - Test set 400



"""
# Library
import numpy as np
import pandas as pd
from massalign.core import *

import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import math, copy, time
# import matplotlib.pyplot as plt



######################################################
# Load library
import re
import random
import os
from datetime import datetime



# Define global constant
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
SOS_WORD = '<SOS>'
EOS_WORD = '<EOS>'
BLANK_WORD = '<PAD>'


# MAX_LENGTH
MAX_LENGTH = 200 # 60
MIN_LENGTH = 4
VALID_MAX_LENGTH = 200
VALID_MIN_LENGTH = 4
N_DATA = -1 # 10000

# VOCAB SIZE
USE_VOCAB_SIZE = True
VOCAB_SIZE = 40000 # 4000
UNK_REPLACEMENT = 3

# USE_GLOVE
USE_GLOVE = True

# Global parameters
N_LAYER = 4
N_HEAD = 5
EMBEDDING_DIM = 100 # 100
HIDDEN_DIM = 100 # 100
FF_DIM = 100
DROPOUT = 0.3 # 0.3


# CUDA
CUDA = torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
DEVICE = torch.device("cuda:2" if CUDA else "cpu")
DEVICES = [0,1,2,3]

# Training
N_ITERATION = 10
N_BATCH = 64


# -----------------------------------------------------------------------
# Load test set

# Define path and load data
if CUDA:
    data_path = '/ssddata1/data/valance/AI/data/text_simplification/newsela_Zhang/'
    model_path = '/ssddata1/data/valance/AI/Tree-Transformer/train_model/'
else:
    data_path = '/Users/valancewang/Dropbox/AI/data/text_simplification/newsela_Zhang/'
    model_path = '/Users/valancewang/Dropbox/AI/Models/Tree Transformer/'



fname_src = 'V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.test.src'
fname_dst = 'V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.test.dst'

# Load the test set
data_test_src = []
data_test_dst = []
data_test_src += open(data_path + fname_src, "r", encoding="utf-8").read().split('\n')
data_test_dst += open(data_path + fname_dst, "r", encoding="utf-8").read().split('\n')

data_test_src = [x.lower() for x in data_test_src]
data_test_dst = [x.lower() for x in data_test_dst]
data_test_src = [x.strip("'") for x in data_test_src]
data_test_dst = [x.strip("'") for x in data_test_dst]


# Clean up the dataset
# Remove cases that src == dst
# Check src == dst after removing non-alphabet
del_index = np.array([]).astype(int)
for k in range(len(data_test_src)-1, -1, -1):
    
    src = data_test_src[k]
    dst = data_test_dst[k]
    src = re.sub(r'\W+', ' ', src)
    dst = re.sub(r'\W+', ' ', dst)
    
    if src == dst:
        del_index = np.append(del_index, k)
        
data_test_src = np.delete(data_test_src, del_index)
data_test_dst = np.delete(data_test_dst, del_index)


# Clean up the dataset 
# Filter sentence length
# Tokenizer

del_index = np.array([]).astype(int)

for k in range(len(data_test_src)-1, -1, -1):
    
    src_len = len(data_test_src[k].split())
    dst_len = len(data_test_dst[k].split())
    
    if src_len <= VALID_MIN_LENGTH or src_len >= VALID_MAX_LENGTH \
        or dst_len <= VALID_MIN_LENGTH or dst_len >= VALID_MAX_LENGTH:
            del_index = np.append(del_index, k)
            
data_test_src = np.delete(data_test_src, del_index)
data_test_dst = np.delete(data_test_dst, del_index)




print('Test set - Source sentences: ', len(data_test_src))
print('Test set - Target sentences: ', len(data_test_dst))













# -------------------------------
# 
# BLEU and SARI
#
# ------------

# Evaluate
def evaluate_idx_test(idx, c):
    
    print("Test set: ", idx)
        
    src = data_test_src[idx]
    dst = data_test_dst[idx]

    print(">", src)
    print("=", dst)
    
    # Write log_file
    f = open('./newsela-400.txt','a')
    f.write("Test set: " + str(idx) + ' = ' + str(c))
    f.write('\n')
    f.write("> " + src)
    f.write('\n')
    f.write("= " + dst)
    f.write('\n')
    f.write('\n')
    f.write('\n')
    f.close()
    

    
#%% -------------------------------
# Evaluate 400

def evaluate_400():
            
    idx_list = list(range(len(data_test_src)))
        
    # random.seed(0)    
    # random.shuffle(idx_list)
    
    # idx_list = idx_list[0:400]
    
    c = 0 
    for idx in idx_list:
        evaluate_idx_test(idx, c)
        c += 1        
    return


evaluate_400()












