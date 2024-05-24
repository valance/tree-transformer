#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 07:47:07 2021

@author: valance

Transformer
- Plot training loss
- Plot validation loss


# How to use Tensorboard: 
tensorboard --logdir=runs

"""


# Library
import math
import pandas as pd

# ------------------------------------------------------
# Training loss
# Read data
path = '/Users/valancewang/Dropbox/AI/data analysis/'
names = ['iteration','Epoch Step','average_loss']
loss_table = pd.read_csv(path + 'log_file_newsela_GloVe_40000_100_lr_3_5000.txt', 
                                 header = None, names = names)
# loss_table_2 = pd.read_csv(path + 'log_file_newsela_GloVe_40000_100_5000.txt', 
#                                   header = None, names = names)


# Process loss_table
row_idx = []
for l in range(len(loss_table)):
    
    iteration = loss_table.loc[l, 'iteration']
    idx = loss_table.loc[l, 'Epoch Step']
    loss = loss_table.loc[l, 'average_loss']
    
    try:
        iteration = int(iteration.split(':')[-1])
        idx = int(idx.split(':')[-1])
        loss = float(loss.split(':')[-1])
    except:
        iteration = math.nan
        idx = math.nan
        loss = math.nan
        row_idx.append(l)
                                      
    loss_table.loc[l, 'iteration'] = iteration  + 4000
    loss_table.loc[l, 'Epoch Step'] = idx
    loss_table.loc[l, 'average_loss'] = loss


# # Process loss_table_2
# row_idx = []
# for l in range(len(loss_table_2)):
    
#     iteration = loss_table_2.loc[l, 'iteration']
#     idx = loss_table_2.loc[l, 'Epoch Step']
#     loss = loss_table_2.loc[l, 'average_loss']
    
#     try:
#         iteration = int(iteration.split(':')[-1])
#         idx = int(idx.split(':')[-1])
#         loss = float(loss.split(':')[-1])
#     except:
#         iteration = math.nan
#         idx = math.nan
#         loss = math.nan
#         row_idx.append(l)
                                      
#     loss_table_2.loc[l, 'iteration'] = iteration + 2000
#     loss_table_2.loc[l, 'Epoch Step'] = idx
#     loss_table_2.loc[l, 'average_loss'] = loss



# # matplotlib
# import matplotlib.pyplot as plt
# plt.ylabel('Training loss')
# plt.plot(loss_table['average_loss'][1:])

# for l in range(row_idx[0],row_idx[1]):
#     loss_table.loc[l, 'iteration'] += 10



#%% tensorboard --logdir=runs

# Tensorboard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

for l in range(len(loss_table)):
    
    iteration = loss_table.loc[l, 'iteration']
    loss = loss_table.loc[l, 'average_loss']
    if not math.isnan(iteration):
        writer.add_scalar('Loss/train', loss, iteration)
        
        
# for l in range(len(loss_table_2)):
    
#     iteration = loss_table_2.loc[l, 'iteration']
#     loss = loss_table_2.loc[l, 'average_loss']
#     if not math.isnan(iteration):
#         writer.add_scalar('Loss/train', loss, iteration)





writer.close()












# # ------------------------------------------------------
# # Validation loss
# # Read data
# path = '/Users/valance/Dropbox/AI/data analysis/'
# names = ['model','validation_loss']
# valid_loss = pd.read_csv(path + 'validation_loss_10Kw.txt', 
#                                  header = None, names = names)

# # Process loss_table
# for l in range(len(valid_loss)):
    
#     loss = valid_loss.loc[l, 'validation_loss']
#     if type(loss) == str:    
#         loss = float(loss.split(': ')[-1])
                                              
#     valid_loss.loc[l, 'validation_loss'] = loss



# # Tensorboard
# for l in range(len(valid_loss)):
    
#     iteration = l+1
#     loss = valid_loss.loc[l, 'validation_loss']
#     if not math.isnan(loss):
#         writer.add_scalar('Loss/validation', loss, iteration)
        
# writer.close()
