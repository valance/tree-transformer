#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 09:22:25 2022

@author: valance

Tree Transformer Evaluate on Turk Dataset

"""


# Library
import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import math, copy, time
# import matplotlib.pyplot as plt

# library
import re
import random
import os
# from datetime import datetime


# Code library
import torch.optim as optim
import subprocess
from models import *
from utils import *
from parse import *
from bert_optimizer import BertAdam
from transformers import BertTokenizer


# --------------------------------------------------------
# Define global constant
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
SOS_WORD = '<SOS>'
EOS_WORD = '<EOS>'
BLANK_WORD = '<PAD>'


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
N_LAYER = 10
N_HEAD = 5
EMBEDDING_DIM = 300  # 300
HIDDEN_DIM = 300  # 300
FF_DIM = 300
DROPOUT = 0.3 # 0.2

# CUDA
MultiGPU = False
CUDA = torch.cuda.is_available()
# os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
DEVICE = torch.device("cuda:3" if CUDA else "cpu")
# DEVICES = [3,4]   #4
if CUDA:
    no_cuda = False
else:
    no_cuda = True


# Training
N_ITERATION = 50
N_BATCH = 128 

# -----------------------------------------------------------------------
# Define path and load data
if CUDA:
    path = '/ssddata1/data/valance/AI/data/text_simplification/wikilarge/'
    model_path = '/ssddata1/data/valance/AI/Tree-Transformer/train_model/'
else:
    path = '/Users/valancewang/Dropbox/AI/data/text_simplification/wikilarge/'
    model_path = '/Users/valancewang/Dropbox/AI/Models/Tree Transformer/'
fname_src = 'wiki.full.aner.train.src'
fname_dst = 'wiki.full.aner.train.dst'


data_src = []
data_dst = []
data_src += open(path + fname_src, "r", encoding="utf-8").read().split('\n')
data_dst += open(path + fname_dst, "r", encoding="utf-8").read().split('\n')

data_src = [x.lower() for x in data_src]
data_dst = [x.lower() for x in data_dst]
data_src = [x.strip("'") for x in data_src]
data_dst = [x.strip("'") for x in data_dst]



print ("Source sentences (before): ", len(data_src))
print ("Target sentences (before): ", len(data_dst))



# Clean up the dataset
# Remove cases that src == dst
# Check src == dst after removing non-alphabet
del_index = np.array([]).astype(int)
for k in range(len(data_src)-1, -1, -1):
    
    src = data_src[k]
    dst = data_dst[k]
    src = re.sub(r'\W+', ' ', src)
    dst = re.sub(r'\W+', ' ', dst)
    
    if src == dst:
        del_index = np.append(del_index, k)
        
data_src = np.delete(data_src, del_index)
data_dst = np.delete(data_dst, del_index)
        
print('Remove src == tgt')
print('Source sentences: ', len(data_src))
print('Target sentences: ', len(data_dst))



# Clean up the dataset 
# Filter sentence length
del_index = np.array([]).astype(int)

for k in range(len(data_src)-1, -1, -1):
    
    src_len = len(data_src[k].split())
    dst_len = len(data_dst[k].split())
    
    if src_len <= MIN_LENGTH or src_len >= MAX_LENGTH \
        or dst_len <= MIN_LENGTH or dst_len >= MAX_LENGTH:
            del_index = np.append(del_index, k)
            
data_src = np.delete(data_src, del_index)
data_dst = np.delete(data_dst, del_index)

print('Filter sentence length')
print('Source sentences: ', len(data_src))
print('Target sentences: ', len(data_dst))



# Reduce data size
if N_DATA > 0:
    data_src = data_src[0:N_DATA]
    data_dst = data_dst[0:N_DATA]

corpus = np.concatenate( (data_src , data_dst) )


print('Source sentences: ', len(data_src))
print('Target sentences: ', len(data_dst))


# -----------------------------------------------------------------------
# Dataset
class TSDataset(Dataset):
    def __init__(self, data_src, data_dst):
        self.n_samples = len(data_src)
        self.x_data = data_src
        self.y_data = data_dst
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples


dataset = TSDataset(data_src, data_dst)

# DataLoader
train_loader = DataLoader(dataset = dataset,
                          batch_size = N_BATCH,
                          shuffle = True)

# -----------------------------------------------------------------------
class LangEmbed:
    def __init__(self, name):
        self.name = name
        self.word2index = {'<PAD>':PAD_token, '<SOS>':SOS_token, '<EOS>':EOS_token, '<UNK>':UNK_token}
        self.word2count = {}
        self.index2word = {PAD_token:'<PAD>', SOS_token:'<SOS>', EOS_token:'<EOS>', UNK_token:'<UNK>'}
        self.n_words = 4
        
    def addSentence(self, corpus): 
        # Corpus is a list of sentences.
        for i in range(len(corpus)):
            sentence = str(corpus[i])
            self.addWord(sentence)

            
    def addWord(self, sentence):
        tokens = sentence.split()
        for word in tokens:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1



# Language embedding
Corpus = LangEmbed('Corpus')
Corpus.addSentence(corpus)
print("Number of Unique Words:")
print("Corpus:", Corpus.n_words)


class LangEmbed_woRareWords:
    def __init__(self, name):
        self.name = name
        self.word2index = {'<PAD>':PAD_token, '<SOS>':SOS_token, '<EOS>':EOS_token, '<UNK>':UNK_token}
        self.word2count = {}
        self.index2word = {PAD_token:'<PAD>', SOS_token:'<SOS>', EOS_token:'<EOS>', UNK_token:'<UNK>'}
        self.n_words = 4

    def addWord(self, Corpus):
        if USE_VOCAB_SIZE:
            Corpus_temp = sorted(Corpus.word2count.items(), key=lambda x: x[1],
                              reverse = True)
                
            # Limit the vocabulary by VOCAB_SIZE
            Corpus_temp = Corpus_temp[0:VOCAB_SIZE]
        
            for (word, count) in Corpus_temp:
                self.word2index[word] = self.n_words
                self.word2count[word] = count
                self.index2word[self.n_words] = word
                self.n_words += 1
        
        else: # Limit the vocabulary by thresholding
            for word in Corpus.word2count:
                n_count = Corpus.word2count[word]
                
                if n_count > UNK_REPLACEMENT:
                    self.word2index[word] = self.n_words
                    self.word2count[word] = n_count
                    self.index2word[self.n_words] = word
                    self.n_words += 1

Corpus_woRareWords = LangEmbed_woRareWords('Corpus')
Corpus_woRareWords.addWord(Corpus)
print("Number of Unique Words:")
print("Corpus_woRareWords:", Corpus_woRareWords.n_words)


# -----------------------------------------------------------------------
# Glove Embedding
if CUDA:
    glove_path = '/ssddata1/data/valance/AI/data/glove.6B/'
else:
    glove_path = '/Users/valancewang/Dropbox/AI/data/glove.6B/'
fname_glove =  'glove.6B.300d.txt' # 'glove.6B.100d.txt'

def load_glove_weights(glove_fpath, embedding_dim, Corpus):
    
    embeddings_index = {}
    
    with open(glove_fpath) as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings_index[word] = vector
    
    print('Found {} word vectors in glove.'.format(len(embeddings_index)))

    embedding_matrix = np.zeros((Corpus.n_words, embedding_dim))

    print('embed_matrix.shape', embedding_matrix.shape)

    found_ct = 0
    for idx in range(Corpus.n_words):
        
        word = Corpus.index2word[idx]
        word = word.lower()
        embedding_vector = embeddings_index.get(word)
        # words not found in embedding index will be all-zeros.
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
            found_ct += 1
    print('{} words are found in glove'.format(found_ct))

    return embedding_matrix

# Parameters
g_weights = load_glove_weights(glove_path + fname_glove, EMBEDDING_DIM, Corpus_woRareWords)
glove_embedding = torch.from_numpy(g_weights).type(torch.FloatTensor)





# -----------------------------------------------------------------------
#%% Load test set - Turk Corpus

if CUDA:
    path = '/ssddata1/data/valance/AI/data/text_simplification/turk/'
else:
    path = '/Users/valancewang/Dropbox/AI/data/text_simplification/turk/'
fname_src = 'test.8turkers.tok.norm'
fname_dst = 'test.8turkers.tok.simp'
fname_ref = ['test.8turkers.tok.turk.0',
             'test.8turkers.tok.turk.1',
             'test.8turkers.tok.turk.2',
             'test.8turkers.tok.turk.3',
             'test.8turkers.tok.turk.4',
             'test.8turkers.tok.turk.5',
             'test.8turkers.tok.turk.6',
             'test.8turkers.tok.turk.7']


# Load the test set
data_test_src = []
data_test_dst = []
data_test_src += open(path + fname_src, "r", encoding="utf-8").read().split('\n')
data_test_dst += open(path + fname_dst, "r", encoding="utf-8").read().split('\n')

data_test_src = [x.lower() for x in data_test_src]
data_test_dst = [x.lower() for x in data_test_dst]
data_test_src = [x.strip("'") for x in data_test_src]
data_test_dst = [x.strip("'") for x in data_test_dst]

# Load the reference set
data_test_ref_0 = []
data_test_ref_1 = []
data_test_ref_2 = []
data_test_ref_3 = []
data_test_ref_4 = []
data_test_ref_5 = []
data_test_ref_6 = []
data_test_ref_7 = []

data_test_ref_0 += open(path + fname_ref[0], "r", encoding="utf-8").read().split('\n')
data_test_ref_1 += open(path + fname_ref[1], "r", encoding="utf-8").read().split('\n')
data_test_ref_2 += open(path + fname_ref[2], "r", encoding="utf-8").read().split('\n')
data_test_ref_3 += open(path + fname_ref[3], "r", encoding="utf-8").read().split('\n')
data_test_ref_4 += open(path + fname_ref[4], "r", encoding="utf-8").read().split('\n')
data_test_ref_5 += open(path + fname_ref[5], "r", encoding="utf-8").read().split('\n')
data_test_ref_6 += open(path + fname_ref[6], "r", encoding="utf-8").read().split('\n')
data_test_ref_7 += open(path + fname_ref[7], "r", encoding="utf-8").read().split('\n')

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



#%%
# -------------------------------
# Embeddings
"""
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
"""



class Embeddings(nn.Module):
    def __init__(self, d_embedding, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_embedding,
                                padding_idx=PAD_token)
        if USE_GLOVE:
            self.lut.weight = nn.Parameter(glove_embedding, requires_grad=True)
            
        if not d_embedding == d_model: 
            self.transform = nn.Linear(d_embedding, d_model)
            self.d_embedding = d_embedding
            self.d_model = d_model
        else:
            self.d_embedding = d_embedding

    def forward(self, x):
        result = self.lut(x) * math.sqrt(self.d_embedding)
        if not EMBEDDING_DIM == HIDDEN_DIM:
            result = self.transform(result)
        return result



# -------------------------------
# Batch
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = \
                self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask










# -------------------------------
# Sentence2index
def sentence2index(lang, sentence):
    tokens = sentence.split()
    index = []
    for tok in tokens:
        if tok in lang.word2index:
            index.append(lang.word2index[tok])
        else:
            index.append(lang.word2index['<UNK>'])
    return index

def indexesFromSentence(lang, sentence):
    indexes = [SOS_token]
    indexes += sentence2index(lang, sentence)
    indexes.append(EOS_token)
    result = torch.LongTensor(indexes)
    return result




#%%

# -------------------------------
# Greedy decode

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    
    model.eval()
    
    memory = model.encode(src, src_mask)
    
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory = memory, src_mask = src_mask, 
                            tgt = Variable(ys), 
                            tgt_mask = Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        if not next_word == EOS_token:
            ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        elif next_word == EOS_token:
            ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            break                
    return ys


# -------------------------------
# 
# Train with iterations
#
# ------------
# Make model - my code
def make_model(src_vocab, tgt_vocab, N=10, d_embedding = 100,
               d_model=512, d_ff=2048, h=8, dropout=0.3):

    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, no_cuda=no_cuda, DEVICE=DEVICE)
    group_attn = GroupAttention(d_model, no_cuda=no_cuda, DEVICE=DEVICE)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), c(group_attn), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        # nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        nn.Sequential(Embeddings(d_embedding, d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_embedding, d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# Make model
model = make_model(Corpus_woRareWords.n_words, 
                   Corpus_woRareWords.n_words,
                   N=N_LAYER, 
                   d_embedding = EMBEDDING_DIM,
                   d_model=HIDDEN_DIM, 
                   d_ff=FF_DIM, 
                   h=N_HEAD, 
                   dropout=DROPOUT)
model.to(DEVICE)
print('Load model.')

# Load model
model_name = "Tree_max_epoch_340_checkpoint.pth"
print(model_name)

if CUDA:
    checkpoint = torch.load(model_path + model_name)
else:
    checkpoint = torch.load(
        model_path + model_name,
        map_location = torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
epoch_checkpoint = checkpoint['epoch']
loss_checkpoint = checkpoint['loss']







# Count number of parameters
tt = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        #print(name)
        ttt = 1
        for s in param.data.size():
            ttt *= s
        tt += ttt
print('total_param_num:',tt)






# -------------------------------
# 
# BLEU and SARI
#
# ------------

# Evaluate
def evaluate(model, sentence, max_length = MAX_LENGTH):
    
    src = indexesFromSentence(Corpus_woRareWords, sentence)     
    src = src.view(1,-1)
    src = src.to(DEVICE)
    
    batch = Batch(src)

    output = greedy_decode(model, 
                           batch.src.long(), batch.src_mask.long(), 
                           MAX_LENGTH, SOS_token)
    output = output[0]
    
    decoded_words = []
    for i in range(0,len(output)):
        index = int(output[i])
        decoded_words.append(Corpus_woRareWords.index2word[index])
        
    return decoded_words


def evaluateRandomly_test(model, n=3):
    
    print("Test set: ")
    
    for i in range(n):
        idx = random.choice(range(len(data_test_src)))
        
        print(idx)
        src = data_test_src[idx]
        dst = data_test_dst[idx]        
        # Replace with <UNK> token
        src = indexesFromSentence(Corpus_woRareWords, src)     
        src_unknown = [Corpus_woRareWords.index2word[int(tok)] for tok in src]
        dst = indexesFromSentence(Corpus_woRareWords, dst) 
        dst_unknown = [Corpus_woRareWords.index2word[int(tok)] for tok in dst]

        print(">", data_test_src[idx])
        print("=", data_test_dst[idx])
        print(">", ' '.join(src_unknown))
        print("=", ' '.join(dst_unknown))
        
        output_words = evaluate(model, data_test_src[idx])
        output_sentence = ' '.join(output_words)
        print("<", output_sentence)
        print("")


def evaluate_idx_test(model, idx):
    
    print("Test set: ")
            
    print(idx)
    src = data_test_src[idx]
    dst = data_test_dst[idx]        
    # Replace with <UNK> token
    src = indexesFromSentence(Corpus_woRareWords, src)     
    src_unknown = [Corpus_woRareWords.index2word[int(tok)] for tok in src]
    dst = indexesFromSentence(Corpus_woRareWords, dst) 
    dst_unknown = [Corpus_woRareWords.index2word[int(tok)] for tok in dst]

    print(">", data_test_src[idx])
    print("=", data_test_dst[idx])
    print(">", ' '.join(src_unknown))
    print("=", ' '.join(dst_unknown))
    
    output_words = evaluate(model, data_test_src[idx])
    output_sentence = ' '.join(output_words)
    print("<", output_sentence)
    print("")


# -------------------------------
# BLEU and SARI

from nltk.translate import bleu_score
from nltk.translate.bleu_score import sentence_bleu
from easse.sari import corpus_sari


def evaluate_all_test(model):
    
    bleu_total = 0 
    sari_total = 0
        
    for idx in range(len(data_test_src)):
        
        
        src = data_test_src[idx]
        dst = data_test_dst[idx]
        ref_0 = data_test_ref_0[idx]
        ref_1 = data_test_ref_1[idx]
        ref_2 = data_test_ref_2[idx]
        ref_3 = data_test_ref_3[idx]
        ref_4 = data_test_ref_4[idx]
        ref_5 = data_test_ref_5[idx]
        ref_6 = data_test_ref_6[idx]
        ref_7 = data_test_ref_7[idx]
        
        # Translate
        output_words = evaluate(model, src)
        
        # Replace with <UNK> token
        src = indexesFromSentence(Corpus_woRareWords, src)     
        src_unknown = [Corpus_woRareWords.index2word[int(tok)] for tok in src]
        dst = indexesFromSentence(Corpus_woRareWords, dst) 
        dst_unknown = [Corpus_woRareWords.index2word[int(tok)] for tok in dst]
        
        ref_0_unk = indexesFromSentence(Corpus_woRareWords, ref_0)
        ref_1_unk = indexesFromSentence(Corpus_woRareWords, ref_1)
        ref_2_unk = indexesFromSentence(Corpus_woRareWords, ref_2)
        ref_3_unk = indexesFromSentence(Corpus_woRareWords, ref_3)
        ref_4_unk = indexesFromSentence(Corpus_woRareWords, ref_4)
        ref_5_unk = indexesFromSentence(Corpus_woRareWords, ref_5)
        ref_6_unk = indexesFromSentence(Corpus_woRareWords, ref_6)
        ref_7_unk = indexesFromSentence(Corpus_woRareWords, ref_7)
        
        ref_0_unk = [Corpus_woRareWords.index2word[int(tok)] for tok in ref_0_unk]
        ref_1_unk = [Corpus_woRareWords.index2word[int(tok)] for tok in ref_1_unk]
        ref_2_unk = [Corpus_woRareWords.index2word[int(tok)] for tok in ref_2_unk]
        ref_3_unk = [Corpus_woRareWords.index2word[int(tok)] for tok in ref_3_unk]
        ref_4_unk = [Corpus_woRareWords.index2word[int(tok)] for tok in ref_4_unk]
        ref_5_unk = [Corpus_woRareWords.index2word[int(tok)] for tok in ref_5_unk]
        ref_6_unk = [Corpus_woRareWords.index2word[int(tok)] for tok in ref_6_unk]
        ref_7_unk = [Corpus_woRareWords.index2word[int(tok)] for tok in ref_7_unk]

        
        # Compute BLEU score
        correct = dst_unknown
        output = output_words                    
        bleu = sentence_bleu([correct], output,
                             smoothing_function=bleu_score.SmoothingFunction().method1)
        bleu_total = bleu_total + bleu * 100

        # Compute SARI score
        output_sent = ' '.join(output)
        src = ' '.join(src_unknown)
        dst = ' '.join(dst_unknown)
        sari = corpus_sari(orig_sents = [src], 
                           sys_sents = [output_sent],
                           refs_sents = [[dst],
                                         [' '.join(ref_0_unk)],
                                         [' '.join(ref_1_unk)],
                                         [' '.join(ref_2_unk)],
                                         [' '.join(ref_3_unk)],
                                         [' '.join(ref_4_unk)],
                                         [' '.join(ref_5_unk)],
                                         [' '.join(ref_6_unk)],
                                         [' '.join(ref_7_unk)]])
        sari_total = sari_total + sari

        # Show progress
        if idx % 10 == 0:
            print('test:', idx)
        
    bleu_total = bleu_total / len(data_test_src)
    sari_total = sari_total / len(data_test_src)

    print('BLEU: ', bleu_total)
    print('SARI: ', sari_total)

    return bleu_total, sari_total
    

evaluateRandomly_test(model, 3)

bleu_total_test, sari_total_test = evaluate_all_test(model)
    
# evaluate_idx_test(model, idx = 351)
