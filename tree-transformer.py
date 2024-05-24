#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 15:45:20 2022
@author: valance

Tree Transformer
- Test the original transformer
- Dock the transformer to my task
    - First get the data batch
    - Second add a Decoder to the model
    - Third get the loss compute
- Change to counting epoch
- Add GloVe embedding
- Add validation in training
- Random seed = 100, 929, 1987

"""




# Library
import numpy as np
import torch
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



# Seed
SEED = 1987
torch.manual_seed(SEED)



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
DEVICE = torch.device("cuda:0" if CUDA else "cpu")
# DEVICES = [3,4]   #4
if CUDA:
    no_cuda = False
else:
    no_cuda = True


# Training
OFFSET = 0
N_ITERATION = 80
N_BATCH = 64 

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
# Load validation set
fname_src = 'wiki.full.aner.test.src'
fname_dst = 'wiki.full.aner.test.dst'

# Load the test set
data_test_src = []
data_test_dst = []
data_test_src += open(path + fname_src, "r", encoding="utf-8").read().split('\n')
data_test_dst += open(path + fname_dst, "r", encoding="utf-8").read().split('\n')

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




# -------------------------------
# Embeddings
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
# Train with iterations
#
# Make model - my code
def make_model(src_vocab, tgt_vocab, N=10, d_embedding = 100,
               d_model=512, d_ff=2048, h=8, dropout=0.3):

    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, no_cuda = no_cuda, DEVICE = DEVICE)
    group_attn = GroupAttention(d_model, no_cuda = no_cuda, DEVICE = DEVICE)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), c(group_attn), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
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
# model_name = "Tree_max_epoch_338_checkpoint.pth"
# print(model_name)

# if CUDA:
#     checkpoint = torch.load(model_path + model_name)
# else:
#     checkpoint = torch.load(
#         model_path + model_name,
#         map_location = torch.device('cpu'))
# model.load_state_dict(checkpoint['model_state_dict'])
# epoch_checkpoint = checkpoint['epoch']
# loss_checkpoint = checkpoint['loss']



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

optim = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
# optim.load_state_dict(checkpoint['optimizer_state_dict'])



# -------------------------------
# Greedy decode

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    
    model.eval()
    
    memory = model.encode(src, src_mask)
    
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory=memory, src_mask = src_mask, 
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


def evaluate(model, sentence, max_length = MAX_LENGTH):
    
    src = indexesFromSentence(Corpus_woRareWords, sentence)     
    src = src.view(1,-1)
    src = src.to(DEVICE)
    
    batch = Batch(src)

    output = greedy_decode(model, 
                           batch.src, batch.src_mask.long(), 
                           MAX_LENGTH, SOS_token)
    output = output[0]
    
    decoded_words = []
    for i in range(0,len(output)):
        index = int(output[i])
        decoded_words.append(Corpus_woRareWords.index2word[index])
        
    return decoded_words

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
        
        # Translate
        output_words = evaluate(model, src)
        
        # Replace with <UNK> token
        src = indexesFromSentence(Corpus_woRareWords, src)     
        src_unknown = [Corpus_woRareWords.index2word[int(tok)] for tok in src]

        dst = indexesFromSentence(Corpus_woRareWords, dst) 
        dst_unknown = [Corpus_woRareWords.index2word[int(tok)] for tok in dst]
        
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
                           refs_sents = [[dst]] )
        sari_total = sari_total + sari
        
        
        
        # Write output
        f = open('./tree_wikilarge_seed_' + str(SEED) + '_' + str(epoch) + '_.txt','a')
        f.write(">" + src + "\n")
        f.write("=" + dst + "\n")        
        f.write("<" + ' '.join(output) + "\n")
        f.write('\n')
        f.close()


        # Show progress
        if idx % 10 == 0:
            print('test:', idx)
        
    bleu_total = bleu_total / len(data_test_src)
    sari_total = sari_total / len(data_test_src)

    print('BLEU: ', bleu_total)
    print('SARI: ', sari_total)

    return bleu_total, sari_total



total_loss = []
start = time.time()
total_step_time = 0.
total_masked = 0.
total_token = 0.

step = 0
for epoch in range(OFFSET+1, N_ITERATION+1):
    
    print(epoch)

    model.train()

    for idx_epoch, (input_str, target_str) in enumerate(train_loader):
    
        # My code for generating the batch
        # input_str, target_str = next(iter(train_loader))
            
        n_data_per_batch = len(input_str)
        
        input_index = []
        target_index = []
        
        for idx_per_batch in range(n_data_per_batch):                
    
            idx = idx_per_batch
            
            string_src = str(input_str[idx])
            string_dst = str(target_str[idx])
            index_src = indexesFromSentence(Corpus_woRareWords, string_src)
            index_dst = indexesFromSentence(Corpus_woRareWords, string_dst)
            input_index.append(index_src)
            target_index.append(index_dst)
            
        # Determine max_length
        max_length_src = max([len(x) for x in input_index])    
        max_length_dst = max([len(x) for x in target_index])
            
        # Add padding
        input_tensor = torch.zeros((N_BATCH, max_length_src), dtype=torch.long)
        target_tensor = torch.zeros((N_BATCH, max_length_dst), dtype=torch.long)
        for i in range(n_data_per_batch):
            input_tensor[i, :len(input_index[i])] = input_index[i]
            target_tensor[i, :len(target_index[i])] = target_index[i]
        
        
        input_tensor = input_tensor.to(DEVICE)
        target_tensor = target_tensor.to(DEVICE)
    
        SRC, TGT = input_tensor, target_tensor
    
        src = SRC
        tgt = TGT
    
        src = Variable(src, requires_grad = False) # English
        tgt = Variable(tgt, requires_grad = False) # French        
        
        batch = Batch(src, tgt)
    
        # My code
        step_start = time.time()
    
        output = model.forward(batch.src, batch.tgt,
                               batch.src_mask.long(), batch.tgt_mask.long())
        loss = model.compute_loss(output, batch.tgt_y)
    
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
        optim.step()
    
        total_loss.append(loss.detach().cpu().numpy())
    
        total_step_time += time.time() - step_start
        
        if step % 200 == 1:
            elapsed = time.time() - start
            print("Iteration: %d Epoch Step: %d Loss: %f Total Time: %f Step Time: %f" %
                    (epoch, step, np.mean(total_loss), elapsed, total_step_time))
            model.train()
            print()
            
            # Write log_file
            f = open('./log_file_tree_seed' + str(SEED) + '_80.txt','a')
            f.write("iteration:%s, Epoch Step: %d, average loss:%.4f" % 
                      (epoch, step, np.mean(total_loss) ))
            f.write('\n')
            f.close()

            
            start = time.time()
            total_loss = []
            total_step_time = 0.
            
        step += 1
        
    print('Epoch ends. Saving!!!!')
         
    # model_name = 'Tree_seed_' + str(SEED) + '_epoch_' + str(epoch) + '_checkpoint.pth'
    # torch.save({
    #          'epoch': epoch,
    #          'model_state_dict': model.state_dict(),
    #          'optimizer_state_dict': optim.state_dict(),
    #          'loss': loss}, 
    #          os.path.join('./train_model', model_name)
    #          )
         
    bleu_total_test, sari_total_test = evaluate_all_test(model)
         
    # Write log_file
    f = open('./log_file_tree_seed_' + str(SEED) + '_BLEU_80.txt','a')
    f.write("iteration:%s, Epoch Step: %d, BLEU:%.2f,  SARI:%.2f" % 
            (epoch, idx_epoch, bleu_total_test, sari_total_test ))
    f.write('\n')
    f.close()
         


# print('Training ends. Saving!!!!')

# model_name = 'Tree_seed_' + str(SEED) + '_epoch_' + str(epoch) + '_checkpoint.pth'
# torch.save({
#     'epoch': epoch,
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optim.state_dict(),
#     'loss': loss}, 
#     os.path.join('./train_model', model_name)
#     )        
        












        
        
        
        
