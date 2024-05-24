import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import *
from torch.nn import CrossEntropyLoss


"""    
class Encoder(nn.Module):
    def __init__(self, layer, N, d_model, vocab_size, word_embed):
        super(Encoder, self).__init__()
        self.word_embed = word_embed
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, inputs, mask):
        break_probs = []
        x = self.word_embed(inputs)
        group_prob = 0.
        for layer in self.layers:
            x, group_prob, break_prob = layer(x, mask,group_prob)
            break_probs.append(break_prob)

        x = self.norm(x)
        break_probs = torch.stack(break_probs, dim=1)
        return self.proj(x),break_probs


    def masked_lm_loss(self, out, y):
        fn = CrossEntropyLoss(ignore_index=-1)
        return fn(out.view(-1, out.size()[-1]), y.view(-1))


    def next_sentence_loss(self):
        pass
"""

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, group_attn, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.group_attn = group_attn
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask, group_prob):
        # What's break_prob?
        group_prob, break_prob = self.group_attn(x, mask, group_prob)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, group_prob, mask))
        return self.sublayer[1](x, self.feed_forward), group_prob, break_prob
    




# My code
# Encoder Decoder
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        
        output = self.encode(src, src_mask)
                
        output = self.decode(tgt, output, src_mask, tgt_mask)
        
        return output
    
    def encode(self, src, src_mask):
        output, group_prob = self.encoder(self.src_embed(src), src_mask)
        return output
    
    def decode(self, tgt, memory, src_mask, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
    def compute_loss(self, output, y):
        output = self.generator(output)
        fn = CrossEntropyLoss(ignore_index = 0) # PAD_token
        output = output.view(-1, output.size()[-1])
        y = y.reshape(-1)
        loss = fn(output, y)
        return loss
    


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)



# Encoder
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        group_prob = 0.
        for layer in self.layers:
            x, group_prob, break_prob = layer(x, mask, group_prob)
        return self.norm(x), group_prob




# Decoder
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)




    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        # group_prob, break_prob = self.group_attn(x, tgt_mask, group_prob)
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask = tgt_mask, group_prob = None))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, mask = src_mask, group_prob = None))
        return self.sublayer[2](x, self.feed_forward)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        