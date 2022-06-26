# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:49:16 2022

@author: njucx
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import math
from data_process import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda')

#%% Transformer

class Gating(nn.Module):
    def __init__(self, d_model, m): # 128,18
        super().__init__()
        self.m = m

        # the reset gate r_i
        self.W_r = nn.Parameter(torch.Tensor(m, m))
        self.V_r = nn.Parameter(torch.Tensor(m, m))
        self.b_r = nn.Parameter(torch.Tensor(m))

        # the update gate u_i
        self.W_u = nn.Parameter(torch.Tensor(m, m))
        self.V_u = nn.Parameter(torch.Tensor(m, m))
        self.b_u = nn.Parameter(torch.Tensor(m))

        # the output
        self.W_e = nn.Parameter(torch.Tensor(m, d_model))
        self.b_e = nn.Parameter(torch.Tensor(d_model))

        self.init_weights()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3, 1), stride=1), 
        )
        

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.m)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x_i = x[:, :, 1:2, :].to(device) #only applying the gating on the current row even with the stack of 3 rows cames as input (1,1,3,14)
        h_i = self.cnn_layers(x).to(device)  # shape becomes 1,1,1,14 as the nn.conv2d has output channel as 1 but the convolution is applied on whole past input (stack of three)

        r_i = torch.sigmoid(torch.matmul(h_i, self.W_r) + torch.matmul(x_i, self.V_r) + self.b_r)
        u_i = torch.sigmoid(torch.matmul(h_i, self.W_u) + torch.matmul(x_i, self.V_u) + self.b_u)

        # the output of the gating mechanism
        hh_i = torch.mul(h_i, u_i) + torch.mul(x_i, r_i)

        return torch.matmul(hh_i, self.W_e) + self.b_e # (the final output is 1,1,1,128 as the encoder has size of 128.)
        
        
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        print('postion', position.size())
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) #[max_len, 1, d_model]
        print('pe', pe.size())
        self.register_buffer('pe', pe)


    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> x = torch.randn(50, 128, 18)
            >>> pos_encoder = PositionalEncoding(18, 0.1)
            >>> output = pos_encoder(x)
        """
        
        x = x + self.pe[:x.size(0), :]
        print(x.size())
        return self.dropout(x)                     


class Transformer(nn.Module):
    r"""
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        nlayers: the number of sublayers of both encoder and decoder
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    Examples::
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)
    """

    def __init__(self, d_model, nhead, dim_feedforward, nlayers, dropout) -> None:
        super(Transformer, self).__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers, encoder_norm)
        
        # decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, nlayers, decoder_norm)

        self.d_model = d_model
        self.nhead = nhead
    
    def init_weights(self):
        initrange = 0.1
        nn.init.xavier_uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, tgt, src_mask) -> Tensor:
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :(S, N, E)
            - tgt: :(T, N, E)
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :(N, S)
            - tgt_key_padding_mask: :(N, T)
            - memory_key_padding_mask: :(N, S)

            - output: :(T, N, E)

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.

            S is the source sequence length, 
            T is the target sequence length, 
            N is the batch size, 
            E is the feature number
        """

        is_batched = src.dim() == 3
        if src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask)
        output = self.decoder(tgt, memory)
        return output


"""
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory) -> Tensor:

        x = tgt
        x = self.norm1(x + self._sa_block(x))
        x = self.norm2(x + self._mha_block(x, memory))
        x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x) -> Tensor:
        x = self.self_attn(x, x, x, need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x, mem) -> Tensor:
        x = self.multihead_attn(x, mem, mem, need_weights=False)[0]
        print('mha_men', men)
        return self.dropout2(x)
    
    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

"""