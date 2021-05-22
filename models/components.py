import math
import copy

import torch
from torch import nn
from torch.nn import functional as F

from models.attention import MultiHeadAttention

class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer
    """

    def __init__(self, d_model, n_head, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        # Attention Layer
        self.self_attention = MultiHeadAttention(n_head, d_model, d_model, d_model // n_head, d_model // n_head)

        # Attention Layer Norm
        self.attention_layer_norm = nn.LayerNorm(d_model)

        # Attention Dropout
        self.attention_dropout = nn.Dropout(dropout)

        # Feedforward Layer
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        # Feedforward Layer Norm
        self.ff_layer_norm = nn.LayerNorm(d_model)

        # Feedforward Dropout
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, src):
        # Self Attention
        src2 = self.self_attention(src, src, src)

        # Dropout 1
        src = src + self.attention_dropout(src2)
        
        # Layer Normalization
        src = self.attention_layer_norm(src)

        # Feed Forward
        src2 = self.ff(src)

        # Dropout 2
        src = src + self.ff_dropout(src2)

        # FF Layer Norm
        src = self.ff_layer_norm(src)

        return src

class TransformerEncoder(nn.Module):
    """Transformer Encoder consisting of N TransformerEncoderLayers
    """

    def __init__(self, encoder_layer, n_layers):
        super(TransformerEncoder, self).__init__()

        self.layers = _get_clones(encoder_layer, n_layers)

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)

        return src

class TransformerDecoderLayer(nn.Module):
    """Single Transformer Decoder Layer
    """

    def __init__(self, d_model, n_head, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        # Self Attention Layer
        self.self_attention = MultiHeadAttention(n_head, d_model, d_model, d_model // n_head, d_model // n_head)

        # Self Attention Layer Norm
        self.self_attention_layer_norm = nn.LayerNorm(d_model)

        # Self Attention Dropout
        self.self_attention_dropout = nn.Dropout(dropout)

        # Multihead Attention Layer
        self.multi_head_attention = MultiHeadAttention(n_head, d_model, d_model, d_model // n_head, d_model // n_head)

        # Self Attention Layer Norm
        self.multi_head_attention_layer_norm = nn.LayerNorm(d_model)

        # Self Attention Dropout
        self.multi_head_attention_dropout = nn.Dropout(dropout)

        # Feedforward Layer
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        # Feedforward Layer Norm
        self.ff_layer_norm = nn.LayerNorm(d_model)

        # Feedforward Dropout
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, tgt, encoded_src, mask=None):
        # Self Attention
        tgt2 = self.self_attention(tgt, tgt, tgt)

        # Dropout 1
        tgt = tgt + self.self_attention_dropout(tgt2)
        
        # Layer Normalization
        tgt = self.self_attention_layer_norm(tgt)

        # Multihead Attention
        tgt2 = self.multi_head_attention(tgt, encoded_src, encoded_src, mask)

        # Dropout 2
        tgt = tgt + self.multi_head_attention_dropout(tgt2)

        # Multihead Later Norm
        tgt = self.multi_head_attention_layer_norm(tgt)

        # Feed Forward
        tgt2 = self.ff(tgt)

        # Dropout 3
        tgt = tgt + self.ff_dropout(tgt2)

        # FF Layer Norm
        tgt = self.ff_layer_norm(tgt)

        return tgt

class TransformerDecoder(nn.Module):
    """Transformer Decoder consisting of N TransformerDecoderLayers
    """

    def __init__(self, decoder_layer, n_layers):
        super(TransformerDecoder, self).__init__()

        self.layers = _get_clones(decoder_layer, n_layers)

    def forward(self, tgt, encoded_src, mask=None):
        for layer in self.layers:
            tgt = layer(tgt, encoded_src, mask)

        return tgt

class PositionalEncoding(nn.Module):
    """Positional Encoding Class
    """

    def __init__(self, d_model, max_length=512):
        super(PositionalEncoding, self).__init__()

        self.matrix = torch.Tensor([
            [(j & 1) * (math.pi / 2) + -((j & 1) * 2 - 1) * i / pow(10000, (j - (j & 1)) / d_model) for j in range(d_model)] for i in range(max_length)
        ])

        self.matrix = torch.sin(self.matrix)

    def forward(self, src):
       """Forward function to perform positional encoding

       Args:
           src ((N, S, D_model) Tensor): Source string to encode
       """
       return src * self.matrix[:src.size()[1]]

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])