import torch
from torch import nn
from torch.nn import functional as F

from models.components import *

class Transformer(nn.Module):
    """Transformer Class
    """

    def __init__(
        self, 
        src_vocab_length, 
        target_vocab_length, 
        max_sequence_length, 
        d_model=512,
        d_feedforward=2048,
        dropout=0.1,
        n_encoders=6,
        n_decoders=6, 
        n_head=8
    ):
        super(Transformer, self).__init__()

        self.source_embedding = nn.Embedding(src_vocab_length, d_model)
        self.target_embedding = nn.Embedding(target_vocab_length, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_sequence_length)

        self.transformer_encoder_layer = TransformerEncoderLayer(d_model, n_head, d_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, n_encoders)

        self.transformer_decoder_layer = TransformerDecoderLayer(d_model, n_head, d_feedforward, dropout)
        self.transformer_decoder = TransformerDecoder(self.transformer_decoder_layer, n_decoders)

        self.fc = nn.Linear(d_model, target_vocab_length)

    def forward(self, src, tgt, mask=None):
        """Forward function of transformer

        Args:
            src ((N, S) Tensor): [description]
            tgt ((N, L) Tensor): [description]
            mask ((L, S) Tensor): [description]

        Returns:
            out ((N, L, target_vocab_length)): [description]
        """

        # Apply embedding
        src = self.source_embedding(src)
        tgt = self.target_embedding(tgt)

        # Apply positional encoding
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # Encoder
        src = self.transformer_encoder(src)

        # Decoder
        tgt = self.transformer_decoder(tgt, src, mask)

        # Linear Transformation
        tgt = self.fc(tgt)

        return tgt