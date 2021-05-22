import torch
from torch import nn
from torch.nn import functional as F

def scaled_dot_product_attention(Q, K: torch.Tensor, V, Dk, mask=None):
    """Scaled Dot Product Attention Function

    Args:
        Q ((N, L, Dk) Tensor): Query Tensor
        K ((N, S, Dk) Tensor): Keys
        V ((N, S, Dv) Tensor): Values
        Dk (number): Dimension of Keys and Queries
        mask ((L, S) Tensor): 2d Mask; 

    Returns:
        Output ((N, L, Dv) Tensor): Scaled Dot Product Value Matrix
    """

    dot = torch.matmul(Q, K.transpose(1, 2)) / (Dk ** 0.5)

    if mask != None:
        dot = dot + mask

    return torch.matmul(F.softmax(dot), V)

class MultiHeadAttention(nn.Module):
    """Class to perform MultiHeadAttention.

    Instead of performing a singular attention function with Dmodel dimension
        queries, keys, and values, we linearly project them h times with 
        different linear projections to dk, dk, and dv dimensions respectively.
    """

    def __init__(self, h, Dk, Dv, dk, dv):
        """Initialize the MultiHeadAttention model with the given parameters.

        Args:
            h (number): Number of parallel attention layers, or heads
            Dk (number): Original key dimension
            Dv (number): Original value dimension
            dk (number): Projected key dimension
            dv (number): Projected value dimension
        """
        super(MultiHeadAttention, self).__init__()

        # Set variables
        self.h = h
        self.Dk = Dk
        self.Dv = Dv
        self.dk = dk
        self.dv = dv

        # Linear projections
        self.Lq = [nn.Linear(self.Dk, self.dk, bias=False) for i in range(self.h)]
        self.Lk = [nn.Linear(self.Dk, self.dk, bias=False) for i in range(self.h)]
        self.Lv = [nn.Linear(self.Dv, self.dv, bias=False) for i in range(self.h)]
        
        # Output projection
        self.Lo = nn.Linear(self.h * self.dv, self.Dv, bias=False) 

    def forward(self, Q, K, V, mask=None):
        """Compute the MultiHeadAttention

        Args:
            Q ((N, L, Dk) Tensor): Query Tensor
            K ((N, S, Dk) Tensor): Keys
            V ((N, S, Dv) Tensor): Values
        """

        # Linear projections
        lq = [self.Lq[i](Q) for i in range(self.h)]
        lk = [self.Lq[i](K) for i in range(self.h)]
        lv = [self.Lq[i](V) for i in range(self.h)]

        # Attention
        heads = torch.cat(
            [scaled_dot_product_attention(q, k, v, self.dk, mask) for q, k, v in zip(lq, lk, lv)],
            dim=-1
        )

        # Output projection
        out = self.Lo(heads)

        return out