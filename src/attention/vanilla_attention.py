
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class VanillaAttention(nn.Module):
    """
    Standard vanilla attention implementation for baseline comparison.
    No optimizations applied to provide clear baseline metrics.
    """
    def __init__(self,dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None) -> Tuple [torch.Tensor,torch.Tensor]:
        """
        Basic attention computation without optimizations.
        
        Args:
            query: [batch_size, num_heads, seq_len, head_dim]
            key: [batch_size, num_heads, seq_len, head_dim]
            value: [batch_size, num_heads, seq_len, head_dim]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output tensor, attention weights)
        """
        