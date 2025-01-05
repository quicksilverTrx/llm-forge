import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional,Tuple
import logging

logger = logging.getLogger(__name__)

class ScaledDotProductAttention(nn.Module):
    """
    Optimized implementation of scaled dot-product attention.
    
    Attributes:
        dropout (float): Attention dropout rate
        scale_factor (float): Scaling factor for attention scores
        max_sequence_length (int): Maximum sequence length for static memory allocation
    """
    def __init__(
        self,
        dropout: float = 0.1,
        scale_factor: Optional[float] = None,
        max_sequence_length: int = 2048
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale_factor = scale_factor
        self.max_sequence_length = max_sequence_length

        # Pre-allocate attention masks for different sequence lengths
        self._cached_masks = {}

        
        # Performance monitoring
        self.profile_data = {
            "attention_scores_mean": [],
            "attention_scores_std": [],
            "memory_usage": []
        }

    def _compute_attention_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:        
        """
        Compute attention scores with memory-efficient implementation.
        
        Args:
            query: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
            key: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
            mask: Optional attention mask
            
        Returns:
            Attention scores tensor
        """