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
        max_sequence_length: int = 2048,
        causal = False
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.causal = causal
        self.scale_factor = scale_factor
        self.max_sequence_length = max_sequence_length

        # Pre-allocate attention masks for different sequence lengths
        self._cached_masks = {}

    

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
         # Compute scaling factor if not provided
        if self.scale_factor is None:
            self.scale_factor = 1.0 / math.sqrt(query.size(-1))
        try:
            # Optimized matrix multiplication with automatic mixed precision
            scores = torch.matmul(query, key.transpose(-2, -1))
            scores = scores * self.scale_factor

            if mask is not None:
                    scores = scores.masked_fill(mask == 0, -1e9)

            return scores
        except RuntimeError as e:
            logger.error(f"Error in attention score computation: {str(e)}")
            raise

    def _apply_attention(self,
        scores: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
         """
        Apply attention scores to values with optimized memory usage.
        
        Args:
            scores: Attention scores tensor
            value: Value tensor
            
        Returns:
            Output tensor"""
         try:
            # Optimized matrix multiplication with automatic mixed precision
            # Apply softmax and dropout
            attn_probs = F.softmax(scores, dim=-1)
            attn_probs = self.dropout(attn_probs)

            output = torch.matmul(attn_probs, value)

            return output     
         except RuntimeError as e:
             logger.error(f"Error in attention application: {str(e)}")
             raise

    def forward(self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for scaled dot-product attention.
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len, head_dim]
            key: Key tensor [batch_size, num_heads, seq_len, head_dim]
            value: Value tensor [batch_size, num_heads, seq_len, head_dim]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output tensor, attention probabilities)
        """
        try:
            scores = self._compute_attention_scores(query, key, mask)
            output = self._apply_attention(scores, value)
            return output, scores
        except RuntimeError as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise
    



       
         
