
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

         # Matrix multiply query and key^T: 
        # => [batch_size, num_heads, seq_len, head_dim] x [batch_size, num_heads, head_dim, seq_len]
        # => [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(query.size(-1))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax over last dimension: => [batch_size, num_heads, seq_len, seq_len]
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum: 
        # => [batch_size, num_heads, seq_len, seq_len] x [batch_size, num_heads, seq_len, head_dim]
        # => [batch_size, num_heads, seq_len, head_dim]
        output = torch.matmul(attention_weights, value)
        return output, attention_weights
class VanillaMultiHeadAttention(nn.Module):
    """
    Standard multi-head attention implementation for baseline comparison.
    No optimizations applied to provide clear baseline metrics.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)  # => [d_model, d_model]
        self.k_proj = nn.Linear(d_model, d_model, bias=bias) # => [d_model, d_model]
        self.v_proj = nn.Linear(d_model, d_model, bias=bias) # => [d_model, d_model]
        self.output_proj = nn.Linear(d_model, d_model, bias=bias)
        
        self.attention = VanillaAttention(dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def _split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Split heads and reshape."""
        # x initially: [batch_size, seq_len, d_model]
        # reshape => [batch_size, seq_len, num_heads, head_dim]
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)  # => [B, S, h, d_k]
        # transpose => [batch_size, num_heads, seq_len, head_dim]
        return x.transpose(1, 2) # => [B, h, S, d_k]
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with standard implementation.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Optional mask tensor
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Tuple of (output tensor, optional attention weights)
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Linear projections => each becomes [batch_size, seq_len, d_model]

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        #  # Split into heads => [batch_size, num_heads, seq_len, head_dim]
        query = self._split_heads(query, batch_size)
        key = self._split_heads(key, batch_size)
        value = self._split_heads(value, batch_size)
        

        # Compute attention => 
        #   query/key/value each: [batch_size, num_heads, seq_len, head_dim]
        attention_output, attention_weights = self.attention(
            query=query,
            key=key,
            value=value,
            mask=attention_mask
        )
        
        # Combine heads
        # => from [batch_size, num_heads, seq_len, head_dim] 
        #    to   [batch_size, seq_len, num_heads, head_dim]
        attention_output = attention_output.transpose(1, 2).contiguous()

        # => [batch_size, seq_len, d_model]
        attention_output = attention_output.view(batch_size, seq_len, self.d_model)
        
        # Output projection  => [batch_size, seq_len, d_model]
        output = self.output_proj(attention_output)
        
        if return_attention_weights:
            return output, attention_weights
        return output, None
