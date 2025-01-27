#src/attention/multi_head.py
import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging
import math
from .base_attention import ScaledDotProductAttention
from .flash_attention import FlashAttentionConfig,FlashAttention

logger = logging.getLogger(__name__)
class MultiHeadAttention(nn.Module):
    """
    Optimized multi-head attention implementation with parallel computation.
    
    Attributes:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        head_dim (int): Dimension of each attention head
        dropout (float): Dropout probability
    """

    def __init__( self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        use_flash: bool = False):

        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = float(self.head_dim) ** -0.5

        self.qkv_proj = nn.Linear(d_model, d_model*3, bias=bias)
        self.output_proj = nn.Linear(d_model, d_model, bias=bias)
        if use_flash:
            self.attention = FlashAttention(FlashAttentionConfig(dropout=dropout))
        else:
            self.attention = ScaledDotProductAttention(dropout=dropout, scale_factor=self.scaling)

        self._reset_parameters()



    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight,gain=1/math.sqrt(2))
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    
    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine multiple heads back into original dimension.
        
        Args:
            x: Input tensor of shape [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            Tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, _, seq_len, _ = x.size()
        return (x
                .transpose(1, 2)
                .contiguous()
                .view(batch_size, seq_len, self.d_model))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optimized parallel computation across heads.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Optional mask tensor
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Tuple of (output tensor, optional attention weights)
        """
        try:
            print( hidden_states.shape)
            batch_size, seq_len, _ = hidden_states.shape

            qkv = self.qkv_proj(hidden_states)
            # Split Q, K, V and heads in parallel
            qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq_len, head_dim]
            query, key, value = qkv[0], qkv[1], qkv[2]
            attention_output, attention_weights = self.attention(
                query=query,
                key=key,
                value=value,
                mask=attention_mask
            )
            attention_output = self._combine_heads(attention_output)
            output = self.output_proj(attention_output)
            return output, attention_weights
        except Exception as e:
            logger.error(f"Error in multi-head attention forward pass: {str(e)}")
            raise

