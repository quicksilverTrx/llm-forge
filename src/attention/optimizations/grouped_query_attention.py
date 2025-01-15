# src/attention/optimizations/grouped_query_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
from ..multi_head import MultiHeadAttention
from ..base_attention import ScaledDotProductAttention
from ..vanilla_attention import VanillaAttention

@dataclass
class GQAConfig:
    num_query_heads: int = 8
    num_kv_heads: int = 2  # Fewer KV heads than query heads
    head_dim: int = 64
    dropout: float = 0.1
    max_seq_length: int = 8192
    use_bias: bool = False

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention implementation using scaled dot-product attention.
    Reduces memory usage by sharing key and value heads across multiple query heads.
    """
    def __init__(self, config: GQAConfig):
        super().__init__()
        self.config = config
        
        # Validate configuration
        if self.config.num_query_heads % self.config.num_kv_heads != 0:
            raise ValueError("Number of query heads must be divisible by number of KV heads")
        
        # Model dimensions
        self.hidden_size = config.num_query_heads * config.head_dim
        self.num_query_heads = config.num_query_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_groups = self.num_query_heads // self.num_kv_heads
        self.dropout_p = config.dropout
        
        # Add scaling factor
        self.scale = self.head_dim ** -0.5
        
        # Initialize projections
        self.q_proj = nn.Linear(
            self.hidden_size, 
            self.head_dim * self.num_query_heads, 
            bias=config.use_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, 
            self.head_dim * self.num_kv_heads, 
            bias=config.use_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, 
            self.head_dim * self.num_kv_heads, 
            
        )
        self.out_proj = nn.Linear(
            self.head_dim * self.num_query_heads, 
            self.hidden_size, 
            bias=config.use_bias
        )
        
        # Initialize attention
        self.attention = ScaledDotProductAttention(
            dropout=self.dropout_p,
            
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        # Initialize with small standard deviation
        std = 0.02
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=std)
        
        if self.config.use_bias:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

    def _split_heads(
        self,
        x: torch.Tensor,
        num_heads: int,
        head_dim: Optional[int] = None
    ) -> torch.Tensor:
        """Split tensor into heads while preserving information flow"""
        head_dim = head_dim or self.head_dim
        batch_size, seq_length, _ = x.shape
        
        x = x.view(batch_size, seq_length, num_heads, head_dim)
        return x.transpose(1, 2)  # (batch, heads, seq_length, head_dim)

    def _repeat_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """
        Repeat key/value heads to match number of query heads.
        Uses more efficient repeat_interleave operation.
        """
        return kv.repeat_interleave(self.num_groups, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  
        use_cache: bool = False, 
        head_mask: Optional[torch.Tensor] = None, 
        output_attentions: bool = False,  
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for Grouped Query Attention
        
        Args:
            hidden_states: Input tensor [batch_size, seq_length, hidden_size]
            attention_mask: Optional mask [batch_size, seq_length]
            causal: Whether to apply causal masking
            layer_past: Optional past key/values for incremental decoding
            use_cache: Whether to return key/values for incremental decoding
            head_mask: Optional mask for heads
            output_attentions: Whether to return attention weights
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project queries, keys, and values
        queries = self.q_proj(hidden_states)  # [batch, seq, num_q_heads * head_dim]
        keys = self.k_proj(hidden_states)     # [batch, seq, num_kv_heads * head_dim]
        values = self.v_proj(hidden_states)   # [batch, seq, num_kv_heads * head_dim]

        # Handle layer_past if provided
        if layer_past is not None:
            past_key, past_value = layer_past
            keys = torch.cat((past_key, keys), dim=1)
            values = torch.cat((past_value, values), dim=1)
        
        # Split heads
        queries = self._split_heads(queries, self.num_query_heads)     # [batch, num_q_heads, seq, head_dim]
        keys = self._split_heads(keys, self.num_kv_heads)             # [batch, num_kv_heads, seq, head_dim]
        values = self._split_heads(values, self.num_kv_heads)         # [batch, num_kv_heads, seq, head_dim]
        
        # Scale queries
        queries = queries * self.scale
        
        # Repeat KV heads
        keys = self._repeat_kv(keys)       # [batch, num_q_heads, seq, head_dim]
        values = self._repeat_kv(values)   # [batch, num_q_heads, seq, head_dim]
        
        # Prepare present key/values for caching if needed
        present = (keys, values) if use_cache else None
        
        # Apply attention
        attn_output,_ = self.attention(
            queries,
            keys,
            values
            # mask=attention_mask,
            # causal=causal,
            # head_mask=head_mask
        )
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, -1)
        
        # Final projection and dropout
        output = self.dropout(self.out_proj(attn_output))
        
        return output, None, present  # output, attention weights, present key/values