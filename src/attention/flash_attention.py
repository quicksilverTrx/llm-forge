# src/attention/flash_attention.py

import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import logging
from .base_attention import ScaledDotProductAttention
logger = logging.getLogger(__name__)

@dataclass
class FlashAttentionConfig:
    block_size: int = 256
    dropout: float = 0.1
    max_sequence_length: int = 8192
    scale_factor: Optional[float] = None

class FlashAttention(nn.Module):
    def __init__(self,config : FlashAttentionConfig):
        super().__init__()
        self.config = config
        self.scale_factor = config.head_dim ** -0.5
        self.core_attention = ScaledDotProductAttention(
            dropout = config.dropout,
            scale_factor=config.scale_factor)
        self.logger = logging.getLogger(__name__)

    def _process_block(
        self,
        q_block: torch.Tensor,  # Shape: [batch_size, num_heads, block_size_q, head_dim]
        k_block: torch.Tensor,  # Shape: [batch_size, num_heads, block_size_k, head_dim]
        v_block: torch.Tensor,  # Shape: [batch_size, num_heads, block_size_k, head_dim]
        mask_block: Optional[torch.Tensor] = None  # Shape: [batch_size, num_heads, block_size_q, block_size_k]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Process a single attention block using multi-head SDPA."""
        return self.core_attention(
            query=q_block,
            key=k_block,
            value=v_block,
            mask=mask_block
        )
    
    def forward(
        self,
        query: torch.Tensor,  # Shape: [batch_size, num_heads, seq_len, head_dim]
        key: torch.Tensor,    # Shape: [batch_size, num_heads, seq_len, head_dim]
        value: torch.Tensor,  # Shape: [batch_size, num_heads, seq_len, head_dim]
        mask: Optional[torch.Tensor] = None,  # Shape: [batch_size, num_heads, seq_len, seq_len]
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass implementing blocked attention computation while maintaining
        compatibility with multi-head SDPA shapes.
        """
        try:
            batch_size, num_heads, seq_length, head_dim = query.shape
            block_size = min(self.config.block_size, seq_length)
            num_blocks = math.ceil(seq_length / block_size)
            # Initialize output tensors maintaining head dimension
            output = torch.zeros_like(query)  # [batch_size, num_heads, seq_len, head_dim]
            attention_weights = None if not return_weights else torch.zeros(
                batch_size, num_heads, seq_length, seq_length,
                device=query.device, dtype=query.dtype
            )
            # Process attention in blocks
            for i in range(num_blocks):
                q_start = i * block_size
                q_end = min((i + 1) * block_size, seq_length)
                # Extract query block maintaining head dimension
                # Shape: [batch_size, num_heads, block_size_q, head_dim]
                q_block = query[:, :, q_start:q_end]
                # Process key/value blocks
                for j in range(num_blocks):
                    k_start = j * block_size
                    k_end = min((j + 1) * block_size, seq_length)

                    # Extract key/value blocks maintaining head dimension
                    # Shape: [batch_size, num_heads, block_size_k, head_dim]
                    k_block = key[:, :, k_start:k_end]
                    v_block = value[:, :, k_start:k_end]
                    # Extract mask block if exists
                    # Shape: [batch_size, num_heads, block_size_q, block_size_k]
                    mask_block = None
                    if mask is not None:
                        mask_block = mask[:, :, q_start:q_end, k_start:k_end]
                    
                    # Compute block attention
                    # Shape: [batch_size, num_heads, block_size_q, head_dim]
                    block_output, block_weights = self._process_block(
                        q_block, k_block, v_block, mask_block
                    )
                    # Accumulate results maintaining head dimension
                    output[:, :, q_start:q_end] += block_output
                    if return_weights and block_weights is not None:
                        attention_weights[:, :, q_start:q_end, k_start:k_end] = block_weights
            return output, attention_weights
        except Exception as e:
            logger.error(f"Flash attention forward pass error: {str(e)}")
            raise