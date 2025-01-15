# src/attention/positional/rope.py

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class RotaryPositionEmbeddings(nn.Module):
    """
    Implements Rotary Position Embeddings (RoPE).
    """
    def __init__(
        self,
        dim: int,
        base: int = 10000,
        scale: float = 1.0,
        max_seq_length: int = 8192
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.scale = scale
        self.max_seq_length = max_seq_length
        
        # Initialize cached sin/cos values
        self.register_buffer(
            "inv_freq",
            1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        )
        self._seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None
        
    def _update_cos_sin_cache(self, seq_length: int, device: torch.device):
        """Update cached cos/sin values if needed."""
        if seq_length > self._seq_len_cached:
            self._seq_len_cached = seq_length
            t = torch.arange(seq_length, device=device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(device)
            
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
            
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply rotary position embeddings to query and key tensors.
        
        Args:
            query: [..., seq_len, dim]
            key: [..., seq_len, dim]
            value: Optional [..., seq_len, dim]
            
        Returns:
            Tuple of (rotated query, rotated key, value)
        """
        seq_length = query.shape[-2]
        device = query.device
        
        self._update_cos_sin_cache(seq_length, device)
        
        # Reshape for rotation
        query_rot = query.reshape(*query.shape[:-1], -1, 2)
        key_rot = key.reshape(*key.shape[:-1], -1, 2)
        
        # Apply rotation using complex multiplication
        query_rot1 = torch.stack([-query_rot[..., 1], query_rot[..., 0]], dim=-1)
        key_rot1 = torch.stack([-key_rot[..., 1], key_rot[..., 0]], dim=-1)
        
        # Apply cached cos/sin
        query_embed = (query_rot * self.cos_cached[:, :, :seq_length, :] + 
                      query_rot1 * self.sin_cached[:, :, :seq_length, :])
        key_embed = (key_rot * self.cos_cached[:, :, :seq_length, :] + 
                    key_rot1 * self.sin_cached[:, :, :seq_length, :])
        
        # Reshape back
        query_embed = query_embed.flatten(-2)
        key_embed = key_embed.flatten(-2)
        
        return query_embed, key_embed, value