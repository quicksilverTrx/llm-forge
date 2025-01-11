# src/attention/masking.py

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
import logging
from .vanilla_attention import VanillaMultiHeadAttention
from .base_attention import ScaledDotProductAttention
from .multi_head import MultiHeadAttention
logger = logging.getLogger(__name__)

class AttentionMaskGenerator:
    """
    Generates optimized attention masks for different attention patterns.
    Compatible with both VanillaMultiHeadAttention and ScaledDotProductAttention.
    """
    
    @staticmethod
    def create_causal_mask(
        seq_len: int,
        batch_size: int,
        num_heads: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create causal (autoregressive) attention mask.
        
        Args:
            seq_len: Sequence length
            batch_size: Batch size
            num_heads: Number of attention heads
            device: Target device
            
        Returns:
            Causal mask tensor
        """
        # Create efficient causal mask using minimal memory
        # Create a mask for causal (autoregressive) attention
        # Prevents attending to future tokens in the sequence.
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )
        # Expand for batch size and heads while sharing underlying memory
        return mask.expand(batch_size, num_heads, seq_len, seq_len)

    @staticmethod
    def create_padding_mask(
        seq_lens: torch.Tensor,
        max_len: int,
        batch_size: int,
        num_heads: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create padding mask for variable sequence lengths.
        
        Args:
            seq_lens: Tensor of sequence lengths
            max_len: Maximum sequence length
            batch_size: Batch size
            num_heads: Number of attention heads
            device: Target device
            
        Returns:
            Padding mask tensor
        """
        # Create efficient padding mask
        # Create a padding mask for variable sequence lengths
        # Marks positions beyond `seq_lens` as invalid.
        mask = torch.arange(max_len, device=device)[None, :] >= seq_lens[:, None]
        # Expand for heads while sharing memory
        return mask[:, None, None, :].expand(batch_size, num_heads, max_len, max_len)

    @staticmethod
    def create_local_mask(
        seq_len: int,
        window_size: int,
        batch_size: int,
        num_heads: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create local attention mask with specified window size.
        
        Args:
            seq_len: Sequence length
            window_size: Local attention window size
            batch_size: Batch size
            num_heads: Number of attention heads
            device: Target device
            
        Returns:
            Local attention mask tensor
        """
        # Compute positions relative to each token
        # Tokens can only attend to a limited range of neighboring tokens.
        positions = torch.arange(seq_len, device=device)
        distances = positions[None, :] - positions[:, None]
        
        # Create window mask efficiently
        mask = torch.abs(distances) > window_size
        # Convert boolean mask to float and assign large negative values for masked positions
        mask = mask.float() * float('-inf')
        
        return mask.expand(batch_size, num_heads, seq_len, seq_len)

class MaskedAttention(nn.Module):
    """
    Attention module with advanced masking capabilities.
    """
    def __init__(
        self,
        base_attention: Union[VanillaMultiHeadAttention, ScaledDotProductAttention,MultiHeadAttention],
        mask_type: str = 'causal',
        window_size: int = 256
    ):
        super().__init__()
        # Initialize with a base attention module and mask configuration
        self.base_attention = base_attention  # Underlying attention module
        self.mask_type = mask_type  # Type of mask to generate (causal, local, or padding)
        self.window_size = window_size  # Used for local attention
        self.mask_generator = AttentionMaskGenerator()  # Utility for generating masks
        self._static_mask_cache = {}  # Cache for static masks (e.g., causal or local masks)

        
    def _get_mask(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        device: torch.device,
        seq_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get appropriate attention mask based on configuration.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            num_heads: Number of attention heads
            device: Target device
            seq_lens: Optional sequence lengths for padding mask
            
        Returns:
            Attention mask tensor
        """
        # Retrieve or generate an attention mask based on the current configuration
        cache_key = (self.mask_type, batch_size, seq_len, num_heads)
        
        # Return cached mask if available and not using variable sequence lengths
        if seq_lens is None and cache_key in self._static_mask_cache:
            return self._static_mask_cache[cache_key]
        
        # Generate appropriate mask
        if self.mask_type == 'causal':
            mask = self.mask_generator.create_causal_mask(
                seq_len, batch_size, num_heads, device
            )
        elif self.mask_type == 'local':
            mask = self.mask_generator.create_local_mask(
                seq_len, self.window_size, batch_size, num_heads, device
            )
        elif self.mask_type == 'padding' and seq_lens is not None:
            mask = self.mask_generator.create_padding_mask(
                seq_lens, seq_len, batch_size, num_heads, device
            )
        else:
            raise ValueError(f"Unsupported mask type: {self.mask_type}")
            
        # Cache static masks
        if seq_lens is None:
            self._static_mask_cache[cache_key] = mask
            
        return mask
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
        additional_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with automatic mask generation and application.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, d_model]
            seq_lens: Optional sequence lengths for padding mask
            additional_mask: Optional additional mask to combine
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Tuple of (output tensor, optional attention weights)
        """
        try:
            batch_size, seq_len, _ = hidden_states.shape
            device = hidden_states.device
            
            # Get base mask
            mask = self._get_mask(
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=self.base_attention.num_heads,
                device=device,
                seq_lens=seq_lens
            )
            
            # Combine with additional mask if provided
            # Retrieve the appropriate mask for the current configuration
            if additional_mask is not None:
                mask = mask + additional_mask
            
            # Apply attention with mask
            output, weights = self.base_attention(
                hidden_states=hidden_states,
                attention_mask=mask,
                return_attention_weights=return_attention_weights
            )
            
            return output, weights
            
        except Exception as e:
            logger.error(f"Error in masked attention forward pass: {str(e)}")
            raise
            
    def clear_cache(self):
        """Clear the mask cache"""
        self._static_mask_cache.clear()