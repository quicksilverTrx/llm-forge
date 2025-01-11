# src/attention/patterns/sliding_window.py

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
import logging
from ..vanilla_attention import VanillaMultiHeadAttention
from ..base_attention import ScaledDotProductAttention
from ..multi_head import MultiHeadAttention
from ..masking import AttentionMaskGenerator

logger = logging.getLogger(__name__)

class SlidingWindowAttention(nn.Module):
    """
    Implements sliding window attention pattern with efficient memory usage.
    Compatible with both VanillaMultiHeadAttention and ScaledDotProductAttention.
    
    Features:
    - Configurable window size
    - Strided window implementation
    - Optional global tokens
    - Memory-efficient implementation
    """
    
    def __init__(
        self,
        base_attention: Union[VanillaMultiHeadAttention, ScaledDotProductAttention,MultiHeadAttention],
        window_size: int = 256,
        stride: int = None,
        num_global_tokens: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.base_attention = base_attention
        self.window_size = window_size
        self.stride = stride or window_size
        self.num_global_tokens = num_global_tokens
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize mask generator
        self.mask_generator = AttentionMaskGenerator()
        
        
    def _create_sliding_windows(
        self,
        x: torch.Tensor,
        batch_size: int,
        seq_len: int,
        dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Create sliding windows over the sequence.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            batch_size: Batch size
            seq_len: Sequence length
            dim: Hidden dimension
            
        Returns:
            Tuple of (windowed tensor, indices tensor, number of windows)
        """
        # Handle global tokens if present
        global_tokens = None
        if self.num_global_tokens > 0:
            global_tokens = x[:, :self.num_global_tokens]
            x = x[:, self.num_global_tokens:]
            seq_len -= self.num_global_tokens
        
        # Calculate number of windows
        num_windows = (seq_len - self.window_size) // self.stride + 1
        
        # Create indices for each window
        indices = torch.arange(
            0, num_windows * self.stride, self.stride,
            device=x.device
        )[:, None] + torch.arange(
            0, self.window_size,
            device=x.device
        )[None, :]
        
        # Gather windows using indices
        windows = x.gather(
            1,
            indices[:, :, None].expand(-1, -1, dim).repeat(batch_size, 1, 1)
        )
        
        # Reshape to [batch_size * num_windows, window_size, dim]
        windows = windows.view(batch_size * num_windows, self.window_size, dim)
        
        return windows, indices, num_windows
        
    def _merge_windows(
        self,
        windowed_output: torch.Tensor,
        indices: torch.Tensor,
        num_windows: int,
        batch_size: int,
        seq_len: int,
        dim: int
    ) -> torch.Tensor:
        """
        Merge windowed outputs back into sequence.
        
        Args:
            windowed_output: Output tensor from windowed attention
            indices: Window indices tensor
            num_windows: Number of windows
            batch_size: Batch size
            seq_len: Sequence length
            dim: Hidden dimension
            
        Returns:
            Merged output tensor
        """
        # Initialize output tensor
        output = torch.zeros(
            batch_size, seq_len, dim,
            device=windowed_output.device
        )
        
        # Count overlaps for averaging
        overlap_count = torch.zeros(
            batch_size, seq_len,
            device=windowed_output.device
        )
        
        # Reshape windowed output
        windowed_output = windowed_output.view(
            batch_size, num_windows, self.window_size, dim
        )
        
        # Scatter windows back to their positions
        for i in range(num_windows):
            window_indices = indices[i]
            output[:, window_indices] += windowed_output[:, i]
            overlap_count[:, window_indices] += 1
        
        # Average overlapping regions
        output = output / (overlap_count.unsqueeze(-1) + 1e-8)
        
        return output
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with sliding window attention.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Tuple of (output tensor, optional attention weights)
        """
        try:
            batch_size, seq_len, d_model = hidden_states.shape
            device = hidden_states.device
            

            
            # Create sliding windows
            windows, indices, num_windows = self._create_sliding_windows(
                hidden_states, batch_size, seq_len, d_model
            )
            
            # Create window-specific attention mask
            window_mask = self.mask_generator.create_local_mask(
                self.window_size,
                self.window_size,
                batch_size * num_windows,
                self.base_attention.num_heads,
                device
            )
            
            if attention_mask is not None:
                # Adjust global attention mask to window size
                window_mask = window_mask + attention_mask.repeat_interleave(
                    num_windows, dim=0
                )
            
            # Apply attention within windows
            windowed_output, attention_weights = self.base_attention(
                windows,
                attention_mask=window_mask,
                return_attention_weights=return_attention_weights
            )
            
            # Merge windows back
            output = self._merge_windows(
                windowed_output, indices, num_windows,
                batch_size, seq_len, d_model
            )
            

            
            if return_attention_weights:
                # Reshape attention weights if needed
                if attention_weights is not None:
                    attention_weights = attention_weights.view(
                        batch_size, num_windows,
                        self.base_attention.num_heads,
                        self.window_size, self.window_size
                    )
                return output, attention_weights
            
            return output, None
            
        except Exception as e:
            logger.error(f"Error in sliding window attention forward pass: {str(e)}")
            raise
            
