# src/model/gpt2_scratch.py

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
import logging
from ..attention.base_attention import ScaledDotProductAttention
from ..attention.multi_head import MultiHeadAttention

logger = logging.getLogger(__name__)

class GPT2Config:
    """Configuration class for GPT-2 Small"""
    def __init__(self):
        self.vocab_size = 50257
        self.n_positions = 1024
        self.n_embd = 768
        self.n_layer = 12
        self.n_head = 12
        self.dropout = 0.1
        self.layer_norm_epsilon = 1e-5
        self.initializer_range = 0.02

class GPT2Block(nn.Module):
    """Transformer block for GPT-2 using Multi-headed SDPA"""
    def __init__(self,config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = MultiHeadAttention(
            d_model=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout
        )
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask : Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        #attention with residual connection
        attn_output, _ =self.attn(self.len_1(x),attention_mask=attention_mask)
        x = x + attn_output

        # MLP with residual connection
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2SmallSDPA(nn.Module):
    """GPT-2 Small implementation with Multi-headed SDPA"""
    def __init__(self, config :GPT2Config):
        super().__init__()
        self.config = config

        # Token embedding layer
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # Positional embedding layer
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)

        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPT2Block(config) for _ in range(config.n_layer)       
        ])

        # Layer norm for final output
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        #initialise weights
        self.apply(self._init_weights)
        logger.info(f"Initialised GPT2 small model with {self.num_parameters():,} parameters")

    def _init_weights(self, module : nn.Module):
        if isinstance(module,(nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module,nn.LayerNorm):
            module.bias.data._zero_()
            module.weight.data.fill_(1.0)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    