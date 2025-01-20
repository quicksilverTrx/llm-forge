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
    
    def get_position_ids(
        self,
        input_ids:torch.LongTensor,
        device : torch.device
    ) -> torch.LongTensor:
        seq_length = input_ids.size(-1)
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=device
        )
        return position_ids.unsqueeze(0).expand_as(input_ids)
    
    def forward(
        self,
        input_ids: torch.Longtensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None ,
        labels: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        try:
            device = input_ids.device
            batch_size, seq_length = input_ids.size()

            if position_ids is None:
                position_ids = self.get_position_ids(input_ids, device)

            #get token embeddings
            input_embeddings = self.wte(input_ids)
            #get positional embeddings
            position_embeddings = self.wpe(position_ids)
            #add both embeddings
            hidden_states = input_embeddings + position_embeddings
            hidden_states = self.drop(hidden_states)'
            
            #transformer blocks
            for block in self.blocks:
                hidden_states = block(hidden_states, attention_mask=attention_mask)

            #Final Layer Norm
            hidden_states = self.ln_f(hidden_states)

            loss = None
            if labels is not None:
                #reshape logits and compute loss
                logits = torch.matmul(hidden_states, self.wte.weight.t())
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))
                
            
            logits = torch.matmul(hidden_states, self.wte.weight.t())
            # Create HuggingFace-style output structure    
            return type('ModelOutput', (), {
                'loss': loss,
                'logits': logits,
                'hidden_states': hidden_states,
                'last_hidden_state': hidden_states,
                'attentions': None  # Can be extended to include attention outputs if needed
            })
        
        except Exception as e:
            logger.error(f"Error in GPT2 forward pass: {str(e)}")
            raise

def create_gpt2_small() -> GPT2SmallSDPA:
    """Factory function to create a GPT-2 Small model"""
    try:
        config = GPT2Config()
        model = GPT2SmallSDPA(config)
        logger.info("Successfully created GPT-2 Small model")
        return model
    except Exception as e:
        logger.error(f"Error creating GPT-2 Small model: {str(e)}")
        raise       
            

if __name__ == "__main__":
    try:
        # Create model
        model = create_gpt2_small()
        
        # Create dummy inputs
        batch_size = 4
        seq_length = 512
        input_ids = torch.randint(0, 50257, (batch_size, seq_length))
        
        # Forward pass
        outputs, _ = model(input_ids)
        print(f"Output shape: {outputs.shape}")
        
    except Exception as e:
        logger.error(f"Error in example usage: {str(e)}")