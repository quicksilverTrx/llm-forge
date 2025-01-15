import torch
import torch.nn as nn
from typing import Type, Dict, Optional
from copy import deepcopy
from transformers import GPT2Model

def patch_attention_in_gpt2(
    model: nn.Module,
    new_attention_class: Type[nn.Module],
    attention_config: Optional[dict] = None,
    layer_indices: Optional[list] = None
) -> nn.Module:
    """
    Patch specific attention variant into GPT2 model.
    
    Args:
        model: Original GPT2 model
        new_attention_class: Attention mechanism to patch in
        layer_indices: Which layers to patch (None means all layers)
    """
    model = deepcopy(model)
    config = model.config

    # Determine which layers to patch
    if layer_indices is None:
        layer_indices = range(len(model.h))

    # Patch each layer
    for idx in layer_indices:
        # Get old attention module
        old_attn = model.h[idx].attn

        # Create new attention instance
        new_attn = new_attention_class(attention_config)

        try:
            print(old_attn.c_attn.shape)
            print(new_attn.q_proj.weight.shape)
            # Try to load weights from old attention
            new_attn.q_proj.weight.data = old_attn.c_attn.weight[:config.n_embd].clone()
            new_attn.k_proj.weight.data = old_attn.c_attn.weight[config.n_embd:2*config.n_embd].clone()
            new_attn.v_proj.weight.data = old_attn.c_attn.weight[2*config.n_embd:].clone()
            new_attn.out_proj.weight.data = old_attn.c_proj.weight.clone()
            print(f"Successfully transferred weights for layer {idx}")
        except:
            print(f"Using random initialization for layer {idx}")

        # Replace attention module
        model.h[idx].attn = new_attn

    return model


def add_rope_to_attention(
    model: nn.Module,
    max_position_embeddings: Optional[int] = None
) -> nn.Module:
    """
    Add RoPE to existing attention mechanisms.
    
    Args:
        model: GPT2 model
        max_position_embeddings: Maximum sequence length for RoPE
    """
    from ..attention.positional.rope import RotaryPositionalEmbeddings

    model = deepcopy(model)
    config = model.config
    
    if max_position_embeddings is None:
        max_position_embeddings = config.n_positions

    # Create RoPE module
    rope = RotaryPositionalEmbeddings(
        dim=config.n_embd // config.n_head,
        max_position_embeddings=max_position_embeddings
    )

    # Store original attention forward
    for block in model.h:
        original_forward = block.attn.forward

        # Define new forward pass with RoPE
        def new_forward(self, x, attention_mask=None, position_ids=None):
            # Get Q, K, V from input

            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.embed_dim, dim=2)

            # Apply RoPE to Q, K
            if position_ids is None:
                position_ids = torch.arange(x.size(1), device=x.device)
            
            q, k = rope(q, k, position_ids)

            # Continue with original attention computation
            return original_forward(self, x, attention_mask)
        
        # Patch new forward method
        block.attn.forward = new_forward.__get__(block.attn)

    return model

def remove_position_embeddings(model: nn.Module) -> nn.Module:
    """
    Remove position embeddings when using RoPE.
    """
    model = deepcopy(model)
    
    if hasattr(model, 'wpe'):
        del model.wpe
        
        # Modify forward pass to skip position embeddings
        original_forward = model.forward

        def new_forward(self, input_ids=None, attention_mask=None, **kwargs):
            inputs_embeds = self.wte(input_ids)

            # Skip position embeddings addition
            hidden_states = self.drop(inputs_embeds)

            for block in self.h:
                hidden_states = block(hidden_states, attention_mask)
            
            hidden_states = self.ln_f(hidden_states)

            for block in self.h:
                hidden_states = block(hidden_states, attention_mask)

            hidden_states = self.ln_f(hidden_states)
            return hidden_states
        
        model.forward = new_forward.__get__(model)

    return model

if __name__ == "__main__":
    # Load base model
    base_model = GPT2Model.from_pretrained('gpt2')
    
    # 1. Patch single attention variant
    from flash_attn import FlashAttention
    flash_model = patch_attention_in_gpt2(base_model, FlashAttention)
    

    
    # 3. Add RoPE to existing model
    rope_model = add_rope_to_attention(base_model)
    
    # 4. Remove position embeddings
    final_model = remove_position_embeddings(rope_model)
    
    # Test forward pass
    test_input = torch.randint(0, 50257, (2, 512))
    output = final_model(test_input)
    print(f"Output shape: {output.shape}")