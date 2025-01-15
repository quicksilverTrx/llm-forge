from transformers import GPT2Model, GPT2Tokenizer
#from flash_attn import flash_attn_func
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
from torch.optim import AdamW
from src.attention.optimizations.grouped_query_attention import GroupedQueryAttention, GQAConfig
from model_patcher import patch_attention_in_gpt2, add_rope_to_attention, remove_position_embeddings
from train_utils import  train_epoch, evaluate, prepare_wikitext2, LanguageModelingWrapper
def main():

    # Load base model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = GPT2Model.from_pretrained('gpt2')
    gpt2_config = base_model.config
    # 1. Patch single attention variant
    gqa_config = GQAConfig(
            num_query_heads=gpt2_config.n_head,  # 12 for GPT-2 small
            num_kv_heads=gpt2_config.n_head ,  # Reduce KV heads by factor of 3
            head_dim=gpt2_config.n_embd // gpt2_config.n_head,  # 768/12 = 64
            dropout=gpt2_config.attn_pdrop
        )

    
    
    gqa_model = patch_attention_in_gpt2(base_model, GroupedQueryAttention,gqa_config)
    
    
    

    final_model = gqa_model
    final_model = LanguageModelingWrapper(final_model)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    
    final_model.to(device)
    # Prepare data
    train_loader, eval_loader = prepare_wikitext2(tokenizer)

    optimizers =  AdamW(final_model.parameters(), lr=5e-5)

    for epoch in range (100):
        train_epoch(final_model, train_loader, optimizers, device)
        evaluate(final_model, eval_loader, device)

if __name__ =="__main__":
  main()