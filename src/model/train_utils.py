import torch
import torch.nn as nn 
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Model, DataCollatorForLanguageModeling
import time
import numpy as np
import logging
import tqdm
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class LanguageModelingWrapper(nn.Module):
    """Wrapper that adds language modeling loss calculation to GPT2Model"""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # Share embedding weights with output projection
        self.lm_head = lambda x: torch.matmul(x, self.base_model.wte.weight.t())

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        try:
            # Get hidden states from base model
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
            
            hidden_states = outputs.last_hidden_state
            logits = self.lm_head(hidden_states)

            loss = None
            if labels is not None:
                # Shift logits and labels for next token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Calculate loss
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

            return type('ModelOutput', (), {
                'loss': loss,
                'logits': logits,
                'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
                'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
            })

        except Exception as e:
            logger.error(f"Forward pass error: {str(e)}")
            raise

def prepare_wikitext2 (tokenizer,batch_size=16, seq_length=512):
    dataset = load_dataset('wikitext','wikitext-2-v1')

    def tokenizer_fn(text):
        return tokenizer(text['text'], truncation=True, max_length=seq_length, padding='max_length',return_tensors="pt")    
    

    dataset = dataset.map(function=tokenizer_fn, batched=True, remove_columns=['text'])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, collate_fn=collator)

    eval_loader = DataLoader(dataset['validation'], batch_size=batch_size, shuffle=False, collate_fn=collator)  

    return train_loader, eval_loader

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for step, batch in enumerate(data_loader):
        batch = {k : v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"])
        loss = outputs.loss

        batch_tokens = batch["input_ids"].numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if(step%100==0):
            print(f"step :{step} training loss : {loss.item()}")

        del batch, outputs
        torch.cuda.empty_cache()  # Optionally clear GPU memory

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for step,batch in enumerate(data_loader):
        batch = {k : v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"])
            loss = outputs.loss

        batch_tokens = batch["input_ids"].numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        del batch, outputs
        torch.cuda.empty_cache()  # Optionally clear GPU memory

