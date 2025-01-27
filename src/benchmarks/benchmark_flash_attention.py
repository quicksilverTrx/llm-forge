# src/benchmarks/benchmark_flash_attention.py

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model
import logging
import time
from typing import Dict, Any
import numpy as np

from ..model.model_patcher import patch_attention_in_gpt2
from ..model.train_utils import (
    prepare_wikitext2, 
    train_epoch_fp16,
    evaluate,
    LanguageModelingWrapper,
    report_gpu_usage
)
from ..attention.flash_attention import FlashAttention, FlashAttentionConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BenchmarkResults:
    def __init__(self):
        self.metrics = {
            'base_model': {},
            'flash_attention': {}
        }

    def add_metric(self, model_type: str, metric_name: str, value: Any):
        self.metrics[model_type][metric_name] = value

    def print_comparison(self):
        logger.info("\n=== Benchmark Results ===")
        for metric in self.metrics['base_model'].keys():
            base_val = self.metrics['base_model'][metric]
            flash_val = self.metrics['flash_attention'][metric]
            improvement = ((flash_val - base_val) / base_val) * 100
            logger.info(f"{metric}:")
            logger.info(f"  Base Model: {base_val:.4f}")
            logger.info(f"  Flash Attention: {flash_val:.4f}")
            logger.info(f"  Improvement: {improvement:.2f}%\n")

def setup_models(device: torch.device):
    # Load base model
    base_model = GPT2Model.from_pretrained('gpt2')
    base_model = LanguageModelingWrapper(base_model)
    
    # Create Flash Attention model
    flash_config = FlashAttentionConfig(
        dropout=0.1,
        max_sequence_length=1024
    )
    flash_model = patch_attention_in_gpt2(
        base_model.base_model,
        FlashAttention,
        attention_config=flash_config
    )
    flash_model = LanguageModelingWrapper(flash_model)

    return base_model.to(device), flash_model.to(device)

def benchmark_training(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int = 1
) -> Dict[str, float]:
    """Benchmark model training and evaluation"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Training metrics
    train_times = []
    train_losses = []
    memory_usage = []
    
    for epoch in range(num_epochs):
        # Training
        start_time = time.time()
        model.train()
        
        # Record initial memory
        memory_usage.append(torch.cuda.max_memory_allocated(device) / 1024**2)
        
        train_epoch_fp16(model, train_loader, optimizer, device)
        
        epoch_time = time.time() - start_time
        train_times.append(epoch_time)
        
        # Evaluation
        model.eval()
        eval_loss = evaluate(model, eval_loader, device)
        train_losses.append(eval_loss)
        
        # Record peak memory
        memory_usage.append(torch.cuda.max_memory_allocated(device) / 1024**2)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Time: {epoch_time:.2f}s")
        logger.info(f"Loss: {eval_loss:.4f}")
        logger.info(f"Peak Memory: {max(memory_usage):.2f}MB")
        
    return {
        'avg_epoch_time': np.mean(train_times),
        'final_loss': train_losses[-1],
        'peak_memory': max(memory_usage)
    }

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Prepare data
    train_loader, eval_loader = prepare_wikitext2(
        tokenizer,
        batch_size=8,
        seq_length=512
    )
    
    # Initialize models
    base_model, flash_model = setup_models(device)
    results = BenchmarkResults()
    
    # Benchmark base model
    logger.info("Benchmarking Base Model...")
    base_metrics = benchmark_training(
        base_model,
        train_loader,
        eval_loader,
        device
    )
    
    for metric, value in base_metrics.items():
        results.add_metric('base_model', metric, value)
    
    # Clear GPU memory
    del base_model
    torch.cuda.empty_cache()
    
    # Benchmark Flash Attention model
    logger.info("\nBenchmarking Flash Attention Model...")
    flash_metrics = benchmark_training(
        flash_model,
        train_loader,
        eval_loader,
        device
    )
    
    for metric, value in flash_metrics.items():
        results.add_metric('flash_attention', metric, value)
    
    # Print results
    results.print_comparison()

if __name__ == "__main__":
    main()