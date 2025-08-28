#!/usr/bin/env python3
"""
Benchmarking script for Transformer models.

This script implements end-to-end benchmarking of forward and backward passes
for Transformer language models, supporting various configurations and mixed precision.
"""

import argparse
import time
import timeit
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import json
import sys
import os

# Add the cs336-basics module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cs336-basics'))

try:
    from cs336_basics.model import BasicsTransformerLM
    from cs336_basics.optimizer import AdamW
except ImportError as e:
    print(f"Error importing cs336_basics: {e}")
    print("Make sure the cs336-basics package is properly installed.")
    sys.exit(1)


# Model configurations from the assignment
MODEL_CONFIGS = {
    'small': {
        'd_model': 768,
        'd_ff': 3072,
        'num_layers': 12,
        'num_heads': 12,
        'vocab_size': 10000,
        'context_length': 256,
        'rope_theta': 10000.0
    },
    'medium': {
        'd_model': 1024,
        'd_ff': 4096,
        'num_layers': 24,
        'num_heads': 16,
        'vocab_size': 10000,
        'context_length': 256,
        'rope_theta': 10000.0
    },
    'large': {
        'd_model': 1280,
        'd_ff': 5120,
        'num_layers': 36,
        'num_heads': 20,
        'vocab_size': 10000,
        'context_length': 256,
        'rope_theta': 10000.0
    },
    'xl': {
        'd_model': 1600,
        'd_ff': 6400,
        'num_layers': 48,
        'num_heads': 25,
        'vocab_size': 10000,
        'context_length': 256,
        'rope_theta': 10000.0
    },
    '2.7B': {
        'd_model': 2560,
        'd_ff': 10240,
        'num_layers': 32,
        'num_heads': 32,
        'vocab_size': 10000,
        'context_length': 256,
        'rope_theta': 10000.0
    }
}


class ModelBenchmarker:
    """Benchmarking harness for Transformer models."""
    
    def __init__(self, device: str = 'cuda', mixed_precision: bool = False, 
                 precision_dtype: torch.dtype = torch.bfloat16):
        """
        Initialize the benchmarker.
        
        Args:
            device: Device to run benchmarks on ('cuda', 'cpu', 'mps')
            mixed_precision: Whether to use mixed precision training
            precision_dtype: Data type for mixed precision (torch.float16 or torch.bfloat16)
        """
        self.device = device
        self.mixed_precision = mixed_precision
        self.precision_dtype = precision_dtype
        
        # Ensure device is available
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = 'cpu'
        elif device == 'mps' and not torch.backends.mps.is_available():
            print("MPS not available, falling back to CPU")
            self.device = 'cpu'
    
    def create_model(self, config: Dict[str, Any]) -> nn.Module:
        """Create a model with the given configuration."""
        model = BasicsTransformerLM(**config)
        model = model.to(self.device)
        
        if self.mixed_precision and self.device == 'cuda':
            # Convert model to half precision for mixed precision training
            model = model.to(self.precision_dtype)
        
        return model
    
    def create_random_batch(self, batch_size: int, context_length: int, 
                          vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create a random batch of input data and targets."""
        # Input tokens
        input_ids = torch.randint(0, vocab_size, (batch_size, context_length), 
                                device=self.device, dtype=torch.long)
        
        # Target tokens (shifted by 1 for language modeling)
        targets = torch.randint(0, vocab_size, (batch_size, context_length), 
                              device=self.device, dtype=torch.long)
        
        return input_ids, targets
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss."""
        # Reshape for cross-entropy: (batch_size * seq_len, vocab_size) and (batch_size * seq_len,)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        loss = nn.functional.cross_entropy(logits_flat, targets_flat)
        return loss
    
    def synchronize(self):
        """Synchronize device operations."""
        if self.device == 'cuda':
            torch.cuda.synchronize()
        elif self.device == 'mps':
            torch.mps.synchronize()
        # CPU doesn't need synchronization
    
    def benchmark_forward(self, model: nn.Module, input_ids: torch.Tensor, 
                         warmup_steps: int = 5, measurement_steps: int = 10) -> Dict[str, float]:
        """Benchmark forward pass only."""
        model.eval()
        
        # Warmup
        for _ in range(warmup_steps):
            with torch.no_grad():
                if self.mixed_precision and self.device == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=self.precision_dtype):
                        _ = model(input_ids)
                else:
                    _ = model(input_ids)
            self.synchronize()
        
        # Measurement
        times = []
        for _ in range(measurement_steps):
            start_time = timeit.default_timer()
            
            with torch.no_grad():
                if self.mixed_precision and self.device == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=self.precision_dtype):
                        _ = model(input_ids)
                else:
                    _ = model(input_ids)
            
            self.synchronize()
            end_time = timeit.default_timer()
            times.append(end_time - start_time)
        
        return {
            'mean_time': sum(times) / len(times),
            'std_time': (sum((t - sum(times) / len(times))**2 for t in times) / len(times))**0.5,
            'times': times
        }
    
    def benchmark_forward_backward(self, model: nn.Module, input_ids: torch.Tensor, 
                                 targets: torch.Tensor, warmup_steps: int = 5, 
                                 measurement_steps: int = 10) -> Dict[str, float]:
        """Benchmark forward and backward pass."""
        model.train()
        
        # Warmup
        for _ in range(warmup_steps):
            if self.mixed_precision and self.device == 'cuda':
                with torch.autocast(device_type='cuda', dtype=self.precision_dtype):
                    logits = model(input_ids)
                    loss = self.compute_loss(logits, targets)
            else:
                logits = model(input_ids)
                loss = self.compute_loss(logits, targets)
            
            loss.backward()
            model.zero_grad()
            self.synchronize()
        
        # Measurement
        times = []
        for _ in range(measurement_steps):
            start_time = timeit.default_timer()
            
            if self.mixed_precision and self.device == 'cuda':
                with torch.autocast(device_type='cuda', dtype=self.precision_dtype):
                    logits = model(input_ids)
                    loss = self.compute_loss(logits, targets)
            else:
                logits = model(input_ids)
                loss = self.compute_loss(logits, targets)
            
            loss.backward()
            model.zero_grad()
            
            self.synchronize()
            end_time = timeit.default_timer()
            times.append(end_time - start_time)
        
        return {
            'mean_time': sum(times) / len(times),
            'std_time': (sum((t - sum(times) / len(times))**2 for t in times) / len(times))**0.5,
            'times': times
        }
    
    def benchmark_training_step(self, model: nn.Module, input_ids: torch.Tensor, 
                              targets: torch.Tensor, optimizer: torch.optim.Optimizer,
                              warmup_steps: int = 5, measurement_steps: int = 10) -> Dict[str, float]:
        """Benchmark a complete training step (forward + backward + optimizer step)."""
        model.train()
        
        # Warmup
        for _ in range(warmup_steps):
            optimizer.zero_grad()
            
            if self.mixed_precision and self.device == 'cuda':
                with torch.autocast(device_type='cuda', dtype=self.precision_dtype):
                    logits = model(input_ids)
                    loss = self.compute_loss(logits, targets)
            else:
                logits = model(input_ids)
                loss = self.compute_loss(logits, targets)
            
            loss.backward()
            optimizer.step()
            self.synchronize()
        
        # Measurement
        times = []
        for _ in range(measurement_steps):
            start_time = timeit.default_timer()
            
            optimizer.zero_grad()
            
            if self.mixed_precision and self.device == 'cuda':
                with torch.autocast(device_type='cuda', dtype=self.precision_dtype):
                    logits = model(input_ids)
                    loss = self.compute_loss(logits, targets)
            else:
                logits = model(input_ids)
                loss = self.compute_loss(logits, targets)
            
            loss.backward()
            optimizer.step()
            
            self.synchronize()
            end_time = timeit.default_timer()
            times.append(end_time - start_time)
        
        return {
            'mean_time': sum(times) / len(times),
            'std_time': (sum((t - sum(times) / len(times))**2 for t in times) / len(times))**0.5,
            'times': times
        }


def run_benchmark(model_size: str, batch_size: int = 4, context_length: Optional[int] = None,
                 device: str = 'cuda', mixed_precision: bool = False, 
                 precision_dtype: str = 'bfloat16', warmup_steps: int = 5, 
                 measurement_steps: int = 10, benchmark_type: str = 'all') -> Dict[str, Any]:
    """
    Run benchmark for a specific model configuration.
    
    Args:
        model_size: Model size ('small', 'medium', 'large', 'xl', '2.7B')
        batch_size: Batch size for benchmarking
        context_length: Context length (if None, use default from config)
        device: Device to run on
        mixed_precision: Whether to use mixed precision
        precision_dtype: Precision dtype ('float16' or 'bfloat16')
        warmup_steps: Number of warmup steps
        measurement_steps: Number of measurement steps
        benchmark_type: Type of benchmark ('forward', 'forward_backward', 'training', 'all')
    
    Returns:
        Dictionary with benchmark results
    """
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_size].copy()
    if context_length is not None:
        config['context_length'] = context_length
    
    # Convert precision dtype string to torch dtype
    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16}
    precision_dtype_torch = dtype_map.get(precision_dtype, torch.bfloat16)
    
    # Initialize benchmarker
    benchmarker = ModelBenchmarker(device=device, mixed_precision=mixed_precision, 
                                 precision_dtype=precision_dtype_torch)
    
    # Create model
    print(f"Creating {model_size} model...")
    model = benchmarker.create_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create random data
    input_ids, targets = benchmarker.create_random_batch(
        batch_size, config['context_length'], config['vocab_size']
    )
    
    results = {
        'model_size': model_size,
        'config': config,
        'batch_size': batch_size,
        'device': device,
        'mixed_precision': mixed_precision,
        'precision_dtype': precision_dtype,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'warmup_steps': warmup_steps,
        'measurement_steps': measurement_steps
    }
    
    # Run benchmarks
    if benchmark_type in ['forward', 'all']:
        print("Benchmarking forward pass...")
        forward_results = benchmarker.benchmark_forward(
            model, input_ids, warmup_steps, measurement_steps
        )
        results['forward'] = forward_results
        print(f"Forward pass: {forward_results['mean_time']:.4f} ± {forward_results['std_time']:.4f} seconds")
    
    if benchmark_type in ['forward_backward', 'all']:
        print("Benchmarking forward + backward pass...")
        forward_backward_results = benchmarker.benchmark_forward_backward(
            model, input_ids, targets, warmup_steps, measurement_steps
        )
        results['forward_backward'] = forward_backward_results
        print(f"Forward + Backward: {forward_backward_results['mean_time']:.4f} ± {forward_backward_results['std_time']:.4f} seconds")
    
    if benchmark_type in ['training', 'all']:
        print("Benchmarking training step...")
        # Create optimizer
        optimizer = AdamW(model.parameters(), lr=1e-4)
        training_results = benchmarker.benchmark_training_step(
            model, input_ids, targets, optimizer, warmup_steps, measurement_steps
        )
        results['training'] = training_results
        print(f"Training step: {training_results['mean_time']:.4f} ± {training_results['std_time']:.4f} seconds")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark Transformer models')
    parser.add_argument('--model-size', type=str, default='small',
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Model size to benchmark')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for benchmarking')
    parser.add_argument('--context-length', type=int, default=None,
                       help='Context length (default: use model config)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'mps'],
                       help='Device to run benchmarks on')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--precision-dtype', type=str, default='bfloat16',
                       choices=['float16', 'bfloat16'],
                       help='Precision dtype for mixed precision')
    parser.add_argument('--warmup-steps', type=int, default=5,
                       help='Number of warmup steps')
    parser.add_argument('--measurement-steps', type=int, default=10,
                       help='Number of measurement steps')
    parser.add_argument('--benchmark-type', type=str, default='all',
                       choices=['forward', 'forward_backward', 'training', 'all'],
                       help='Type of benchmark to run')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save results (JSON format)')
    parser.add_argument('--all-sizes', action='store_true',
                       help='Benchmark all model sizes')
    parser.add_argument('--context-lengths', type=int, nargs='+', default=None,
                       help='Multiple context lengths to benchmark')
    
    args = parser.parse_args()
    
    # Determine which model sizes to benchmark
    if args.all_sizes:
        model_sizes = list(MODEL_CONFIGS.keys())
    else:
        model_sizes = [args.model_size]
    
    # Determine which context lengths to benchmark
    if args.context_lengths:
        context_lengths = args.context_lengths
    else:
        context_lengths = [args.context_length]
    
    all_results = []
    
    for model_size in model_sizes:
        for context_length in context_lengths:
            print(f"\n{'='*60}")
            print(f"Benchmarking {model_size} model")
            if context_length:
                print(f"Context length: {context_length}")
            print(f"{'='*60}")
            
            try:
                results = run_benchmark(
                    model_size=model_size,
                    batch_size=args.batch_size,
                    context_length=context_length,
                    device=args.device,
                    mixed_precision=args.mixed_precision,
                    precision_dtype=args.precision_dtype,
                    warmup_steps=args.warmup_steps,
                    measurement_steps=args.measurement_steps,
                    benchmark_type=args.benchmark_type
                )
                all_results.append(results)
                
            except Exception as e:
                print(f"Error benchmarking {model_size}: {e}")
                continue
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    
    for result in all_results:
        print(f"\nModel: {result['model_size']}")
        print(f"Context Length: {result['config']['context_length']}")
        print(f"Parameters: {result['total_params']:,}")
        
        if 'forward' in result:
            print(f"Forward: {result['forward']['mean_time']:.4f} ± {result['forward']['std_time']:.4f}s")
        if 'forward_backward' in result:
            print(f"Forward+Backward: {result['forward_backward']['mean_time']:.4f} ± {result['forward_backward']['std_time']:.4f}s")
        if 'training' in result:
            print(f"Training Step: {result['training']['mean_time']:.4f} ± {result['training']['std_time']:.4f}s")


if __name__ == '__main__':
    main()
