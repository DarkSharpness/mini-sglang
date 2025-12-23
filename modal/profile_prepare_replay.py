import sys
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from minisgl.attention.fa3 import FlashAttentionBackend, FA3CaptureData, FA3Metadata
from minisgl.core import Batch, Req
from minisgl.kvcache import create_kvcache
from minisgl.models import ModelConfig, RotaryConfig


def create_dummy_batch(bs: int, max_seq_len: int, device: torch.device) -> Batch:
    """Create a dummy batch for testing."""
    dummy_reqs = []
    for i in range(bs):
        req = Req(
            input_ids=torch.randint(0, 1000, (1,), dtype=torch.int32, device="cpu"),
            table_idx=i,
            cached_len=0,
            output_len=1,
            uid=i,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        dummy_reqs.append(req)
    
    batch = Batch(reqs=dummy_reqs, phase="decode")
    batch.input_ids = torch.randint(0, 1000, (bs,), dtype=torch.int32, device=device)
    batch.out_loc = torch.arange(bs, dtype=torch.int32, device=device)
    
    # Create dummy metadata
    metadata = FA3Metadata(
        cu_seqlens_k=torch.arange(bs + 1, dtype=torch.int32, device=device),
        cu_seqlens_q=torch.arange(bs + 1, dtype=torch.int32, device=device),
        positions=torch.arange(bs, dtype=torch.int32, device=device),
        cache_seqlens=torch.ones(bs, dtype=torch.int32, device=device) * 10,
        max_seqlen_k=max_seq_len,
        max_seqlen_q=1,
        page_table=torch.randint(0, 100, (bs, max_seq_len), dtype=torch.int32, device=device),
    )
    batch.attn_metadata = metadata
    batch.padded_reqs = batch.reqs
    
    return batch


def benchmark_prepare_for_replay(
    backend: FlashAttentionBackend,
    batch: Batch,
    num_iterations: int = 1000,
    warmup: int = 10,
) -> dict:
    """
    Benchmark prepare_for_replay() copy operations.
    
    Returns:
        dict with timing information
    """
    # Warmup
    for _ in range(warmup):
        backend.prepare_for_replay(batch)
    
    torch.cuda.synchronize()
    
    # Measure
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iterations):
        backend.prepare_for_replay(batch)
    end_event.record()
    
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_us = (elapsed_ms * 1000) / num_iterations
    
    return {
        "total_time_ms": elapsed_ms,
        "num_iterations": num_iterations,
        "avg_time_us": avg_us,
        "throughput_per_sec": num_iterations / (elapsed_ms / 1000),
    }


def main():
    """Main benchmark function."""
    device = torch.device("cuda:0")
    max_seq_len = 4096
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Create minimal config for testing
    config = ModelConfig(
        vocab_size=10000,
        hidden_size=512,
        num_layers=1,  # Minimal for testing
        num_qo_heads=8,
        num_kv_heads=8,
        head_dim=64,
        intermediate_size=2048,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=False,
        rotary_config=RotaryConfig(
            head_dim=64,
            rotary_dim=64,
            max_position=4096,
            base=10000.0,
            scaling=None,
        ),
    )
    
    # Create KV cache
    kv_cache = create_kvcache(
        model_config=config,
        num_pages=1000,
        dtype=torch.float16,
        device=device,
    )
    
    # Create page table
    page_table = torch.zeros((256, max_seq_len), dtype=torch.int32, device=device)
    
    # Create backend
    backend = FlashAttentionBackend(config, kv_cache, page_table)
    
    # Initialize capture for different batch sizes
    bs_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=bs_list)
    
    print("=" * 80)
    print("Benchmarking prepare_for_replay() Copy Overhead")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Max Seq Len: {max_seq_len}")
    print(f"Num Iterations: 1000")
    print("=" * 80)
    print(f"{'Batch Size':<12} {'Avg Time (μs)':<15} {'Throughput (ops/s)':<20}")
    print("-" * 80)
    
    results: List[dict] = []
    
    for bs in batch_sizes:
        if bs > 256:
            continue
            
        batch = create_dummy_batch(bs, max_seq_len, device)
        
        # Prepare for capture first
        backend.prepare_for_capture(batch)
        
        # Benchmark
        stats = benchmark_prepare_for_replay(backend, batch, num_iterations=1000)
        results.append({"batch_size": bs, **stats})
        
        print(
            f"{bs:<12} {stats['avg_time_us']:<15.2f} {stats['throughput_per_sec']:<20.0f}"
        )
    
    print("=" * 80)
    print("\nSummary:")
    print(f"Min overhead: {min(r['avg_time_us'] for r in results):.2f} μs (bs={min(r['batch_size'] for r in results)})")
    print(f"Max overhead: {max(r['avg_time_us'] for r in results):.2f} μs (bs={max(r['batch_size'] for r in results)})")
    print(f"Average overhead: {sum(r['avg_time_us'] for r in results) / len(results):.2f} μs")
    
    return results


if __name__ == "__main__":
    main()

