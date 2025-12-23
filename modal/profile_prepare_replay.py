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
    Benchmark prepare_for_replay() copy operations with detailed metrics.
    
    Returns:
        dict with timing information including breakdown
    """
    # Warmup
    for _ in range(warmup):
        backend.prepare_for_replay(batch)
    
    torch.cuda.synchronize()
    
    # Measure prepare_for_replay time
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iterations):
        backend.prepare_for_replay(batch)
    end_event.record()
    
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_us = (elapsed_ms * 1000) / num_iterations
    
    # Calculate copy operation details
    bs = batch.size
    max_seqlen_k = batch.attn_metadata.max_seqlen_k
    
    # All copies are int32 (4 bytes)
    copy_ops = [
        {"name": "input_ids", "size": bs, "bytes": bs * 4},
        {"name": "out_loc", "size": bs, "bytes": bs * 4},
        {"name": "cu_seqlens_k", "size": bs + 1, "bytes": (bs + 1) * 4},
        {"name": "positions", "size": bs, "bytes": bs * 4},
        {"name": "seq_lens", "size": bs, "bytes": bs * 4},
        {"name": "page_table", "size": bs * max_seqlen_k, "bytes": bs * max_seqlen_k * 4},
    ]
    total_bytes = sum(op["bytes"] for op in copy_ops)
    
    return {
        "total_time_ms": elapsed_ms,
        "num_iterations": num_iterations,
        "avg_time_us": avg_us,
        "throughput_per_sec": num_iterations / (elapsed_ms / 1000),
        "batch_size": bs,
        "max_seqlen_k": max_seqlen_k,
        "copy_operations": copy_ops,
        "total_bytes_per_call": total_bytes,
        "bandwidth_gbps": (total_bytes * num_iterations) / (elapsed_ms / 1000) / 1e9,
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
    
    print("=" * 70)
    print("prepare_for_replay() Copy Overhead - Baseline")
    print("=" * 70)
    print(f"Device: {device} | Max Seq Len: {max_seq_len} | Iterations: 1000\n")
    print(f"{'BS':<6} {'Time (μs)':<12} {'Throughput':<15} {'Bandwidth':<12} {'Bytes (KB)':<12}")
    print("-" * 70)
    
    results: List[dict] = []
    
    for bs in batch_sizes:
        if bs > 256:
            continue
            
        batch = create_dummy_batch(bs, max_seq_len, device)
        backend.prepare_for_capture(batch)
        stats = benchmark_prepare_for_replay(backend, batch, num_iterations=1000)
        results.append({"batch_size": bs, **stats})
        
        print(
            f"{bs:<6} {stats['avg_time_us']:<12.2f} "
            f"{stats['throughput_per_sec']:<15.0f} "
            f"{stats['bandwidth_gbps']:<12.2f} "
            f"{stats['total_bytes_per_call'] / 1024:<12.2f}"
        )
    
    print("=" * 70)
    avg_time = sum(r['avg_time_us'] for r in results) / len(results)
    print(f"\nSummary: {min(r['avg_time_us'] for r in results):.2f} - {max(r['avg_time_us'] for r in results):.2f} μs (avg: {avg_time:.2f} μs)")
    
    # Show copy breakdown for largest batch
    largest = max(results, key=lambda x: x['batch_size'])
    print(f"\nCopy Breakdown (BS={largest['batch_size']}, seq_len={largest['max_seqlen_k']}):")
    total_bytes = largest['total_bytes_per_call']
    for op in largest['copy_operations']:
        pct = (op['bytes'] / total_bytes * 100) if total_bytes > 0 else 0
        print(f"  {op['name']:<15} {op['size']:>8} elems  {op['bytes']/1024:>8.2f} KB  ({pct:>5.1f}%)")
    
    return results


if __name__ == "__main__":
    main()

