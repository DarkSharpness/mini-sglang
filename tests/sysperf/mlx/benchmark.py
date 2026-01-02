"""
MLX CPU Benchmark - Using MLX Community Models
Runs offline inference benchmark using mlx-lm and logs Total, Time, Throughput
"""
import time
from random import randint, seed

try:
    from mlx_lm import load, generate
    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False
    print("Note: mlx-lm not installed. Install with: uv pip install mlx-lm")


def main():
    """Run offline benchmark on CPU using MLX community models"""
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_output_len = 1024
    
    print("=" * 70)
    print("MLX CPU Benchmark (using mlx-lm)")
    print("=" * 70)
    
    if not MLX_LM_AVAILABLE:
        print("ERROR: mlx-lm not available.")
        print("Install with: uv pip install mlx-lm")
        return
    
    # Use Qwen3-0.6B-4bit to match Modal benchmark model
    model_name = "mlx-community/Qwen3-0.6B-4bit"
    
    print(f"Loading model: {model_name}")
    model, tokenizer = load(model_name)
    print("Model loaded successfully")
    
    # Generate prompts
    prompts = [
        " ".join([str(randint(0, 1000)) for _ in range(randint(100, max_input_len))])
        for _ in range(num_seqs)
    ]
    
    # Warmup
    print("Warming up...")
    generate(model, tokenizer, prompt="Benchmark", max_tokens=10, verbose=False)
    
    # Benchmark
    print(f"Running benchmark with {num_seqs} sequences...")
    start_time = time.time()
    
    total_tokens = 0
    for i, prompt in enumerate(prompts):
        if i % 50 == 0:
            print(f"  Progress: {i}/{num_seqs}")
        
        # Generate with random output length
        output_len = randint(100, max_output_len)
        generate(model, tokenizer, prompt=prompt, max_tokens=output_len, verbose=False)
        total_tokens += output_len
    
    elapsed_time = time.time() - start_time
    throughput = total_tokens / elapsed_time if elapsed_time > 0 else 0
    
    print("=" * 70)
    print(f"Total: {total_tokens}tok, Time: {elapsed_time:.2f}s, Throughput: {throughput:.2f}tok/s")
    print("=" * 70)
    
    return {
        "total_tokens": total_tokens,
        "time": elapsed_time,
        "throughput": throughput
    }


if __name__ == "__main__":
    main()

