import asyncio
import time
import argparse
import numpy as np
import scipy.stats as stats
from openai import AsyncOpenAI

# API CONFIGURATION
API_BASE = "http://127.0.0.1:1920/v1"
API_KEY = "EMPTY"

client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)

def compute_mean_and_ci_stats(data, confidence=0.95):
    """Computes the mean and confidence interval using the Student's t-distribution."""
    # safety check: if there is only 1 data point, CI is 0
    if len(data) < 2:
        return np.mean(data) if data else 0.0, 0.0
        
    mean = np.mean(data)
    sem = stats.sem(data)
    conf_bound = (1. + confidence) / 2.  # e.g. 0.975 for 95% CI
    ci = sem * stats.t.ppf(conf_bound, len(data) - 1)
    return mean, ci

async def generate_and_measure(req_id: str, args: argparse.Namespace, is_warmup: bool = False):
    prompt = f"Please write a highly detailed, extremely long essay about the history of artificial intelligence. Request ID: {req_id}"
    
    start_time = time.perf_counter()
    first_token_time = None
    chunk_count = 0
    
    try:
        stream = await client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=args.num_tokens, # generate this many tokens, since we set ignore_eos=True
            temperature=0.0,
            stream=True,
            extra_body={"ignore_eos": True}
        )
        
        async for chunk in stream:
            if first_token_time is None:
                first_token_time = time.perf_counter()
 
            if chunk.choices:
                chunk_count += 1

        end_time = time.perf_counter()
        if first_token_time is None:
            print(f"  Request {req_id} failed: No tokens received.")
            return None # request failed

        expected_chunks = args.num_tokens + 1
        assert chunk_count >= expected_chunks - 1, f"Expected {expected_chunks-1} chunks but got {chunk_count} for Req {req_id}"
        
        token_count = chunk_count - 1  # last chunk is an end chunk (not actual token)
        
        # metrics
        ttft = first_token_time - start_time
        latency = end_time - start_time
        decode_time = end_time - first_token_time
        tpot = decode_time / max(1, (token_count - 1)) 
        throughput = token_count / latency  # tokens per second
        
        if not is_warmup:
            print(f"  Req {req_id} | TTFT: {ttft*1000:.1f}ms | TPOT: {tpot*1000:.1f}ms | Latency: {latency*1000:.1f}ms | Throughput: {throughput:.1f} tks/s")
            
        return ttft, tpot, latency, throughput, token_count
        
    except Exception as e:
        print(f"  Request {req_id} failed: {e}")
        return None

async def run_batch(run_id: str, args: argparse.Namespace, is_warmup: bool):
    batch_start = time.perf_counter()
    
    tasks = [
        generate_and_measure(f"{run_id}-{i}", args, is_warmup) 
        for i in range(args.concurrency)
    ]
    results = await asyncio.gather(*tasks)
    
    batch_end = time.perf_counter()
    batch_time = batch_end - batch_start
    
    # filter out failures
    valid_results = [r for r in results if r is not None]
    return valid_results, batch_time

async def main(args: argparse.Namespace):
    print(f"Starting {args.num_warmups} Warmup run(s)...")
    for w in range(args.num_warmups):
        await run_batch(f"warmup{w+1}", args, is_warmup=True)
    print("Warmups complete.\n")
    
    all_ttfts = []
    all_tpots = []
    all_latencies = []
    all_throughputs = []
    
    total_batch_time = 0.0
    total_tokens_generated = 0
    
    print(f"Starting {args.num_runs} Benchmark run(s) with Concurrency={args.concurrency}...")
    for r in range(args.num_runs):
        print(f"--- Run {r+1}/{args.num_runs} ---")
        results, batch_time = await run_batch(f"run{r+1}", args, is_warmup=False)
        
        total_batch_time += batch_time
        
        for ttft, tpot, latency, throughput, token_count in results:
            all_ttfts.append(ttft * 1000)       
            all_tpots.append(tpot * 1000)       
            all_latencies.append(latency * 1000) 
            all_throughputs.append(throughput)   
            total_tokens_generated += token_count

    if not all_ttfts:
        print("All requests failed. No metrics to display.")
        return

    # compute 95% confidence intervals
    mean_ttft, ci_ttft = compute_mean_and_ci_stats(all_ttfts)
    mean_tpot, ci_tpot = compute_mean_and_ci_stats(all_tpots)
    mean_latency, ci_latency = compute_mean_and_ci_stats(all_latencies)
    mean_throughput, ci_throughput = compute_mean_and_ci_stats(all_throughputs)

    print("\n" + "="*60)
    print("BENCHMARK CONFIGURATION")
    print(f"Model: {args.model}.  Num Tokens: {args.num_tokens}. Concurrency: {args.concurrency}. Warmup Runs: {args.num_warmups}. Benchmark Runs: {args.num_runs}")
    print("\n" + "="*60)
    print("BENCHMARK RESULTS (95% Confidence Intervals)")
    print("="*60)
    
    print(f"TTFT (Prefill) : {mean_ttft:.2f} ms ± {ci_ttft:.2f} ms")
    print(f"TPOT (Decode)  : {mean_tpot:.2f} ms ± {ci_tpot:.2f} ms")
    print(f"Latency (E2E)  : {mean_latency:.2f} ms ± {ci_latency:.2f} ms")
    print(f"Throughput/Req : {mean_throughput:.2f} tk/s ± {ci_throughput:.2f} tk/s")
    
    # system throughput 
    true_system_throughput = total_tokens_generated / total_batch_time if total_batch_time > 0 else 0
    print(f"Sys Throughput : {true_system_throughput:.2f} tokens/sec")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark mini sglang utilizing AsyncOpenAI.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="The model name to benchmark.")
    parser.add_argument("--num_tokens", type=int, default=256, help="Number of tokens to generate per request.")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests per run.")
    parser.add_argument("--num_warmups", type=int, default=1, help="Number of warmup runs.")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of actual measured benchmark runs.")
    
    args = parser.parse_args()
    
    asyncio.run(main(args))
