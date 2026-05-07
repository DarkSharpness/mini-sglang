import asyncio
import time
import numpy as np
from openai import AsyncOpenAI

# BENCHMARK CONFIGURATION
API_BASE = "http://127.0.0.1:1920/v1"
API_KEY = "EMPTY"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

MAX_TOKENS = 128
CONCURRENCY = 1      # number of concurrent requests per run (Batch Size)
NUM_WARMUPS = 1      # number of warmup runs
NUM_RUNS = 2         # number of actual measured runs

client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)

async def generate_and_measure(req_id: str, is_warmup: bool = False):
    prompt = f"Please write a highly detailed, extremely long essay about the history of artificial intelligence. Request ID: {req_id}"
    
    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0
    
    try:
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            stream=True,
        )
        
        async for chunk in stream:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            
            if chunk.choices and chunk.choices[0].delta.content:
                token_count += 1
                
        end_time = time.perf_counter()
        if first_token_time is None:
            print(f"  Request {req_id} failed: No tokens received.")
            return None # request failed

        assert token_count >= MAX_TOKENS-1, f"Expected {MAX_TOKENS-1} tokens but got {token_count}"

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

async def run_batch(run_id: str, is_warmup: bool):
    batch_start = time.perf_counter()
    
    tasks = [
        generate_and_measure(f"{run_id}-{i}", is_warmup) 
        for i in range(CONCURRENCY)
    ]
    results = await asyncio.gather(*tasks)
    
    batch_end = time.perf_counter()
    batch_time = batch_end - batch_start
    
    # filter out failures
    valid_results = [r for r in results if r is not None]
    return valid_results, batch_time

async def main():
    print(f"Starting {NUM_WARMUPS} Warmup run(s)...")
    for w in range(NUM_WARMUPS):
        await run_batch(f"warmup{w+1}", is_warmup=True)
    print("Warmups complete.\n")
    
    all_ttfts = []
    all_tpots = []
    all_latencies = []
    all_throughputs = []
    
    total_batch_time = 0.0
    total_tokens_generated = 0
    
    print(f"Starting {NUM_RUNS} Benchmark run(s) with Concurrency={CONCURRENCY}...")
    for r in range(NUM_RUNS):
        print(f"--- Run {r+1}/{NUM_RUNS} ---")
        results, batch_time = await run_batch(f"run{r+1}", is_warmup=False)
        
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

    print("\n" + "="*50)
    print(f"BENCHMARK RESULTS (Runs={NUM_RUNS}, Concurrency={CONCURRENCY}, MaxTokens={MAX_TOKENS})")
    print("="*50)
    
    print(f"TTFT (Prefill) : {np.mean(all_ttfts):.2f} ms ± {np.std(all_ttfts):.2f} ms")
    print(f"TPOT (Decode)  : {np.mean(all_tpots):.2f} ms ± {np.std(all_tpots):.2f} ms")
    print(f"Latency (E2E)  : {np.mean(all_latencies):.2f} ms ± {np.std(all_latencies):.2f} ms")
    print(f"Throughput/Req : {np.mean(all_throughputs):.2f} tk/s ± {np.std(all_throughputs):.2f} tk/s")
    
    # system throughput 
    true_system_throughput = total_tokens_generated / total_batch_time
    print(f"Sys Throughput : {true_system_throughput:.2f} tokens/sec")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
