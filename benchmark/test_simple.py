import asyncio
import os
import random
import sys
from typing import List

from minisgl.benchmark.utils import (
    benchmark_batch,
    benchmark_one,
    generate_message,
    process_benchmark_results,
)
from minisgl.utils import init_logger
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer

MODEL_PATH = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct")

logger = init_logger(__name__)


async def main():
    try:
        random.seed(42)
        model = MODEL_PATH
        tokenizer = AutoTokenizer.from_pretrained(model)
        print(f"Loaded tokenizer from {model}")

        MAX_LENGTH = 8192

        async def generate_task(max_bs: int) -> List[str]:
            """Generate a list of tasks with random lengths."""
            result = []
            for _ in range(max_bs):
                length = random.randint(1, MAX_LENGTH)
                message = generate_message(tokenizer, length)
                result.append(message)
                await asyncio.sleep(0)
            return result

        TEST_BS = [16]
        PORT = 1919

        # Create the async client
        async with OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="") as client:
            logger.info("Testing connection to server...")

            # Test connection with a simple request first
            try:
                gen_task = asyncio.create_task(generate_task(max(TEST_BS)))
                test_msg = generate_message(tokenizer, 100)
                test_result = await benchmark_one(client, test_msg, 1, model)
                if len(test_result.tics) < 2:
                    logger.info("Server connection test failed")
                    return
                logger.info("Server connection successful")
            except Exception as e:
                logger.warning("Server connection failed")
                logger.warning(f"Make sure the server is running on http://127.0.0.1:{PORT}")
                raise e

            msgs = await gen_task
            logger.info(f"Generated {len(msgs)} test messages")

            logger.info("Running benchmark...")
            for batch_size in TEST_BS:
                try:
                    results = await benchmark_batch(client, msgs[:batch_size], 1024, model)
                    process_benchmark_results(results)
                except Exception as e:
                    logger.info(f"Error with batch size {batch_size}: {e}")
                    continue
            logger.info("Benchmark completed.")

    except Exception as e:
        print(f"Error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
