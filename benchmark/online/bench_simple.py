import asyncio
import random
import sys
from typing import List

from minisgl.benchmark.client import (
    benchmark_one,
    benchmark_one_batch,
    generate_prompt,
    get_model_name,
    process_benchmark_results,
)
from minisgl.utils import init_logger
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer

logger = init_logger(__name__)


async def main():
    try:
        random.seed(42)  # reproducibility

        async def generate_task(max_bs: int) -> List[str]:
            """Generate a list of tasks with random lengths."""
            result = []
            for _ in range(max_bs):
                length = random.randint(1, MAX_INPUT)
                message = generate_prompt(tokenizer, length)
                result.append(message)
                await asyncio.sleep(0)
            return result

        TEST_BS = [64]
        PORT = 1919
        MAX_INPUT = 8192
        # Create the async client
        async with OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="") as client:
            MODEL = await get_model_name(client)
            tokenizer = AutoTokenizer.from_pretrained(MODEL)

            logger.info(f"Loaded tokenizer from {MODEL}")
            logger.info("Testing connection to server...")

            # Test connection with a simple request first
            try:
                gen_task = asyncio.create_task(generate_task(max(TEST_BS)))
                test_msg = generate_prompt(tokenizer, 100)
                test_result = await benchmark_one(client, test_msg, 2, MODEL, pbar=False)
                if len(test_result.tics) <= 2:
                    logger.info("Server connection test failed")
                    return
                logger.info("Server connection successful")
            except Exception as e:
                logger.warning("Server connection failed")
                logger.warning(f"Make sure the server is running on http://127.0.0.1:{PORT}")
                raise e from e

            msgs = await gen_task
            output_lengths = [random.randint(16, 1024) for _ in range(max(TEST_BS))]
            logger.info(f"Generated {len(msgs)} test messages")

            logger.info("Running benchmark...")
            for batch_size in TEST_BS:
                try:
                    results = await benchmark_one_batch(
                        client, msgs[:batch_size], output_lengths[:batch_size], MODEL
                    )
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
