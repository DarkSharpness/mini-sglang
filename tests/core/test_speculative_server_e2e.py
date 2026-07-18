"""Manual online integration checks for mixed traffic and request abort."""

from __future__ import annotations

import argparse
import asyncio

import httpx
from openai import AsyncOpenAI


async def _mixed_traffic(base_url: str) -> None:
    async with AsyncOpenAI(base_url=f"{base_url}/v1", api_key="dummy") as client:
        model = (await client.models.list()).data[0].id

        async def generate(temperature: float) -> str:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Explain paged attention briefly."}],
                max_tokens=32,
                temperature=temperature,
                extra_body={"top_k": 1 if temperature == 0 else -1},
            )
            return response.choices[0].message.content or ""

        greedy, sampled = await asyncio.gather(generate(0.0), generate(0.7))
        assert greedy and sampled


async def _abort_and_recover(base_url: str) -> None:
    payload = {
        "prompt": "Write a very long explanation of speculative decoding.",
        "max_tokens": 1024,
        "ignore_eos": True,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{base_url}/generate", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data:") and "[DONE]" not in line:
                    break
        # Give the disconnect detector time to enqueue AbortMsg.
        await asyncio.sleep(0.5)

        followup = {
            "prompt": "Reply with one short sentence.",
            "max_tokens": 8,
            "ignore_eos": True,
        }
        saw_output = False
        async with client.stream(
            "POST", f"{base_url}/generate", json=followup
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                saw_output |= line.startswith("data:") and "[DONE]" not in line
        assert saw_output


async def run(port: int) -> None:
    base_url = f"http://127.0.0.1:{port}"
    await _mixed_traffic(base_url)
    await _abort_and_recover(base_url)
    print("PASS: mixed greedy/sampled traffic and abort recovery")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=1919)
    args = parser.parse_args()
    asyncio.run(run(args.port))


if __name__ == "__main__":
    main()
