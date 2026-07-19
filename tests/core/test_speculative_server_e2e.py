"""Manual online checks against an already-running speculative server.

These tests cover scheduler behavior that the offline token-equivalence test
cannot: serving greedy and sampled requests concurrently, handling a client
disconnect, and continuing to serve afterward.
"""

from __future__ import annotations

import argparse
import asyncio

import httpx
from openai import AsyncOpenAI


async def _mixed_traffic(base_url: str) -> None:
    """Verify greedy speculation and normal sampled decoding can run together."""
    async with AsyncOpenAI(base_url=f"{base_url}/v1", api_key="dummy") as client:
        # Discover the configured model so the test works with any server launch.
        model = (await client.models.list()).data[0].id

        async def generate(temperature: float) -> str:
            # Only temperature=0 is eligible for n-gram speculation. Sending both
            # requests concurrently exercises the scheduler's verify/decode split.
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Explain paged attention briefly."}],
                max_tokens=32,
                temperature=temperature,
                extra_body={"top_k": 1 if temperature == 0 else -1},
            )
            return response.choices[0].message.content or ""

        greedy, sampled = await asyncio.gather(generate(0.0), generate(0.7))
        # This is a liveness/integration check, not an output-correctness comparison.
        assert greedy and sampled, "one or both concurrent requests returned empty output"


async def _abort_and_recover(base_url: str) -> None:
    """Disconnect mid-generation, then verify the server remains usable."""
    # A long request gives the client enough time to disconnect while decoding.
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
                    # Exit after the first token; closing the stream simulates abort.
                    break
        # Give the disconnect detector time to enqueue AbortMsg.
        await asyncio.sleep(0.5)

        followup = {
            "prompt": "Reply with one short sentence.",
            "max_tokens": 8,
            "ignore_eos": True,
        }
        saw_output = False
        # Successful follow-up output proves abort cleanup did not strand scheduler
        # state, KV pages, or the request loop.
        async with client.stream("POST", f"{base_url}/generate", json=followup) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                saw_output |= line.startswith("data:") and "[DONE]" not in line
        assert saw_output, "follow-up request produced no output after client abort"


async def run(port: int) -> None:
    """Run both online scheduler contracts against the selected local port."""
    base_url = f"http://127.0.0.1:{port}"
    print("[1/2] Concurrent greedy + sampled requests...", flush=True)
    await _mixed_traffic(base_url)
    print("      PASS: both request types completed with non-empty output", flush=True)
    print("[2/2] Client abort + follow-up recovery...", flush=True)
    await _abort_and_recover(base_url)
    print("      PASS: server remained usable after abort cleanup", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=1919)
    args = parser.parse_args()
    asyncio.run(run(args.port))


if __name__ == "__main__":
    main()
