from __future__ import annotations

import argparse
import json

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one minimal OLMo3 API smoke")
    parser.add_argument("--base-url", default="http://127.0.0.1:1919")
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_response = requests.get(f"{args.base_url}/v1/models", timeout=30)
    model_response.raise_for_status()
    models = model_response.json()["data"]
    assert models and models[0]["id"] == args.model

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": "Answer with one word: what is the capital of France?",
            }
        ],
        "temperature": 0,
        "max_tokens": 8,
    }
    response = requests.post(
        f"{args.base_url}/v1/chat/completions", json=payload, timeout=60
    )
    response.raise_for_status()
    body = response.json()
    choice = body["choices"][0]
    content = choice["message"]["content"]
    assert choice["finish_reason"] == "stop"
    assert "Paris" in content
    assert "<|im_end|>" not in content and "<|endoftext|>" not in content

    stream_response = requests.post(
        f"{args.base_url}/v1/chat/completions",
        json={**payload, "stream": True},
        timeout=60,
        stream=True,
    )
    stream_response.raise_for_status()
    chunks = []
    done_count = 0
    for line in stream_response.iter_lines(decode_unicode=True):
        if not line:
            continue
        assert line.startswith("data: ")
        data = line.removeprefix("data: ")
        if data == "[DONE]":
            done_count += 1
        else:
            chunks.append(json.loads(data))

    assert done_count == 1
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    streamed_content = "".join(
        chunk["choices"][0]["delta"].get("content", "") for chunk in chunks
    )
    assert streamed_content == content
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    print(f"OLMO3_API_SMOKE=passed content={content!r}")


if __name__ == "__main__":
    main()
