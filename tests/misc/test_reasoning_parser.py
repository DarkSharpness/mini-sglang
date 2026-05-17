#!/usr/bin/env python3
"""
test_reasoning_parser.py
Send several requests to a running mini-sglang engine (with --reasoning-parser enabled),
to verify whether reasoning_content / content are split correctly.

Usage:
    python test_reasoning_parser.py [--host 127.0.0.1] [--port 1919]

Dependency: requests (install via: pip install requests)
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from typing import Iterator

import requests

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"
RED    = "\033[31m"
DIM    = "\033[2m"

def _c(text: str, *codes: str) -> str:
    if sys.stdout.isatty():
        return "".join(codes) + text + RESET
    return text

def _iter_sse_chunks(resp: requests.Response) -> Iterator[dict]:
    """Parse text/event-stream line by line and yield JSON chunks."""
    for raw_line in resp.iter_lines():
        if isinstance(raw_line, bytes):
            raw_line = raw_line.decode()
        line = raw_line.strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue

# ──────────────────────────────────────────────────────────────────────────────
# Single Request
# ──────────────────────────────────────────────────────────────────────────────
def chat_stream(
    base_url: str,
    messages: list[dict],
    model: str,
    max_tokens: int = 1024,
    temperature: float = 0.6,
) -> tuple[str, str]:
    """
    Send a streaming /v1/chat/completions request.
    Returns (reasoning_text, normal_text) as complete strings.
    """
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    reasoning_buf = ""
    normal_buf    = ""

    with requests.post(url, json=payload, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for chunk in _iter_sse_chunks(resp):
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            reasoning_buf += delta.get("reasoning_content") or ""
            normal_buf    += delta.get("content")           or ""

    return reasoning_buf, normal_buf

_W = 72

def _hr(char: str = "─") -> str:
    return _c(char * _W, DIM)

def _section(title: str) -> None:
    print()
    print(_c(f"{'─' * 3} {title} {'─' * (_W - 5 - len(title))}", BOLD + CYAN))

def _print_result(reasoning: str, normal: str) -> None:
    if reasoning:
        print(_c("  [reasoning_content]", YELLOW))
        wrapped = textwrap.fill(reasoning.strip(), width=_W - 4,
                                initial_indent="    ", subsequent_indent="    ")
        print(_c(wrapped, YELLOW))
    else:
        print(_c("  [reasoning_content]  (Empty — model might not have output <think> tags)", DIM))

    print()
    print(_c("  [content]", GREEN))
    wrapped = textwrap.fill(normal.strip() or "(Empty)", width=_W - 4,
                            initial_indent="    ", subsequent_indent="    ")
    print(_c(wrapped, GREEN))

def _check(label: str, cond: bool) -> None:
    icon = _c("✓", GREEN + BOLD) if cond else _c("✗", RED + BOLD)
    print(f"  {icon}  {label}")

CASES: list[dict] = [
    {
        "name": "Simple Math Reasoning",
        "messages": [
            {"role": "user", "content": "Which is larger, 9.11 or 9.9? Please think carefully before answering."},
        ],
        "max_tokens": 4096,
    },
    {
        "name": "Logic Puzzle",
        "messages": [
            {
                "role": "user",
                "content": (
                    "There are 3 light bulbs in a room and 3 switches outside. "
                    "Each switch controls one bulb. You can only enter the room once. "
                    "How do you determine which switch controls which bulb?"
                ),
            }
        ],
        "max_tokens": 4096,
    },
    {
        "name": "Code Generation",
        "messages": [
            {
                "role": "user",
                "content": "Write a concise Python function to check if a number is prime.",
            }
        ],
        "max_tokens": 4096,
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Main Flow
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Test mini-sglang reasoning parser")
    parser.add_argument("--host",       default="127.0.0.1")
    parser.add_argument("--port",       default=1919, type=int)
    parser.add_argument("--max-tokens", default=None, type=int,
                        help="Override max_tokens for all cases")
    parser.add_argument("--temperature", default=0.6, type=float)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    print(_c(f"\nmini-sglang reasoning-parser test script", BOLD))
    print(_c(f"Target: {base_url}", DIM))
    print(_hr())

    try:
        r = requests.get(f"{base_url}/v1/models", timeout=5)
        r.raise_for_status()
        model_name = r.json()["data"][0]["id"]
        print(f"  Service Online  model = {_c(model_name, BOLD)}")
    except Exception as exc:
        print(_c(f"  ✗ Cannot connect to server: {exc}", RED))
        sys.exit(1)

    passed = 0
    failed = 0

    for i, case in enumerate(CASES, 1):
        max_tokens = args.max_tokens or case.get("max_tokens", 512)
        _section(f"[{i}/{len(CASES)}] {case['name']}")

        user_msg = case["messages"][-1]["content"]
        print(_c(f"  Q: {textwrap.shorten(user_msg, 80)}", DIM))
        print()

        try:
            reasoning, normal = chat_stream(
                base_url,
                case["messages"],
                model=model_name,
                max_tokens=max_tokens,
                temperature=args.temperature,
            )
        except requests.HTTPError as exc:
            print(_c(f"  HTTP Error: {exc}", RED))
            failed += 1
            continue
        except Exception as exc:
            print(_c(f"  Request Failed: {exc}", RED))
            failed += 1
            continue

        _print_result(reasoning, normal)
        print()

        _check("Received non-empty output", bool(reasoning or normal))
        _check("reasoning_content is not empty (think tags identified)", bool(reasoning))
        _check("content is not empty (normal response preserved)", bool(normal))
        _check("reasoning_content does not contain </think>", "</think>" not in reasoning)
        _check("content does not contain <think>", "<think>" not in normal)

        case_ok = all([bool(reasoning or normal), bool(reasoning), bool(normal),
                       "</think>" not in reasoning, "<think>" not in normal])
        if case_ok:
            passed += 1
        else:
            failed += 1

    print()
    print(_hr("═"))
    total = passed + failed
    summary_color = GREEN if failed == 0 else RED
    print(_c(f"  Result: {passed}/{total} passed", BOLD + summary_color))
    if failed > 0:
        print(_c(
            "\n  Tip: If reasoning_content is empty, ensure:\n"
            "    1. The engine was started with --reasoning-parser\n"
            "    2. You are using a reasoning model (e.g., Qwen-R1, DeepSeek-R1)\n"
            "    3. Temperature > 0 (temperature=0 might cause the model to skip <think>)",
            YELLOW,
        ))
    print()

if __name__ == "__main__":
    main()