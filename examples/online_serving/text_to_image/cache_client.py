#!/usr/bin/env python3
"""
Qwen-Image online client with inter-request cache support.

Demonstrates:
1. First request: generates image, caches DiT state (with step latents)
2. Second request (same params): cache HIT, returns instantly
3. Third request (same prompt, resume_from_step=10): resumes from step 10

Usage:
    python cache_client.py --server http://localhost:8091

    # Or use individual modes:
    python cache_client.py --server http://localhost:8091 --mode first
    python cache_client.py --server http://localhost:8091 --mode reuse
    python cache_client.py --server http://localhost:8091 --mode resume --resume-from-step 10
"""

import argparse
import base64
import time
from pathlib import Path

import requests


def generate_image(
    server_url: str,
    prompt: str,
    seed: int = 42,
    negative_prompt: str | None = None,
    output: str = "output.png",
    resume_from_step: int | None = None,
) -> dict:
    payload: dict = {
        "prompt": prompt,
        "seed": seed,
        "response_format": "b64_json",
        "n": 1,
    }

    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if resume_from_step is not None and resume_from_step > 0:
        payload["resume_from_step"] = resume_from_step

    start = time.perf_counter()
    response = requests.post(
        f"{server_url}/v1/images/generations",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=300,
    )
    elapsed = time.perf_counter() - start

    response.raise_for_status()
    data = response.json()

    items = data.get("data", [])
    if items and items[0].get("b64_json"):
        image_bytes = base64.b64decode(items[0]["b64_json"])
        Path(output).write_bytes(image_bytes)
        return {"output": output, "time": elapsed, "size_kb": len(image_bytes) / 1024}
    else:
        print(f"Unexpected response: {data}")
        return {"output": None, "time": elapsed, "size_kb": 0}


def main():
    parser = argparse.ArgumentParser(description="Qwen-Image cache client")
    parser.add_argument("--server", "-s", default="http://localhost:8091", help="Server URL")
    parser.add_argument("--prompt", "-p", default="a cup of coffee on the table", help="Prompt")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--negative", default=None, help="Negative prompt")
    parser.add_argument("--output-dir", default="./online_outputs", help="Output directory")
    parser.add_argument(
        "--mode",
        choices=["all", "first", "reuse", "resume"],
        default="all",
        help="Mode: all (default), first, reuse, or resume",
    )
    parser.add_argument("--resume-from-step", type=int, default=10, help="Resume from step (resume mode)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("all", "first"):
        print(f"\n{'=' * 60}")
        print("Request 1: First generation (cache MISS, will store)")
        print(f"{'=' * 60}")
        result = generate_image(
            server_url=args.server,
            prompt=args.prompt,
            seed=args.seed,
            negative_prompt=args.negative,
            output=str(out_dir / "first.png"),
        )
        print(f"  Output: {result['output']}")
        print(f"  Time: {result['time']:.4f}s ({result['time'] * 1000:.0f}ms)")
        print(f"  Size: {result['size_kb']:.1f} KB")
        first_time = result["time"]

    if args.mode in ("all", "reuse"):
        print(f"\n{'=' * 60}")
        print("Request 2: Same request (expect cache HIT)")
        print(f"{'=' * 60}")
        result = generate_image(
            server_url=args.server,
            prompt=args.prompt,
            seed=args.seed,
            negative_prompt=args.negative,
            output=str(out_dir / "reuse.png"),
        )
        print(f"  Output: {result['output']}")
        print(f"  Time: {result['time']:.4f}s ({result['time'] * 1000:.0f}ms)")
        print(f"  Size: {result['size_kb']:.1f} KB")
        if args.mode == "all":
            speedup = first_time / result["time"] if result["time"] > 0 else float("inf")
            print(f"  Speedup: {speedup:.1f}x")
            if result["time"] < first_time * 0.5:
                print("  Cache HIT confirmed!")

    if args.mode in ("all", "resume"):
        print(f"\n{'=' * 60}")
        print(f"Request 3: Resume from step {args.resume_from_step}")
        print(f"{'=' * 60}")
        result = generate_image(
            server_url=args.server,
            prompt=args.prompt,
            seed=args.seed,
            negative_prompt=args.negative,
            resume_from_step=args.resume_from_step,
            output=str(out_dir / f"resume_step{args.resume_from_step}.png"),
        )
        print(f"  Output: {result['output']}")
        print(f"  Time: {result['time']:.4f}s ({result['time'] * 1000:.0f}ms)")
        print(f"  Size: {result['size_kb']:.1f} KB")
        if args.mode == "all":
            ratio = result["time"] / first_time if first_time > 0 else 0
            expected = 1 - args.resume_from_step / 50
            print(f"  Time ratio: {ratio:.2f} (expected ~{expected:.2f})")

    print(f"\nImages saved to {out_dir}/")


if __name__ == "__main__":
    main()
