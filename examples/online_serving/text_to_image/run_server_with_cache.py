#!/usr/bin/env python3
"""
Simple HTTP server wrapping Omni with inter-request cache support.

This bypasses the vllm serve CLI to avoid version compatibility issues.

Usage:
    ASCEND_RT_VISIBLE_DEVICES=0,1 python run_server_with_cache.py

Endpoints:
    POST /v1/images/generations  - Generate image (with cache support)
    GET  /health                  - Health check
"""

import argparse
import base64
import io
import json
import logging
import random
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

omni = None
SAMPLING_PARAMS_CLS = None


class CacheAwareHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        logger.info("%s - %s", self.client_address[0], format % args)

    def _send_json(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/v1/images/generations":
            self._handle_generate()
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_generate(self):
        global omni, SAMPLING_PARAMS_CLS
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)
        try:
            req = json.loads(body)
        except json.JSONDecodeError as e:
            self._send_json(400, {"error": f"invalid JSON: {e}"})
            return

        prompt = req.get("prompt", "")
        negative_prompt = req.get("negative_prompt")
        seed = req.get("seed", random.randint(0, 2**32 - 1))
        width = req.get("width")
        height = req.get("height")
        size_str = req.get("size")
        if size_str and "x" in str(size_str):
            parts = str(size_str).split("x")
            if width is None:
                width = int(parts[0])
            if height is None:
                height = int(parts[1])
        steps = req.get("num_inference_steps", 50)
        true_cfg_scale = req.get("true_cfg_scale", 4.0)
        resume_from_step = req.get("resume_from_step", 0) or 0

        prompt_dict = {"prompt": prompt}
        if negative_prompt:
            prompt_dict["negative_prompt"] = negative_prompt

        sampling_params = SAMPLING_PARAMS_CLS(
            height=height,
            width=width,
            seed=seed,
            true_cfg_scale=true_cfg_scale,
            num_inference_steps=steps,
            num_outputs_per_prompt=1,
            resume_from_step=resume_from_step,
        )

        start = time.perf_counter()
        try:
            outputs = omni.generate(prompt_dict, sampling_params)
        except Exception as e:
            logger.error("Generation failed: %s", e, exc_info=True)
            self._send_json(500, {"error": str(e)})
            return
        elapsed = time.perf_counter() - start
        logger.info("Generation completed in %.2fs", elapsed)

        images = []
        for out in outputs:
            inner = out.request_output
            if inner and inner.images:
                for img in inner.images:
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                    images.append({"b64_json": b64})

        self._send_json(200, {
            "created": int(time.time()),
            "data": images,
            "time_ms": elapsed * 1000,
        })


def main():
    global omni, SAMPLING_PARAMS_CLS

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/mnt/shared/models/Qwen-Image")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--cache-backend", default="inter_request")
    parser.add_argument("--persistent-cache-dir", default="./persistent_cache")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-entries", type=int, default=100)
    parser.add_argument("--max-memory-gb", type=float, default=4.0)
    parser.add_argument("--clip-model-path", default=None, help="Path to CLIP model for semantic matching")
    parser.add_argument("--clip-threshold", type=float, default=0.75, help="CLIP similarity threshold (tau)")
    parser.add_argument("--clip-min-skip", type=int, default=5, help="Minimum skip steps when similarity just exceeds threshold")
    parser.add_argument("--clip-max-skip-ratio", type=float, default=0.5, help="Max skip ratio of total steps when similarity=1.0")
    args = parser.parse_args()

    from vllm_omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    SAMPLING_PARAMS_CLS = OmniDiffusionSamplingParams

    cache_config = {
        "inter_request_max_entries": args.max_entries,
        "inter_request_max_memory_gb": args.max_memory_gb,
        "inter_request_persistent_cache_dir": args.persistent_cache_dir,
    }
    if args.clip_model_path:
        cache_config["inter_request_clip_model_path"] = args.clip_model_path
        cache_config["inter_request_clip_threshold"] = args.clip_threshold
        cache_config["inter_request_clip_min_skip"] = args.clip_min_skip
        cache_config["inter_request_clip_max_skip_ratio"] = args.clip_max_skip_ratio

    logger.info("Initializing Omni engine...")
    logger.info("  Model: %s", args.model)
    logger.info("  Cache backend: %s", args.cache_backend)
    logger.info("  Persistent cache dir: %s", args.persistent_cache_dir)

    omni = Omni(
        model=args.model,
        cache_backend=args.cache_backend,
        cache_config=cache_config,
        tensor_parallel_size=args.tensor_parallel_size,
        mode="text-to-image",
        init_timeout=1200,
    )

    server = HTTPServer((args.host, args.port), CacheAwareHandler)
    logger.info("Server listening on %s:%d", args.host, args.port)
    logger.info("POST /v1/images/generations - Generate image")
    logger.info("GET  /health - Health check")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.server_close()
        omni.shutdown()


if __name__ == "__main__":
    main()
