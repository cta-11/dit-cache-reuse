from __future__ import annotations

import logging
from typing import Any

import torch

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.inter_request.cache_store import (
    CacheKey,
    DiTCacheStore,
    build_cache_key_from_request,
)
from vllm_omni.diffusion.data import DiffusionCacheConfig

logger = logging.getLogger(__name__)


class InterRequestCacheBackend(CacheBackend):
    """
    Inter-request cache backend for DiT full reuse (Chorus Stage-1).

    This backend implements the Stage-1 caching strategy from the Chorus paper:
    when two requests have identical inputs (same prompt, dimensions, seed, etc.),
    the DiT computation can be entirely skipped by reusing cached latent features
    from a previous request.

    Unlike intra-request caching backends (cache_dit, TeaCache) that optimize
    within a single denoising process, this backend caches the final latents
    across different requests, enabling complete DiT computation reuse.

    The cache stores:
    - Key: Hash of all inputs that determine the DiT output (prompt, seed, etc.)
    - Value: Final latents after all denoising steps (before VAE decode)

    Usage:
        omni = Omni(
            model="Qwen/Qwen-Image",
            cache_backend="inter_request",
            cache_config={
                "inter_request_max_entries": 100,
                "inter_request_max_memory_gb": 4.0,
            }
        )
    """

    def __init__(self, config: DiffusionCacheConfig):
        super().__init__(config)
        max_entries = getattr(config, "inter_request_max_entries", 100)
        max_memory_gb = getattr(config, "inter_request_max_memory_gb", 4.0)
        self._cache_store = DiTCacheStore(
            max_entries=max_entries,
            max_memory_gb=max_memory_gb,
        )
        self._pipeline = None
        logger.info(
            "InterRequestCacheBackend initialized: max_entries=%d, max_memory_gb=%.1f",
            max_entries,
            max_memory_gb,
        )

    def enable(self, pipeline: Any) -> None:
        self._pipeline = pipeline
        self.enabled = True
        logger.info(
            "InterRequestCacheBackend enabled on pipeline %s",
            pipeline.__class__.__name__,
        )

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        pass

    def lookup(self, req: Any, target_device: torch.device | str | None = None) -> torch.Tensor | None:
        if not self.enabled or self._pipeline is None:
            return None

        cache_key = build_cache_key_from_request(req, self._pipeline)
        if cache_key is None:
            return None

        return self._cache_store.get(cache_key, target_device=target_device)

    def store(self, req: Any, latents: torch.Tensor, metadata: dict[str, Any] | None = None) -> None:
        if not self.enabled or self._pipeline is None:
            return

        cache_key = build_cache_key_from_request(req, self._pipeline)
        if cache_key is None:
            return

        self._cache_store.put(cache_key, latents, metadata=metadata)

    @property
    def cache_store(self) -> DiTCacheStore:
        return self._cache_store

    def stats(self) -> dict[str, Any]:
        return self._cache_store.stats()

    def clear(self) -> None:
        self._cache_store.clear()
