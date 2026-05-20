from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.inter_request.cache_store import (
    CacheKey,
    DiTCacheStore,
    StepLatentData,
    build_cache_key_from_request,
)
from vllm_omni.diffusion.cache.inter_request.step_recorder import StepLatentsRecorder
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
    - Step latents: Intermediate latents at every denoising step (for future
      partial-resume capability)

    A :class:`StepLatentsRecorder` is always attached to the pipeline to capture
    intermediate latents during denoising.  These are stored in the in-memory
    cache alongside the final latent.  When ``record_step_latents`` is enabled,
    the step latents are additionally saved to disk.

    Usage:
        omni = Omni(
            model="Qwen/Qwen-Image",
            cache_backend="inter_request",
            cache_config={
                "inter_request_max_entries": 100,
                "inter_request_max_memory_gb": 4.0,
                "inter_request_record_step_latents": True,
                "inter_request_step_latents_dir": "./step_latents",
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

        self._record_step_latents = getattr(config, "inter_request_record_step_latents", False)
        self._step_latents_dir = getattr(config, "inter_request_step_latents_dir", "./step_latents")
        self._persistent_cache_dir = getattr(config, "inter_request_persistent_cache_dir", None)
        self._recorder: StepLatentsRecorder | None = None

        logger.info(
            "InterRequestCacheBackend initialized: max_entries=%d, max_memory_gb=%.1f, record_step_latents=%s, persistent_cache_dir=%s",
            max_entries,
            max_memory_gb,
            self._record_step_latents,
            self._persistent_cache_dir,
        )

    def enable(self, pipeline: Any) -> None:
        self._pipeline = pipeline
        self.enabled = True
        self._recorder = StepLatentsRecorder()
        pipeline._step_latents_recorder = self._recorder

        if self._persistent_cache_dir is not None:
            loaded = self._cache_store.load_from_disk(self._persistent_cache_dir)
            if loaded > 0:
                logger.info(
                    "Loaded %d cache entries from persistent storage %s",
                    loaded,
                    self._persistent_cache_dir,
                )

        logger.info(
            "InterRequestCacheBackend enabled on pipeline %s (save_step_latents_to_disk=%s)",
            pipeline.__class__.__name__,
            self._record_step_latents,
        )

    def shutdown(self) -> None:
        logger.info(
            "InterRequestCacheBackend shutdown: persistent_cache_dir=%s, cache_size=%d",
            self._persistent_cache_dir,
            self._cache_store.size,
        )
        if self._persistent_cache_dir is not None and self._cache_store.size > 0:
            saved = self._cache_store.save_to_disk(self._persistent_cache_dir)
            logger.info(
                "Persisted %d cache entries to %s",
                saved,
                self._persistent_cache_dir,
            )

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        pass

    def before_forward(self, is_dummy: bool = False) -> None:
        if self._recorder is not None and not is_dummy:
            self._recorder.clear()
            self._recorder.enable()

    def after_forward(self, cache_key_hash: str | None = None, is_dummy: bool = False) -> list[str] | None:
        if self._recorder is None or is_dummy:
            return None

        self._recorder.disable()

        if not self._record_step_latents:
            self._recorder.clear()
            return None

        if self._recorder.num_steps == 0:
            return None

        save_dir = Path(self._step_latents_dir)
        if cache_key_hash is not None:
            save_dir = save_dir / cache_key_hash

        saved_paths = self._recorder.save(save_dir)
        self._recorder.clear()
        return saved_paths

    def lookup(self, req: Any, target_device: torch.device | str | None = None) -> torch.Tensor | None:
        if not self.enabled or self._pipeline is None:
            return None

        cache_key = build_cache_key_from_request(req, self._pipeline)
        if cache_key is None:
            return None

        return self._cache_store.get(cache_key, target_device=target_device)

    def lookup_step_latents(
        self, req: Any, target_device: torch.device | str | None = None
    ) -> list[StepLatentData] | None:
        if not self.enabled or self._pipeline is None:
            return None

        cache_key = build_cache_key_from_request(req, self._pipeline)
        if cache_key is None:
            return None

        return self._cache_store.get_step_latents(cache_key, target_device=target_device)

    def store(
        self,
        req: Any,
        latents: torch.Tensor,
        step_latents: list[StepLatentData] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        if not self.enabled or self._pipeline is None:
            return None

        cache_key = build_cache_key_from_request(req, self._pipeline)
        if cache_key is None:
            return None

        self._cache_store.put(cache_key, latents, step_latents=step_latents, metadata=metadata)
        return cache_key.to_hash()

    @property
    def cache_store(self) -> DiTCacheStore:
        return self._cache_store

    @property
    def recorder(self) -> StepLatentsRecorder | None:
        return self._recorder

    def stats(self) -> dict[str, Any]:
        return self._cache_store.stats()

    def clear(self) -> None:
        self._cache_store.clear()
