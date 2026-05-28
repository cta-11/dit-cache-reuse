from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheKey:
    prompt: str
    negative_prompt: str
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    true_cfg_scale: float
    seed: int
    sigmas: tuple[float, ...] | None
    max_sequence_length: int | None
    num_images_per_prompt: int

    def to_hash(self) -> str:
        data = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "height": self.height,
            "width": self.width,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "true_cfg_scale": self.true_cfg_scale,
            "seed": self.seed,
            "sigmas": list(self.sigmas) if self.sigmas is not None else None,
            "max_sequence_length": self.max_sequence_length,
            "num_images_per_prompt": self.num_images_per_prompt,
        }
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()


@dataclass
class StepLatentData:
    step_index: int
    timestep: float
    latent: torch.Tensor


@dataclass
class CacheEntry:
    latents: torch.Tensor
    cache_key_hash: str
    step_latents: list[StepLatentData] | None = None
    metadata: dict[str, Any] | None = None
    clip_embedding: torch.Tensor | None = None
    cache_key: CacheKey | None = None


class DiTCacheStore:
    def __init__(self, max_entries: int = 100, max_memory_gb: float = 4.0):
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_entries = max_entries
        self._max_memory_bytes = max_memory_gb * 1024**3
        self._current_memory_bytes = 0
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _estimate_tensor_bytes(self, tensor: torch.Tensor) -> int:
        return tensor.nelement() * tensor.element_size()

    def _estimate_step_latents_bytes(self, step_latents: list[StepLatentData]) -> int:
        return sum(self._estimate_tensor_bytes(s.latent) for s in step_latents)

    def _estimate_entry_bytes(self, entry: CacheEntry) -> int:
        total = self._estimate_tensor_bytes(entry.latents)
        if entry.step_latents is not None:
            total += self._estimate_step_latents_bytes(entry.step_latents)
        return total

    def _evict_if_needed(self, required_bytes: int):
        while (
            len(self._store) >= self._max_entries
            or (self._current_memory_bytes + required_bytes > self._max_memory_bytes and len(self._store) > 0)
        ):
            oldest_key, oldest_entry = self._store.popitem(last=False)
            freed = self._estimate_entry_bytes(oldest_entry)
            self._current_memory_bytes -= freed
            logger.debug(
                "Evicted cache entry %s, freed %.2f MB",
                oldest_key[:8],
                freed / 1024**2,
            )

    def put(
        self,
        key: CacheKey,
        latents: torch.Tensor,
        step_latents: list[StepLatentData] | None = None,
        metadata: dict[str, Any] | None = None,
        clip_embedding: torch.Tensor | None = None,
    ):
        key_hash = key.to_hash()
        tensor_bytes = self._estimate_tensor_bytes(latents)
        if step_latents is not None:
            tensor_bytes += self._estimate_step_latents_bytes(step_latents)

        with self._lock:
            if key_hash in self._store:
                old_entry = self._store[key_hash]
                self._current_memory_bytes -= self._estimate_entry_bytes(old_entry)
                del self._store[key_hash]

            self._evict_if_needed(tensor_bytes)

            cached_latents = latents.detach().clone().cpu()
            cached_step_latents = None
            if step_latents is not None:
                cached_step_latents = [
                    StepLatentData(
                        step_index=s.step_index,
                        timestep=s.timestep,
                        latent=s.latent.detach().clone().cpu(),
                    )
                    for s in step_latents
                ]
            cached_clip = clip_embedding.detach().clone().cpu() if clip_embedding is not None else None
            entry = CacheEntry(
                latents=cached_latents,
                cache_key_hash=key_hash,
                step_latents=cached_step_latents,
                metadata=metadata,
                clip_embedding=cached_clip,
                cache_key=key,
            )
            self._store[key_hash] = entry
            self._current_memory_bytes += tensor_bytes

            num_steps = len(cached_step_latents) if cached_step_latents is not None else 0
            logger.info(
                "Cached DiT state for key %s, size %.2f MB (final + %d steps), total cache %.2f MB / %d entries",
                key_hash[:8],
                tensor_bytes / 1024**2,
                num_steps,
                self._current_memory_bytes / 1024**2,
                len(self._store),
            )

    def get(self, key: CacheKey, target_device: torch.device | str | None = None) -> torch.Tensor | None:
        key_hash = key.to_hash()

        with self._lock:
            entry = self._store.get(key_hash)
            if entry is None:
                self._misses += 1
                logger.debug("Cache MISS for key %s", key_hash[:8])
                return None

            self._store.move_to_end(key_hash)
            self._hits += 1

            latents = entry.latents
            if target_device is not None:
                latents = latents.to(device=target_device)
            else:
                latents = latents.clone()

            logger.info(
                "Cache HIT for key %s (hits=%d, misses=%d, hit_rate=%.2f%%)",
                key_hash[:8],
                self._hits,
                self._misses,
                self.hit_rate * 100,
            )
            return latents

    def semantic_search(
        self,
        query_embedding: torch.Tensor,
        threshold: float = 0.75,
        target_device: torch.device | str | None = None,
        required_height: int | None = None,
        required_width: int | None = None,
        required_num_inference_steps: int | None = None,
    ) -> tuple[torch.Tensor | None, list[StepLatentData] | None, float]:
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        query_norm = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        best_sim = 0.0
        best_key_hash = None

        with self._lock:
            for key_hash, entry in self._store.items():
                if entry.clip_embedding is None:
                    continue
                if entry.cache_key is not None:
                    if required_height is not None and entry.cache_key.height != required_height:
                        continue
                    if required_width is not None and entry.cache_key.width != required_width:
                        continue
                    if required_num_inference_steps is not None and entry.cache_key.num_inference_steps < required_num_inference_steps:
                        continue
                cached_emb = entry.clip_embedding
                if cached_emb.dim() == 1:
                    cached_emb = cached_emb.unsqueeze(0)
                cached_norm = cached_emb / cached_emb.norm(dim=-1, keepdim=True)
                sim = torch.nn.functional.cosine_similarity(query_norm.cpu(), cached_norm.cpu()).item()
                if sim > best_sim:
                    best_sim = sim
                    best_key_hash = key_hash

            if best_key_hash is None or best_sim < threshold:
                self._misses += 1
                logger.info(
                    "CLIP semantic search: no match (best_sim=%.4f, threshold=%.2f)",
                    best_sim,
                    threshold,
                )
                return None, None, best_sim

            self._store.move_to_end(best_key_hash)
            self._hits += 1
            entry = self._store[best_key_hash]

            latents = entry.latents
            if target_device is not None:
                latents = latents.to(device=target_device)
            else:
                latents = latents.clone()

            step_latents = None
            if entry.step_latents is not None and target_device is not None:
                step_latents = [
                    StepLatentData(
                        step_index=s.step_index,
                        timestep=s.timestep,
                        latent=s.latent.to(device=target_device),
                    )
                    for s in entry.step_latents
                ]
            elif entry.step_latents is not None:
                step_latents = [
                    StepLatentData(
                        step_index=s.step_index,
                        timestep=s.timestep,
                        latent=s.latent.clone(),
                    )
                    for s in entry.step_latents
                ]

            logger.info(
                "CLIP semantic HIT: key=%s similarity=%.4f (threshold=%.2f, hits=%d, misses=%d)",
                best_key_hash[:8],
                best_sim,
                threshold,
                self._hits,
                self._misses,
            )
            return latents, step_latents, best_sim

    def get_step_latents(
        self, key: CacheKey, target_device: torch.device | str | None = None
    ) -> list[StepLatentData] | None:
        key_hash = key.to_hash()

        with self._lock:
            entry = self._store.get(key_hash)
            if entry is None or entry.step_latents is None:
                return None

            self._store.move_to_end(key_hash)

            if target_device is not None:
                return [
                    StepLatentData(
                        step_index=s.step_index,
                        timestep=s.timestep,
                        latent=s.latent.to(device=target_device),
                    )
                    for s in entry.step_latents
                ]
            return [
                StepLatentData(
                    step_index=s.step_index,
                    timestep=s.timestep,
                    latent=s.latent.clone(),
                )
                for s in entry.step_latents
            ]

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)

    @property
    def memory_usage_mb(self) -> float:
        with self._lock:
            return self._current_memory_bytes / 1024**2

    def clear(self):
        with self._lock:
            self._store.clear()
            self._current_memory_bytes = 0
            self._hits = 0
            self._misses = 0
            logger.info("DiT cache store cleared")

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._store),
                "max_entries": self._max_entries,
                "memory_mb": self._current_memory_bytes / 1024**2,
                "max_memory_gb": self._max_memory_bytes / 1024**3,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self.hit_rate,
            }

    def save_to_disk(self, cache_dir: str | Path) -> int:
        import os
        from pathlib import Path as PathLib

        cache_dir = PathLib(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        with self._lock:
            for key_hash, entry in self._store.items():
                entry_dir = cache_dir / key_hash
                try:
                    entry_dir.mkdir(parents=True, exist_ok=True)

                    torch.save(
                        entry.latents.cpu(),
                        entry_dir / "final_latent.pt",
                    )

                    meta = {
                        "cache_key_hash": entry.cache_key_hash,
                        "metadata": entry.metadata,
                    }

                    if entry.step_latents is not None:
                        step_data = []
                        for s in entry.step_latents:
                            step_file = entry_dir / f"step_{s.step_index:04d}.pt"
                            torch.save(
                                {
                                    "step_index": s.step_index,
                                    "timestep": s.timestep,
                                    "latent": s.latent.cpu(),
                                },
                                step_file,
                            )
                            step_data.append(
                                {
                                    "step_index": s.step_index,
                                    "timestep": s.timestep,
                                    "file": step_file.name,
                                }
                            )
                        meta["step_latents"] = step_data
                        meta["num_steps"] = len(step_data)

                    import json as json_mod

                    with open(entry_dir / "meta.json", "w") as f:
                        json_mod.dump(meta, f, indent=2)

                    saved_count += 1
                except Exception as e:
                    logger.warning(
                        "Failed to save cache entry %s to disk: %s",
                        key_hash[:8],
                        e,
                    )

        logger.info(
            "Saved %d cache entries to %s (%.2f MB)",
            saved_count,
            cache_dir,
            self._current_memory_bytes / 1024**2,
        )
        return saved_count

    def load_from_disk(self, cache_dir: str | Path) -> int:
        import json as json_mod
        from pathlib import Path as PathLib

        cache_dir = PathLib(cache_dir)
        if not cache_dir.exists():
            logger.info("Cache directory %s does not exist, skipping load", cache_dir)
            return 0

        loaded_count = 0
        with self._lock:
            for entry_dir in sorted(cache_dir.iterdir()):
                if not entry_dir.is_dir():
                    continue
                meta_file = entry_dir / "meta.json"
                latent_file = entry_dir / "final_latent.pt"
                if not meta_file.exists() or not latent_file.exists():
                    continue

                try:
                    key_hash = entry_dir.name

                    with open(meta_file) as f:
                        meta = json_mod.load(f)

                    latents = torch.load(latent_file, map_location="cpu", weights_only=True)

                    step_latents = None
                    if "step_latents" in meta and meta["step_latents"]:
                        step_latents = []
                        for step_info in meta["step_latents"]:
                            step_file = entry_dir / step_info["file"]
                            if step_file.exists():
                                step_data = torch.load(
                                    step_file, map_location="cpu", weights_only=True
                                )
                                step_latents.append(
                                    StepLatentData(
                                        step_index=step_data["step_index"],
                                        timestep=step_data["timestep"],
                                        latent=step_data["latent"],
                                    )
                                )

                    if key_hash in self._store:
                        old_entry = self._store[key_hash]
                        self._current_memory_bytes -= self._estimate_entry_bytes(old_entry)

                    entry = CacheEntry(
                        latents=latents,
                        cache_key_hash=meta.get("cache_key_hash", key_hash),
                        step_latents=step_latents,
                        metadata=meta.get("metadata"),
                    )
                    self._store[key_hash] = entry
                    self._current_memory_bytes += self._estimate_entry_bytes(entry)
                    loaded_count += 1

                except Exception as e:
                    logger.warning(
                        "Failed to load cache entry from %s: %s",
                        entry_dir,
                        e,
                    )

        logger.info(
            "Loaded %d cache entries from %s (%.2f MB)",
            loaded_count,
            cache_dir,
            self._current_memory_bytes / 1024**2,
        )
        return loaded_count


def build_cache_key_from_request(
    req: Any,
    pipeline: Any,
) -> CacheKey | None:
    try:
        prompts = req.prompts
        if not prompts:
            return None

        prompt_text = ""
        negative_prompt_text = ""
        if isinstance(prompts[0], str):
            prompt_text = prompts[0]
        elif isinstance(prompts[0], dict):
            prompt_text = prompts[0].get("prompt", "")
            negative_prompt_text = prompts[0].get("negative_prompt", "") or ""

        sampling = req.sampling_params

        height = sampling.height
        width = sampling.width
        if height is None and hasattr(pipeline, "default_sample_size"):
            vae_sf = getattr(pipeline, "vae_scale_factor", 8)
            height = pipeline.default_sample_size * vae_sf
        if width is None and hasattr(pipeline, "default_sample_size"):
            vae_sf = getattr(pipeline, "vae_scale_factor", 8)
            width = pipeline.default_sample_size * vae_sf

        num_inference_steps = sampling.num_inference_steps or 50
        guidance_scale = sampling.guidance_scale if sampling.guidance_scale_provided else 1.0
        true_cfg_scale = sampling.true_cfg_scale or 1.0
        seed = sampling.seed if sampling.seed is not None else -1
        if seed == -1 and sampling.generator is not None:
            try:
                if isinstance(sampling.generator, torch.Generator):
                    seed = sampling.generator.initial_seed()
            except Exception:
                pass

        sigmas = tuple(sampling.sigmas) if sampling.sigmas is not None else None
        max_sequence_length = sampling.max_sequence_length
        num_images_per_prompt = sampling.num_outputs_per_prompt if sampling.num_outputs_per_prompt > 0 else 1

        return CacheKey(
            prompt=prompt_text,
            negative_prompt=negative_prompt_text,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            seed=seed,
            sigmas=sigmas,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt,
        )
    except Exception as e:
        logger.warning("Failed to build cache key: %s", e)
        return None
