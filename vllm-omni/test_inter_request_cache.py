from __future__ import annotations

import sys
import os
import types
import importlib
import importlib.util
import torch
import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

# ============================================================
# Directly load cache_store.py and backend.py without going
# through vllm_omni.__init__ (which requires full vllm install)
# ============================================================

# First, mock the vllm_omni.diffusion.cache.base module that backend.py imports
mock_base = types.ModuleType('vllm_omni.diffusion.cache.base')

class CacheBackend:
    def __init__(self, config):
        self.config = config
        self.enabled = False

    def enable(self, pipeline):
        self.enabled = True

    def is_enabled(self):
        return self.enabled

    def lookup(self, req, target_device=None):
        raise NotImplementedError

    def store(self, req, latents, metadata=None):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def stats(self):
        raise NotImplementedError

mock_base.CacheBackend = CacheBackend
sys.modules['vllm_omni.diffusion.cache.base'] = mock_base

# Mock vllm_omni.diffusion.data for DiffusionCacheConfig
mock_data = types.ModuleType('vllm_omni.diffusion.data')

@dataclass
class DiffusionCacheConfig:
    inter_request_max_entries: int = 100
    inter_request_max_memory_gb: float = 4.0

    @classmethod
    def from_dict(cls, d):
        return cls(
            inter_request_max_entries=d.get('inter_request_max_entries', 100),
            inter_request_max_memory_gb=d.get('inter_request_max_memory_gb', 4.0),
        )

mock_data.DiffusionCacheConfig = DiffusionCacheConfig
sys.modules['vllm_omni.diffusion.data'] = mock_data

# Mock vllm_omni.diffusion.cache.inter_request
mock_ir = types.ModuleType('vllm_omni.diffusion.cache.inter_request')
sys.modules['vllm_omni.diffusion.cache.inter_request'] = mock_ir

# Mock vllm_omni.diffusion.cache
mock_cache = types.ModuleType('vllm_omni.diffusion.cache')
sys.modules['vllm_omni.diffusion.cache'] = mock_cache

# Mock vllm_omni.diffusion
mock_diff = types.ModuleType('vllm_omni.diffusion')
sys.modules['vllm_omni.diffusion'] = mock_diff

# Mock vllm_omni
mock_omni = types.ModuleType('vllm_omni')
sys.modules['vllm_omni'] = mock_omni

# Now load cache_store.py directly
BASE = '/vllm-workspace/vllm-omni/vllm_omni/diffusion/cache/inter_request'

spec_cs = importlib.util.spec_from_file_location(
    'vllm_omni.diffusion.cache.inter_request.cache_store',
    os.path.join(BASE, 'cache_store.py'))
mod_cs = importlib.util.module_from_spec(spec_cs)
sys.modules['vllm_omni.diffusion.cache.inter_request.cache_store'] = mod_cs
spec_cs.loader.exec_module(mod_cs)

CacheKey = mod_cs.CacheKey
DiTCacheStore = mod_cs.DiTCacheStore

# Load backend.py directly
spec_be = importlib.util.spec_from_file_location(
    'vllm_omni.diffusion.cache.inter_request.backend',
    os.path.join(BASE, 'backend.py'))
mod_be = importlib.util.module_from_spec(spec_be)
sys.modules['vllm_omni.diffusion.cache.inter_request.backend'] = mod_be
spec_be.loader.exec_module(mod_be)

InterRequestCacheBackend = mod_be.InterRequestCacheBackend

print("=" * 60)
print("Inter-Request Cache (Chorus Stage-1) Unit Tests")
print("=" * 60)

# Test 1: CacheKey hashing - same inputs produce same hash
key1 = CacheKey(
    prompt='a cat', negative_prompt='', height=1024, width=1024,
    num_inference_steps=50, guidance_scale=1.0, true_cfg_scale=4.0,
    seed=42, sigmas=None, max_sequence_length=512, num_images_per_prompt=1,
)
key2 = CacheKey(
    prompt='a cat', negative_prompt='', height=1024, width=1024,
    num_inference_steps=50, guidance_scale=1.0, true_cfg_scale=4.0,
    seed=42, sigmas=None, max_sequence_length=512, num_images_per_prompt=1,
)
key3 = CacheKey(
    prompt='a dog', negative_prompt='', height=1024, width=1024,
    num_inference_steps=50, guidance_scale=1.0, true_cfg_scale=4.0,
    seed=42, sigmas=None, max_sequence_length=512, num_images_per_prompt=1,
)
assert key1.to_hash() == key2.to_hash(), 'Same inputs should produce same hash'
assert key1.to_hash() != key3.to_hash(), 'Different inputs should produce different hash'
print('[PASS] Test 1: CacheKey hashing - identical inputs => identical hash')

# Test 2: CacheKey hashing - different seed produces different hash
key4 = CacheKey(
    prompt='a cat', negative_prompt='', height=1024, width=1024,
    num_inference_steps=50, guidance_scale=1.0, true_cfg_scale=4.0,
    seed=99, sigmas=None, max_sequence_length=512, num_images_per_prompt=1,
)
assert key1.to_hash() != key4.to_hash(), 'Different seed should produce different hash'
print('[PASS] Test 2: CacheKey hashing - different seed => different hash')

# Test 3: DiTCacheStore put/get
store = DiTCacheStore(max_entries=10, max_memory_gb=1.0)
latents = torch.randn(1, 4, 64, 64)
store.put(key1, latents)
retrieved = store.get(key1)
assert retrieved is not None, 'Should find cached entry'
assert torch.allclose(latents, retrieved), 'Retrieved latents should match original'
print('[PASS] Test 3: DiTCacheStore put/get - stored and retrieved latents match')

# Test 4: Cache miss
miss = store.get(key3)
assert miss is None, 'Should not find non-cached entry'
print('[PASS] Test 4: Cache miss - non-cached key returns None')

# Test 5: LRU eviction
key_lru1 = CacheKey(
    prompt='a cat', negative_prompt='', height=1024, width=1024,
    num_inference_steps=50, guidance_scale=1.0, true_cfg_scale=4.0,
    seed=1, sigmas=None, max_sequence_length=512, num_images_per_prompt=1,
)
key_lru2 = CacheKey(
    prompt='a dog', negative_prompt='', height=1024, width=1024,
    num_inference_steps=50, guidance_scale=1.0, true_cfg_scale=4.0,
    seed=2, sigmas=None, max_sequence_length=512, num_images_per_prompt=1,
)
key_lru3 = CacheKey(
    prompt='a bird', negative_prompt='', height=1024, width=1024,
    num_inference_steps=50, guidance_scale=1.0, true_cfg_scale=4.0,
    seed=3, sigmas=None, max_sequence_length=512, num_images_per_prompt=1,
)
store2 = DiTCacheStore(max_entries=2, max_memory_gb=1.0)
store2.put(key_lru1, torch.randn(1, 4, 64, 64))
store2.put(key_lru2, torch.randn(1, 4, 64, 64))
store2.put(key_lru3, torch.randn(1, 4, 64, 64))
assert store2.get(key_lru1) is None, 'key_lru1 should be evicted (LRU)'
assert store2.get(key_lru2) is not None, 'key_lru2 should remain'
assert store2.get(key_lru3) is not None, 'key_lru3 should remain'
print('[PASS] Test 5: LRU eviction - oldest entry evicted when max_entries reached')

# Test 6: InterRequestCacheBackend enable/disable
config = DiffusionCacheConfig.from_dict({
    'inter_request_max_entries': 50,
    'inter_request_max_memory_gb': 2.0,
})
backend = InterRequestCacheBackend(config)
assert not backend.is_enabled(), 'Should not be enabled before calling enable()'

class MockPipeline:
    pass

backend.enable(MockPipeline())
assert backend.is_enabled(), 'Should be enabled after calling enable()'
print('[PASS] Test 6: InterRequestCacheBackend enable/disable')

# Test 7: Store and lookup (cache hit)
class MockSamplingParams:
    height = 1024
    width = 1024
    num_inference_steps = 50
    guidance_scale = 1.0
    guidance_scale_provided = True
    true_cfg_scale = 4.0
    seed = 42
    sigmas = None
    max_sequence_length = 512
    num_outputs_per_prompt = 1

class MockRequest:
    prompts = ['a cat']
    sampling_params = MockSamplingParams()

req = MockRequest()
output_tensor = torch.randn(1, 3, 64, 64)
backend.store(req, output_tensor)

cached = backend.lookup(req)
assert cached is not None, 'Should find cached output'
assert torch.allclose(output_tensor, cached), 'Cached output should match original'
print('[PASS] Test 7: Store and lookup - cache hit with identical request')

# Test 8: Cache miss for different request
class MockRequest2:
    prompts = ['a dog']
    sampling_params = MockSamplingParams()

cached2 = backend.lookup(MockRequest2())
assert cached2 is None, 'Should not find cached output for different request'
print('[PASS] Test 8: Cache miss - different prompt returns None')

# Test 9: Cache hit with same request again
cached3 = backend.lookup(req)
assert cached3 is not None, 'Should still find cached output for original request'
print('[PASS] Test 9: Repeated lookup - cache hit still works')

# Test 10: Stats
stats = backend.stats()
assert stats['entries'] > 0, 'Should have at least one entry'
assert stats['hits'] > 0, 'Should have at least one hit'
assert stats['misses'] > 0, 'Should have at least one miss'
print(f'[PASS] Test 10: Stats - entries={stats["entries"]}, hits={stats["hits"]}, misses={stats["misses"]}, hit_rate={stats["hit_rate"]:.2%}')

# Test 11: Clear cache
backend.clear()
assert backend.stats()['entries'] == 0, 'Cache should be empty after clear'
print('[PASS] Test 11: Clear cache - cache empty after clear()')

# Test 12: DiffusionCacheConfig inter_request parameters
config2 = DiffusionCacheConfig.from_dict({
    'inter_request_max_entries': 200,
    'inter_request_max_memory_gb': 8.0,
})
assert config2.inter_request_max_entries == 200
assert config2.inter_request_max_memory_gb == 8.0
print('[PASS] Test 12: DiffusionCacheConfig - inter_request parameters parsed correctly')

# Test 13: Default config values
config3 = DiffusionCacheConfig.from_dict({})
assert config3.inter_request_max_entries == 100
assert config3.inter_request_max_memory_gb == 4.0
print('[PASS] Test 13: DiffusionCacheConfig - default values correct')

# Test 14: End-to-end simulation - first request computes, second reuses
print()
print("--- End-to-end simulation ---")

config_e2e = DiffusionCacheConfig.from_dict({
    'inter_request_max_entries': 100,
    'inter_request_max_memory_gb': 4.0,
})
backend_e2e = InterRequestCacheBackend(config_e2e)
backend_e2e.enable(MockPipeline())

req1 = MockRequest()
dit_output = torch.randn(1, 3, 1024, 1024)

cached1 = backend_e2e.lookup(req1)
assert cached1 is None, 'First request should be a cache miss'
backend_e2e.store(req1, dit_output)
print('[PASS] Test 14a: First request - cache miss, computed and stored')

cached2 = backend_e2e.lookup(req1)
assert cached2 is not None, 'Second request with same inputs should be a cache hit'
assert torch.allclose(dit_output, cached2), 'Cached output should match original'
print('[PASS] Test 14b: Second request - cache hit, computation skipped!')

stats_e2e = backend_e2e.stats()
assert stats_e2e['hits'] == 1, f'Expected 1 hit, got {stats_e2e["hits"]}'
assert stats_e2e['misses'] == 1, f'Expected 1 miss, got {stats_e2e["misses"]}'
print(f'[PASS] Test 14c: Stats - hits={stats_e2e["hits"]}, misses={stats_e2e["misses"]}, hit_rate={stats_e2e["hit_rate"]:.2%}')

print()
print("=" * 60)
print("All 14 tests passed! Inter-request cache is working correctly.")
print("=" * 60)
