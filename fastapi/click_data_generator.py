"""Deprecated shim. Use `synth.clicks.generate_basic` or `generate_enriched`.

Kept for backward compatibility with older imports.
"""

from synth.clicks import generate_basic as generate_click_dataset, generate_enriched, clear_prefix as clear_click_keys, pipeline_set_json as push_to_redis

__all__ = [
    "generate_click_dataset","generate_enriched","clear_click_keys","push_to_redis"
]

