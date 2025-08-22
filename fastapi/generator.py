"""Deprecated: use functions in `synth.clicks` instead.

This legacy module once contained a large monolithic simulation. It is kept as a
thin shim for backward compatibility and will be removed in a future cleanup.

Migration:
    - generate_basic          -> synth.clicks.generate_basic
    - generate_enriched       -> synth.clicks.generate_enriched
    - BotFarmConfig / botfarm -> synth.clicks.BotFarmConfig & synth.clicks.generate_botfarm
    - Redis helpers           -> synth.clicks.clear_prefix / pipeline_set_json
"""

from synth.clicks import (
        generate_basic as generate_click_dataset,
        generate_enriched,
        BotFarmConfig,
        generate_botfarm,
        clear_prefix,
        pipeline_set_json,
)

__all__ = [
        "generate_click_dataset","generate_enriched","BotFarmConfig","generate_botfarm","clear_prefix","pipeline_set_json"
]
