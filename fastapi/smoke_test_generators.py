"""Minimal smoke tests for unified synthetic click generators.

Run inside the FastAPI container or an environment with required deps:
  python smoke_test_generators.py
"""
from synth.clicks import generate_basic, generate_enriched, BotFarmConfig, generate_botfarm, pipeline_set_json, clear_prefix
import redis

def main():
    basic = generate_basic(rows=250, days=10)
    enriched = generate_enriched(rows=300, days=15, fraud_rate=0.07)
    botfarm = generate_botfarm(BotFarmConfig(target_rows=5_000, days=5, max_extra_days=5))
    print("basic rows", len(basic), "cols", len(basic.columns))
    print("enriched rows", len(enriched), "fraud%", round(enriched['y_fraud'].mean()*100,2))
    print("botfarm rows", len(botfarm), "fraud%", round(botfarm['y_fraud'].mean()*100,2))
    r = redis.Redis(host="redis", port=6379, decode_responses=True)
    deleted = clear_prefix(r, "click:")
    print("cleared", deleted, "existing click:* keys")
    inserted = pipeline_set_json(r, enriched.head(100), prefix="click:")
    print("inserted", inserted, "sample enriched rows into redis")
    sample_key = next(r.scan_iter(match="click:*"), None)
    print("sample redis key", sample_key, "value len", len(r.get(sample_key)) if sample_key else None)
    print("OK")

if __name__ == "__main__":
    main()
