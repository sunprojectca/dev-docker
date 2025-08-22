# Helper Scripts

Centralized Python scripts callable from host or container.

| Script | Purpose |
|--------|---------|
| `openai_check.py` | Quick verification of `OPENAI_API_KEY` via minimal chat completion. |
| `train_xgb_enriched.py` | Train fraud + conversion XGBoost models from enriched synthetic dataset. |

General guidance:
- Keep business logic reusable; if a script grows large, consider promoting it into a proper module/package.
- Avoid duplicating code already available in `fastapi/synth/clicks.py`.
