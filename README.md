# 5 in 1 Dev Docker Environment so you dont have to run around apps and frameworks to build an idea and when you are done deploy the docker to cloud of choice.Alpha stage

Unified development stack (FastAPI + Redis Stack + RedisInsight + JupyterLab + code-server) with shared persistent volumes.Alpa version main purpose fraud and anmoly detection.

## Services
| Service | URL | Purpose |
|---------|-----|---------|
| FastAPI | http://localhost:8000/ | API + CSV upload demo + AI endpoint |
| Redis | localhost:6379 | Data store (redis-stack-server) |
| RedisInsight | http://localhost:5540/ | Redis GUI |
| JupyterLab | http://localhost:8888/ | Notebooks / data exploration |
| code-server | https://localhost:8443/ | VS Code in browser |

## Shared Volumes (host relative paths)
- `volumes/shared` -> `/shared` (all services)
- `volumes/datasets` -> `/data/datasets` (fastapi, jupyter, code)
- `volumes/uploads` -> `/app/uploads` (FastAPI uploads)
- `volumes/notebooks` -> `/home/jovyan/work` (Jupyter notebooks)
- `volumes/jupyter-user` -> `/home/jovyan/.local` (Jupyter user packages)
- `volumes/code` -> `/config` (code-server data)
- `volumes/redis` -> `/data` (Redis persistence)
- `volumes/redisinsight` -> `/db` (RedisInsight metadata)

These directories are gitignored so teammates create them locally when they run the stack.

## Environment Variables
All config is in `.env` (NOT committed). Provide a template `.env.example` (committed) so others can copy:

```bash
cp .env.example .env
# then edit secrets
```

## Sharing With Teammates
1. Commit everything except `.env` and `volumes/` data.
2. Teammate clones repo.
3. They run: `cp .env.example .env` (Windows PowerShell: `Copy-Item .env.example .env`) and fill real secrets.
4. Start stack: `docker compose up -d --build`.
5. Changes to FastAPI code (`fastapi/`) hot-reload (if you add `--reload`) else rebuild needed.

## Adding New Env Vars
1. Add to `.env`.
2. Reference in `docker-compose.yml` using `${VAR}`.
3. Recreate affected service: `docker compose up -d --build fastapi` (example).

## OpenAI Usage
Set `OPENAI_API_KEY` in `.env`. Then GET `/ai/complete?prompt=Hello`.

df = generate_botfarm(BotFarmConfig(target_rows=120_000))
## Synthetic Click Datasets (Unified Generators + Storage Conventions)
All click generation logic lives in `fastapi/synth/clicks.py`.

Persist CSV outputs under the host path `volumes/datasets/` (mounted inside containers at `/data/datasets`).

Container path mapping:
| Host | In FastAPI / Jupyter / code-server |
|------|------------------------------------|
| `volumes/datasets` | `/data/datasets` |

### Generate via API (writes to Redis optionally)
Basic (lightweight fields only):
```
POST http://localhost:8000/generate_clicks?rows=2000&days=45
```

Enriched (adds fraud + conversion labels). Append `&to_redis=1` to store each row as JSON key `click:<n>`:
```
POST http://localhost:8000/generate_clicks?rows=2000&days=45&enriched=1&fraud_rate=0.06&to_redis=1
```

Bulk chunked insertion:
```
POST http://localhost:8000/generate_clicks_bulk?total_rows=250000&chunk=50000&enriched=1&to_redis=1
```

### Generate via Python (inside repo or notebook)
```python
from fastapi.synth.clicks import generate_basic, generate_enriched, BotFarmConfig, generate_botfarm
df_basic = generate_basic(rows=10_000, days=30)
df_enriched = generate_enriched(rows=25_000, days=60, fraud_rate=0.05)
bot_df = generate_botfarm(BotFarmConfig(target_rows=120_000))
df_enriched.to_csv('volumes/datasets/enriched_sample.csv', index=False)  # host path
# or inside container: df_enriched.to_csv('/data/datasets/enriched_sample.csv', index=False)
```

### Training Models From Enriched Data
Helper script: `python helpers/train_xgb_enriched.py --csv volumes/datasets/enriched_sample.csv` (optionally adapt script to point at your CSV).

### Redis Key Layout (when using `to_redis=1`)
Each row stored as JSON under `click:<ordinal>`; enriched rows include fraud / conversion labels. Use RedisInsight (port 5540) to explore.

### Legacy / Deprecated Scripts
The following legacy generators remain only for backwards compatibility and will be removed: `generate_google_click_data.py`, `generate_search_clicks_enriched.py`, plus shims `fastapi/generator.py`, `fastapi/click_data_generator.py`. Prefer importing from `fastapi/synth/clicks.py`.

### Dataset Hygiene
Do not commit large CSV/Parquet outputs — keep them inside `volumes/datasets/` (gitignored). Promote any reusable feature engineering logic into helper scripts or modules, not notebooks.

## Persisting Additional Data
Add new folder under `volumes/` and mount in `docker-compose.yml`. Keep business logic code under version control, large data in mounted volumes.

## Safety
- Never commit real secrets.
- Rotate keys if exposed.
- Use HTTPS for code-server (self-signed by default). For production, terminate behind a reverse proxy with proper certificates.

## Tear Down
`docker compose down` keeps data (volumes). To clean all data: delete the `volumes/` subfolders.

---
Feel free to extend this README with project-specific notes.
