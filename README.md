# Dev Docker Environment

Unified development stack (FastAPI + Redis Stack + RedisInsight + JupyterLab + code-server) with shared persistent volumes.

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
