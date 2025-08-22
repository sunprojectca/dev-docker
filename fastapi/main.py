from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse
import redis, csv, io, os, math, json
from pathlib import Path
from typing import List, Optional

"""FastAPI app using unified synthetic click generators (see synth/clicks.py)."""
from synth.clicks import (
  generate_basic as generate_click_dataset,  # maintain old name mapping
  generate_enriched,
  BotFarmConfig,
  generate_botfarm,
  clear_prefix as clear_click_keys,
  pipeline_set_json,
)

# ML imports (lazy to avoid startup cost if unused)
try:
  import pandas as pd
  from sklearn.model_selection import train_test_split
  import xgboost as xgb
  _ml_available = True
except Exception:
  _ml_available = False

try:
  from openai import OpenAI
  _openai_available = True
except Exception:
  _openai_available = False

app = FastAPI()
r = redis.Redis(host="redis", port=6379, decode_responses=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if (_openai_available and OPENAI_API_KEY) else None
_xgb_model: Optional[xgb.Booster] = None if _ml_available else None
_xgb_features: Optional[List[str]] = None
_model_dir = Path("/shared/models")
try:
  _model_dir.mkdir(parents=True, exist_ok=True)
except Exception:
  pass

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <h1>Developer Sandbox</h1>
    <div style="display:flex;gap:10px;flex-wrap:wrap;">
      <a href="/docs"><button>FastAPI Docs</button></a>
      <a href="http://localhost:5540" target="_blank"><button>RedisInsight</button></a>
      <a href="http://localhost:8888" target="_blank"><button>JupyterLab</button></a>
      <a href="http://localhost:8080" target="_blank"><button>VS Code (code-server)</button></a>
    </div>
    <hr/>
    <h3>Functions</h3>
    <form action="/upload_csv" method="post" enctype="multipart/form-data">
      <p><b>Upload CSV into Redis:</b></p>
      <input type="file" name="file" />
      <button type="submit">Upload</button>
    </form>
    <br/>
    <form action="/list_keys" method="get">
      <button type="submit">List Redis Keys</button>
    </form>
    <br/>
    <form action="/clear_clicks" method="post" onsubmit="return confirm('Delete all click:* keys?');">
      <button type="submit" style="background:#b44;color:#fff;">Clear click:* Keys</button>
    </form>
    <br/>
    <form action="/query?key=row:0" method="get">
      <input type="text" name="key" placeholder="Enter Redis key" />
      <button type="submit">Query Redis</button>
    </form>
    <br/>
    <form action="/ai/complete" method="get">
      <input type="text" name="prompt" placeholder="Ask AI (needs OPENAI_API_KEY)" style="width:300px;"/>
      <button type="submit">AI Complete</button>
    </form>
    <br/>
    <form action="/train_xgb" method="post">
      <input type="text" name="feature_prefix" placeholder="row:" value="row:" />
      <input type="text" name="target_field" placeholder="target field name" />
      <button type="submit">Train XGBoost</button>
    </form>
    <br/>
    <form action="/save_xgb" method="post">
      <button type="submit">Save XGBoost Model</button>
    </form>
    <br/>
    <form action="/load_xgb" method="post">
      <button type="submit">Load XGBoost Model</button>
    </form>
    <br/>
    <form action="/xgb_status" method="get">
      <button type="submit">XGBoost Status</button>
    </form>
    <br/>
    <form action="/generate_clicks" method="post">
      <input type="number" name="rows" value="1000" />
      <input type="number" name="days" value="30" />
  <label><input type="checkbox" name="enriched" value="1"/> Enriched schema</label>
      <button type="submit">Generate Click Dataset</button>
    </form>
    """

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    csvfile = io.StringIO(content.decode())
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader):
        r.hset(f"row:{i}", mapping=row)
    return {"status": "ok", "rows_loaded": i+1}

@app.get("/list_keys")
def list_keys():
    keys = r.keys("*")
    return {"keys": keys}

@app.get("/query")
def query(key: str):
    data = r.hgetall(key)
    return {"key": key, "data": data}

@app.get("/ai/complete")
def ai_complete(prompt: str):
  if client is None:
    raise HTTPException(status_code=400, detail="OpenAI not configured")
  try:
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=60)
    text = resp.choices[0].message.content
    return {"prompt": prompt, "completion": text}
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_xgb")
def train_xgb(feature_prefix: str = "row:", target_field: str = "target"):
  if not _ml_available:
    raise HTTPException(status_code=400, detail="ML libs not installed")
  keys = [k for k in r.scan_iter(match=f"{feature_prefix}*")]
  if not keys:
    raise HTTPException(status_code=404, detail="No matching keys")
  rows = []
  for k in keys:
    h = r.hgetall(k)
    if target_field not in h:
      continue
    rows.append(h)
  if not rows:
    raise HTTPException(status_code=400, detail="No rows with target field")
  df = pd.DataFrame(rows)
  for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors='ignore')
  if target_field not in df.columns:
    raise HTTPException(status_code=400, detail="Target field missing after load")
  y = df[target_field]
  X = df.drop(columns=[target_field])
  X = X.select_dtypes(include=['number'])
  if X.empty:
    raise HTTPException(status_code=400, detail="No numeric feature columns")
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  dtrain = xgb.DMatrix(X_train, label=y_train)
  dtest = xgb.DMatrix(X_test, label=y_test)
  params = {"objective": "reg:squarederror", "max_depth": 4, "eta": 0.1, "subsample": 0.9}
  booster = xgb.train(params, dtrain, num_boost_round=50)
  pred = booster.predict(dtest)
  import numpy as np
  rmse = math.sqrt(((pred - y_test.values)**2).mean())
  global _xgb_model, _xgb_features
  _xgb_model = booster
  _xgb_features = list(X.columns)
  return {"rows_used": int(len(X_train) + len(X_test)), "features": _xgb_features, "rmse": rmse}

@app.get("/predict_xgb")
def predict_xgb(**features: float):
  if _xgb_model is None:
    raise HTTPException(status_code=400, detail="Model not trained")
  import xgboost as xgb  # ensure available
  cols = sorted(features.keys())
  data = [[features[c] for c in cols]]
  dmatrix = xgb.DMatrix(data, feature_names=cols)
  pred = _xgb_model.predict(dmatrix)
  return {"prediction": float(pred[0]), "features_order": cols}

@app.get("/xgb_status")
def xgb_status():
  return {
    "in_memory": _xgb_model is not None,
    "saved_model": (_model_dir / "xgb_model.json").exists(),
    "features": _xgb_features
  }

@app.post("/save_xgb")
def save_xgb():
  if _xgb_model is None:
    raise HTTPException(status_code=400, detail="No trained model in memory")
  try:
    _model_dir.mkdir(parents=True, exist_ok=True)
    model_path = _model_dir / "xgb_model.json"
    _xgb_model.save_model(str(model_path))
    if _xgb_features:
      with open(_model_dir / "xgb_model.features.json", "w") as f:
        json.dump(_xgb_features, f)
    return {"status": "saved", "path": str(model_path), "features": _xgb_features}
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

@app.post("/load_xgb")
def load_xgb():
  if not _ml_available:
    raise HTTPException(status_code=400, detail="ML libs not installed")
  model_path = _model_dir / "xgb_model.json"
  if not model_path.exists():
    raise HTTPException(status_code=404, detail="Saved model file not found")
  try:
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    features_path = _model_dir / "xgb_model.features.json"
    global _xgb_model, _xgb_features
    _xgb_model = booster
    if features_path.exists():
      _xgb_features = json.loads(features_path.read_text())
    return {"status": "loaded", "features": _xgb_features}
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_clicks")
def clear_clicks():
  deleted = clear_click_keys(r, prefix="click:")
  return {"deleted": deleted}

@app.get("/clear_clicks")
def clear_clicks_get():
  return clear_clicks()

@app.post("/generate_clicks")
def generate_clicks(rows: int = 1000, days: int = 30, seed: int = 1337, enriched: int = 0, to_redis: int = 0, fraud_rate: float = 0.05):
  # enriched flag toggles which generator style
  if enriched:
    df = generate_enriched(rows=rows, days=days, seed=seed, fraud_rate=fraud_rate)
  else:
    df = generate_click_dataset(rows=rows, days=days, seed=seed)
  inserted = 0
  if to_redis:
    inserted = pipeline_set_json(r, df, prefix="click:")
  fraud_count = int(df["y_fraud"].sum()) if "y_fraud" in df.columns else 0
  return {"rows": len(df), "fraud_rows": fraud_count, "fraud_rate_actual": round(fraud_count/len(df),4) if len(df) else 0, "redis_inserted": inserted, "columns": list(df.columns)}

@app.get("/generate_clicks")
def generate_clicks_get(rows: int = 1000, days: int = 30, seed: int = 1337, enriched: int = 0, to_redis: int = 0, fraud_rate: float = 0.05):
  return generate_clicks(rows=rows, days=days, seed=seed, enriched=enriched, to_redis=to_redis, fraud_rate=fraud_rate)

@app.post("/generate_clicks_bulk")
def generate_clicks_bulk(total_rows: int = 1_000_000, chunk: int = 100_000, days: int = 60, seed: int = 1337, enriched: int = 0, to_redis: int = 1, fraud_rate: float = 0.05):
  from math import ceil
  iterations = ceil(total_rows / chunk)
  total_inserted = 0
  total_fraud = 0
  for i in range(iterations):
    part_rows = chunk if (i < iterations-1) else (total_rows - chunk*(iterations-1))
    if enriched:
      df = generate_enriched(rows=part_rows, days=days, seed=seed + i, fraud_rate=fraud_rate)
    else:
      df = generate_click_dataset(rows=part_rows, days=days, seed=seed + i)
    if to_redis:
      total_inserted += pipeline_set_json(r, df, prefix="click:")
    if "y_fraud" in df.columns:
      total_fraud += int(df["y_fraud"].sum())
  return {
    "requested_rows": total_rows,
    "chunks": iterations,
    "redis_inserted": total_inserted,
    "fraud_rows": total_fraud,
    "fraud_rate_actual": round(total_fraud/total_rows,4) if total_rows else 0
  }

@app.get("/generate_clicks_bulk")
def generate_clicks_bulk_get(total_rows: int = 1_000_000, chunk: int = 100_000, days: int = 60, seed: int = 1337, enriched: int = 0, to_redis: int = 1, fraud_rate: float = 0.05):
  return generate_clicks_bulk(total_rows=total_rows, chunk=chunk, days=days, seed=seed, enriched=enriched, to_redis=to_redis, fraud_rate=fraud_rate)

@app.get("/redis_health")
def redis_health():
  try:
    pong = r.ping()
    sample = None
    for k in r.scan_iter(match="click:*", count=1):
      sample = k
      break
    return {"status":"ok" if pong else "fail", "sample_key": sample, "dbsize": r.dbsize()}
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
