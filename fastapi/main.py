from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import redis, csv, io, os

try:
  from openai import OpenAI
  _openai_available = True
except Exception:
  _openai_available = False

app = FastAPI()
r = redis.Redis(host="redis", port=6379, decode_responses=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if (_openai_available and OPENAI_API_KEY) else None

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <h1>Developer Sandbox</h1>
    <div style="display:flex;gap:10px;flex-wrap:wrap;">
      <a href="/docs"><button>FastAPI Docs</button></a>
      <a href="http://localhost:5540" target="_blank"><button>RedisInsight</button></a>
      <a href="http://localhost:8888" target="_blank"><button>JupyterLab</button></a>
      <a href="http://localhost:8890" target="_blank"><button>Voilà Dashboard</button></a>
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
    <form action="/query?key=row:0" method="get">
      <input type="text" name="key" placeholder="Enter Redis key" />
      <button type="submit">Query Redis</button>
    </form>
    <br/>
    <form action="/ai/complete" method="get">
      <input type="text" name="prompt" placeholder="Ask AI (needs OPENAI_API_KEY)" style="width:300px;"/>
      <button type="submit">AI Complete</button>
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
