## Synthetic Search Click Dataset Generator

Script: `generate_google_click_data.py`

Generates a realistic, anonymized search click log for experimentation, analytics, and ML prototyping.

### Features
* Intent categories (navigational / informational / transactional / local)
* Device, country, language, referrer surfaces
* SERP rank, impressions, clicks, CTR
* Dwell time, bounce flag
* SERP feature types and brand query flag
* Verbose descriptive schema (default) or legacy short schema (`--legacy`)

### Install (inside project root)
Uses Python 3.11+.

```bash
python generate_google_click_data.py --help
```

### Command Options
| Flag | Default | Description |
|------|---------|-------------|
| `--rows` | 10000 | Number of rows to generate |
| `--days` | 60 | Days history span (back from today) |
| `--seed` | 1337 | RNG seed |
| `--out` | synthetic_google_clicks.csv | Output CSV path |
| `--legacy` | (off) | Use short legacy column names |

### Verbose Schema Columns
| Column | Meaning |
|--------|---------|
| event_date | ISO date of the event |
| search_query | Raw query text |
| query_intent_category | Inferred intent category |
| landing_page_url | Destination page URL |
| page_title | Page title variant |
| device_type | desktop / mobile / tablet |
| user_country | Country code |
| user_language | Language/locale code |
| traffic_referrer | Surface (google, google_news, etc.) |
| user_id | Synthetic user ID |
| session_id | Synthetic session ID |
| rank_position | Organic rank (1–50) |
| serp_impressions | Impressions for this combination |
| serp_clicks | Clicks out of impressions |
| click_through_rate | serp_clicks / serp_impressions |
| dwell_time_seconds | Estimated dwell time if clicked |
| bounced_session | True if considered a bounce |
| serp_feature | SERP feature present |
| is_brand_query | Brand / navigational query heuristic |

### Quick Start
Generate 5k rows, 30 days history:
```bash
python generate_google_click_data.py --rows 5000 --days 30 --out clicks_verbose.csv
```

Legacy schema:
```bash
python generate_google_click_data.py --legacy --out clicks_legacy.csv
```

### Sample (first 5 rows verbose)
```text
event_date,search_query,query_intent_category,landing_page_url,page_title,device_type,user_country,user_language,traffic_referrer,user_id,session_id,rank_position,serp_impressions,serp_clicks,click_through_rate,dwell_time_seconds,bounced_session,serp_feature,is_brand_query
...
```

### Load Into Pandas
```python
import pandas as pd
df = pd.read_csv('clicks_verbose.csv')
print(df.head())
```

### (Optional) Write to Redis
Add a helper or use existing FastAPI upload route or Jupyter:
```python
import redis, pandas as pd
r = redis.Redis(host='redis', port=6379, decode_responses=True)
df = pd.read_csv('clicks_verbose.csv')
for i, row in df.iterrows():
    r.hset(f'click:{i}', mapping={k: str(v) for k,v in row.items()})
```

### ML Quickstart (XGBoost)
```python
import pandas as pd, xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
df = pd.read_csv('clicks_verbose.csv')
target = 'serp_clicks'
features = [c for c in df.columns if c not in {target,'search_query','landing_page_url','page_title','user_id','session_id'}]
X = df[features].select_dtypes(include=['number']).fillna(0)
y = df[target]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = xgb.XGBRegressor(n_estimators=120,max_depth=5,learning_rate=0.1,subsample=0.9)
model.fit(X_train,y_train)
pred = model.predict(X_test)
print('RMSE', mean_squared_error(y_test,pred,squared=False))
```

### Git Ignore Reminder
Do not commit real data dumps if they become large; adjust `.gitignore` if needed. Synthetic CSVs are usually safe to commit for examples.

### Future Enhancements
* Parquet output
* Direct Redis export flag (`--to-redis`)
* Conversion / label simulation
* Multi-session behavior modeling

Contributions welcome.