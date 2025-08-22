"""Unified synthetic clickstream generation & Redis helper module.

Provides three generator styles:
- basic_clicks: lighter schema similar to generate_google_click_data
- enriched_clicks: detailed schema with fraud + conversion labels
- botfarm_clicks: human + bot farm simulation with target_rows auto-extension

Shared helpers avoid duplicated logic across previous modules (generator.py,
click_data_generator.py, generate_* scripts, notebook inlining).
"""
from __future__ import annotations
import random, math, string, uuid, ipaddress, json, time
from dataclasses import dataclass
from datetime import date, timedelta, datetime, timezone
from typing import List, Tuple, Dict, Any, Iterable, Optional

import numpy as np
import pandas as pd

# ------------------------------ common weighted picker ------------------------------
def wchoice(pairs: List[Tuple[Any,float]]):
    items, weights = zip(*pairs)
    return random.choices(items, weights=weights, k=1)[0]

# Shared categorical universes
QUERY_CATEGORIES = [
    ("navigational", 0.28), ("informational", 0.44), ("transactional", 0.23), ("local", 0.05)
]
DEVICES = [("desktop", 0.46), ("mobile", 0.47), ("tablet", 0.07)]
COUNTRIES = [("US",0.46),("GB",0.10),("CA",0.09),("AU",0.06),("IN",0.12),("DE",0.06),("FR",0.05),("BR",0.04),("JP",0.02)]
LANGS = [("en",0.55),("en-US",0.20),("en-GB",0.10),("de",0.04),("fr",0.04),("pt",0.03),("hi",0.02),("ja",0.01),("es",0.01)]
BROWSERS = [("Chrome",0.62),("Safari",0.18),("Edge",0.10),("Firefox",0.08),("Other",0.02)]
OSES = [("Windows",0.35),("Android",0.32),("iOS",0.20),("macOS",0.10),("Linux",0.03)]
SERP_FEATURES = ["none","sitelinks","faq","video","image","news","localpack","shopping"]
CAMPAIGNS = ["Brand","Generic","Competitor","Retargeting","DisplayProspecting"]
ADGROUPS = {
    "Brand":["Brand Core","Brand Location"],
    "Generic":["Apartments City","Amenities","Near Me"],
    "Competitor":["Comp A","Comp B"],
    "Retargeting":["Visitors 7d","Abandoned Tours"],
    "DisplayProspecting":["In-Market","Affinity Home"]
}
CREATIVE_TYPES = ["RSA","Text","Display","Video"]

# ------------------------------ low-level helpers ------------------------------

def random_query(category: str) -> str:
    topics = {
        "navigational": ["facebook login","youtube","gmail","amazon","wikipedia","bank","official site"],
        "informational": ["how to","what is","best way to","benefits of","guide to","tutorial"],
        "transactional": ["buy","price","discount","near me","coupon","best"],
        "local": ["near me","closest","open now","hours","directions","map"],
    }
    nouns = ["apartments","redis","python","laptop","headphones","router","ssd","credit card","insurance","pizza","coffee","gym"]
    t = random.choice(topics[category]); n = random.choice(nouns)
    if category in ("navigational","local"): return f"{n} {t}"
    return f"{t} {n}"

def positional_ctr(position: int) -> float:
    base = 0.38 * (1 / math.log(position + 1.8))
    noise = random.uniform(-0.03, 0.03)
    return max(0.0, min(0.8, base + noise))

# ------------------------------ basic/enriched shared ------------------------------

def build_query_universe(size: int) -> Tuple[List[str], np.ndarray, Dict[str,str], set]:
    cats = [wchoice(QUERY_CATEGORIES) for _ in range(size)]
    q = [random_query(c) for c in cats]
    q = list(dict.fromkeys(q))
    idx = np.arange(1, len(q)+1)
    zipf_w = (1 / (idx ** 1.1)); zipf_w /= zipf_w.sum()
    serp_bias = {qq: random.choice(SERP_FEATURES) for qq in q}
    brand_terms = set(x for x in q if any(k in x for k in ("official","login","gmail")))
    return q, zipf_w, serp_bias, brand_terms

def pick_campaign() -> Tuple[str,str,int,str]:
    c = random.choice(CAMPAIGNS)
    g = random.choice(ADGROUPS[c])
    ad_id = random.randint(100000, 999999)
    creative = random.choice(CREATIVE_TYPES)
    return c, g, ad_id, creative

# ------------------------------ Generators ------------------------------

def generate_basic(rows: int = 10_000, days: int = 60, seed: int = 1337) -> pd.DataFrame:
    random.seed(seed); np.random.seed(seed)
    q_universe, zipf_w, serp_bias, brand_terms = build_query_universe(max(500, int(rows*0.1)))
    today = date.today(); start = today - timedelta(days=days-1)
    ranks = np.arange(1,51); rank_w = np.linspace(0.25,0.75,50)[::-1]; rank_w /= rank_w.sum()
    recs = []
    for _ in range(rows):
        d = start + timedelta(days=random.randint(0, days-1))
        q = random.choices(q_universe, weights=zipf_w, k=1)[0]
        if any(x in q for x in ["buy","price","discount","coupon"]): cat = "transactional"
        elif any(x in q for x in ["near me","open now","hours","directions"]): cat = "local"
        elif any(x in q for x in ["login","official","site"]): cat = "navigational"
        else: cat = "informational"
        position = int(np.random.choice(ranks, p=rank_w))
        impressions = max(1,int(np.random.lognormal(mean=1.2,sigma=0.6)))
        ctr = positional_ctr(position)
        is_brand = q in brand_terms or any(k in q for k in ("official","login"))
        if is_brand: ctr = min(0.85, ctr*1.4)
        clicks = np.random.binomial(impressions, max(0,min(0.95,ctr)))
        recs.append({
            "event_date": d.isoformat(),
            "search_query": q,
            "query_intent_category": cat,
            "rank_position": position,
            "serp_impressions": impressions,
            "serp_clicks": clicks,
            "click_through_rate": round(clicks/impressions,4),
            "is_brand_query": bool(is_brand),
            "serp_feature": serp_bias[q],
        })
    df = pd.DataFrame.from_records(recs).sort_values(["event_date","search_query"]).reset_index(drop=True)
    return df

# Enriched generator (adapted & trimmed)
ISP_CLASSES = [("residential",0.74),("mobile_carrier",0.18),("datacenter",0.08)]

def generate_enriched(rows: int = 50_000, days: int = 90, seed: int = 7, fraud_rate: float = 0.08, value_per_conv: float = 300.0) -> pd.DataFrame:
    random.seed(seed); np.random.seed(seed)
    q_universe, zipf_w, serp_bias, brand_terms = build_query_universe(max(500,int(rows*0.12)))
    today = date.today(); start = today - timedelta(days=days-1)
    ranks = np.arange(1,51); rank_w = np.linspace(0.25,0.75,50)[::-1]; rank_w /= rank_w.sum()
    recs = []
    for _ in range(rows):
        d = start + timedelta(days=random.randint(0, days-1))
        hour = random.randint(0,23)
        q = random.choices(q_universe, weights=zipf_w, k=1)[0]
        if any(x in q for x in ["buy","price","discount","coupon"]): cat = "transactional"
        elif any(x in q for x in ["near me","open now","hours","directions"]): cat = "local"
        elif any(x in q for x in ["login","official","site"]): cat = "navigational"
        else: cat = "informational"
        position = int(np.random.choice(ranks, p=rank_w))
        impressions = max(1,int(np.random.lognormal(mean=1.3,sigma=0.6)))
        ctr = positional_ctr(position)
        is_brand = q in brand_terms or any(k in q for k in ("official","login"))
        if is_brand: ctr = min(0.85, ctr*1.4)
        isp_class = wchoice(ISP_CLASSES)
        base_fraud = 0.5 if isp_class == "datacenter" else 0.0
        p_fraud = min(0.95, fraud_rate*0.7 + base_fraud)
        is_fraud = random.random() < p_fraud
        if is_fraud: ctr *= 0.6
        clicks = np.random.binomial(impressions, max(0,min(0.95,ctr)))
        base_cvr = {"navigational":0.11,"informational":0.03,"transactional":0.07,"local":0.05}[cat]
        if is_fraud: base_cvr *= 0.05
        conv = int(np.random.binomial(clicks, base_cvr)) if clicks>0 else 0
        revenue = round(conv * value_per_conv * np.random.uniform(0.9,1.1),2)
        recs.append({
            "event_date": d.isoformat(),
            "hour_of_day": hour,
            "search_query": q,
            "query_intent_category": cat,
            "rank_position": position,
            "serp_impressions": impressions,
            "serp_clicks": clicks,
            "click_through_rate": round(clicks/impressions,4),
            "is_brand_query": bool(is_brand),
            "isp_class": isp_class,
            "conversions": conv,
            "revenue": revenue,
            "y_fraud": int(is_fraud),
            "y_conv": int(conv>0),
        })
    df = pd.DataFrame.from_records(recs).sort_values(["event_date","search_query"]).reset_index(drop=True)
    df["ctr"] = df["click_through_rate"]
    df["cvr"] = (df["conversions"] / df["serp_clicks"].replace(0,np.nan)).fillna(0.0)
    return df

# Bot farm style (auto-extend days)
@dataclass
class BotFarmConfig:
    days: int = 20
    humans_per_day: int = 2000
    bots_per_day: int = 3000
    n_farms: int = 3
    target_rows: Optional[int] = 100_000
    max_extra_days: int = 30
    value_per_conv: float = 300.0


def generate_botfarm(cfg: BotFarmConfig = BotFarmConfig(), seed: int = 7) -> pd.DataFrame:
    random.seed(seed); np.random.seed(seed)
    queries, zipf_w, _, brand_terms = build_query_universe(600)
    ranks = np.arange(1,31); rank_w = np.linspace(0.3,0.7,len(ranks))[::-1]; rank_w /= rank_w.sum()
    farms = [
        {"name": f"farm{i+1}", "blocks": [f"52.{23+i}.{x}.0/24" for x in (10,20,30)], "active_hours": sorted(random.sample(range(24), k=10))}
        for i in range(cfg.n_farms)
    ]
    def sample_bot_ip(farm):
        block = random.choice(farm['blocks'])
        net = ipaddress.IPv4Network(block)
        host = int(net.network_address) + random.randint(1, net.num_addresses-2)
        return str(ipaddress.IPv4Address(host)), block
    rows: List[Dict[str,Any]] = []
    start_day = (datetime.utcnow() - timedelta(days=cfg.days-1)).date()
    def simulate_one(day):
        # humans
        n_h = np.random.poisson(cfg.humans_per_day)
        for _ in range(n_h):
            hour = random.randint(0,23)
            q = random.choices(queries, weights=zipf_w, k=1)[0]
            if any(x in q for x in ["buy","price","discount","coupon"]): cat = "transactional"
            elif any(x in q for x in ["near me","open now","hours","directions"]): cat = "local"
            elif any(x in q for x in ["login","official","site"]): cat = "navigational"
            else: cat = "informational"
            pos = int(np.random.choice(ranks, p=rank_w))
            impr = max(1,int(np.random.lognormal(mean=1.3,sigma=0.5)))
            base_ctr = positional_ctr(pos)
            if q in brand_terms or any(k in q for k in ("official","login")):
                base_ctr = min(0.85, base_ctr*1.3)
            clicks = np.random.binomial(impr, max(0,min(0.95,base_ctr)))
            base_cvr = {"navigational":0.11,"informational":0.035,"transactional":0.08,"local":0.05}[cat]
            conv = int(np.random.binomial(clicks, base_cvr)) if clicks>0 else 0
            rows.append({
                "event_ts": int(datetime(day.year,day.month,day.day,hour,random.randint(0,59),random.randint(0,59),tzinfo=timezone.utc).timestamp()*1000),
                "event_date": str(day),
                "hour_of_day": hour,
                "source_type": "human",
                "search_query": q,
                "query_intent_category": cat,
                "rank_position": pos,
                "serp_impressions": impr,
                "serp_clicks": clicks,
                "click_through_rate": round(clicks/impr,4),
                "conversions": conv,
                "revenue": round(conv*cfg.value_per_conv*np.random.uniform(0.9,1.1),2),
                "y_fraud": 0,
                "y_conv": int(conv>0),
            })
        # bots
        for farm in farms:
            n_b = np.random.poisson(cfg.bots_per_day / max(1,len(farms)))
            for _ in range(n_b):
                hour = random.choice(farm['active_hours'])
                q = random.choices(queries, weights=zipf_w, k=1)[0]
                if any(x in q for x in ["buy","price","discount","coupon"]): cat = "transactional"
                elif any(x in q for x in ["near me","open now","hours","directions"]): cat = "local"
                elif any(x in q for x in ["login","official","site"]): cat = "navigational"
                else: cat = "informational"
                pos = int(np.random.choice(ranks, p=rank_w))
                impr = max(1,int(np.random.lognormal(mean=1.5,sigma=0.6)))
                base_ctr = positional_ctr(pos) * 0.5
                clicks = np.random.binomial(impr, max(0,min(0.9,base_ctr)))
                ip, subnet = sample_bot_ip(farm)
                rows.append({
                    "event_ts": int(datetime(day.year,day.month,day.day,hour,random.randint(0,59),random.randint(0,59),tzinfo=timezone.utc).timestamp()*1000),
                    "event_date": str(day),
                    "hour_of_day": hour,
                    "source_type": "bot",
                    "search_query": q,
                    "query_intent_category": cat,
                    "rank_position": pos,
                    "serp_impressions": impr,
                    "serp_clicks": clicks,
                    "click_through_rate": round(clicks/impr,4),
                    "conversions": 0,
                    "revenue": 0.0,
                    "ip": ip,
                    "ip_subnet24": subnet,
                    "y_fraud": 1,
                    "y_conv": 0,
                })
    # initial window
    for d_off in range(cfg.days):
        simulate_one(start_day + timedelta(days=d_off))
    extra = 0
    while cfg.target_rows and len(rows) < cfg.target_rows and extra < cfg.max_extra_days:
        simulate_one(start_day + timedelta(days=cfg.days + extra))
        extra += 1
        if extra % 2 == 0:
            print(f"Extended generation: {extra} extra days, rows so far {len(rows):,}")
    if cfg.target_rows and len(rows) < cfg.target_rows:
        print(f"WARNING: only {len(rows):,} rows < target {cfg.target_rows:,}")
    df = pd.DataFrame(rows).sort_values('event_ts').reset_index(drop=True)
    df['ctr'] = df['click_through_rate']
    df['cvr'] = (df['conversions'] / df['serp_clicks'].replace(0,np.nan)).fillna(0.0)
    df['roas'] = (df['revenue'] / df['cost'].replace(0,np.nan)) if 'cost' in df.columns else 0
    return df

# ------------------------------ Redis helpers ------------------------------

def clear_prefix(redis_client, prefix: str) -> int:
    pipe = redis_client.pipeline(transaction=False)
    count = 0
    for k in redis_client.scan_iter(match=f"{prefix}*", count=1000):
        pipe.unlink(k); count += 1
        if count % 10000 == 0:
            pipe.execute()
    pipe.execute()
    return count


def pipeline_set_json(redis_client, df: pd.DataFrame, prefix: str = "click:", batch: int = 5000) -> int:
    pipe = redis_client.pipeline(transaction=False)
    inserted = 0
    for i, row in df.iterrows():
        pipe.set(f"{prefix}{i}", json.dumps(row.to_dict(), separators=(',',':')))
        if (i+1) % batch == 0:
            inserted += len(pipe.execute())
    inserted += len(pipe.execute())
    return inserted

__all__ = [
    "generate_basic","generate_enriched","BotFarmConfig","generate_botfarm",
    "clear_prefix","pipeline_set_json"
]
