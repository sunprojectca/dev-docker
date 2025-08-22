#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
import random, string, math, uuid
import argparse

QUERY_CATEGORIES = [
    ("navigational", 0.30),
    ("informational", 0.45),
    ("transactional", 0.20),
    ("local", 0.05),
]

DEVICES = [("desktop", 0.45), ("mobile", 0.5), ("tablet", 0.05)]
COUNTRIES = [
    ("US", 0.45), ("GB", 0.10), ("CA", 0.08), ("AU", 0.05),
    ("IN", 0.12), ("DE", 0.05), ("FR", 0.05), ("BR", 0.05), ("JP", 0.05)
]
LANGS = [("en", 0.7), ("en-GB", 0.1), ("en-US", 0.15), ("de", 0.02), ("fr", 0.02), ("pt", 0.01)]
SERP_FEATURES = ["none","sitelinks","faq","video","image","news","localpack","shopping"]
REFERRERS = ["google", "google_news", "google_images", "google_discover"]

def wchoice(pairs):
    items, weights = zip(*pairs)
    return random.choices(items, weights=weights, k=1)[0]

def random_query(category):
    topics = {
        "navigational": ["facebook login", "youtube", "gmail", "amazon", "wikipedia", "bank"],
        "informational": ["how to", "what is", "best way to", "benefits of", "guide to", "tutorial"],
        "transactional": ["buy", "price", "discount", "near me", "coupon", "best"],
        "local": ["near me", "closest", "open now", "hours", "directions", "map"]
    }
    nouns = ["climate change","redis","c++","python","fpga","laptop","headphones","router","ssd","credit card","insurance","pizza","coffee"]
    t = random.choice(topics[category])
    n = random.choice(nouns)
    if category == "navigational":
        return f"{n} {t}" if random.random()<0.4 else f"{t}"
    if category == "local":
        return f"{n} {t}"
    if category == "transactional":
        return f"{t} {n}"
    return f"{t} {n}"

def make_title(query):
    base = query.title()
    suffix = random.choice([" | Official Site", " – Ultimate Guide", " | Pricing", " – Reviews", "", " | Docs"])
    return base + suffix

def positional_ctr(position):
    base = 0.38 * (1 / math.log(position + 1.8))
    noise = random.uniform(-0.03, 0.03)
    ctr = max(0.0, min(0.8, base + noise))
    return ctr

def dwell_time_s(position, category, clicked):
    if not clicked:
        return 0
    base = {"navigational": 30, "informational": 80, "transactional": 60, "local": 45}[category]
    pos_factor = 1.0 + (max(0, position-1) * 0.02)
    noise = random.gauss(1.0, 0.2)
    return max(5, int(base * pos_factor * noise))

def bounce_from_dwell(dwell, device):
    if dwell == 0:
        return True
    base = 0.25 if device == "desktop" else 0.32
    p = max(0.02, base - (dwell / 300.0))
    return random.random() < p

def make_page(domain_seed=0):
    domains = [
        "https://www.example.com",
        "https://docs.example.com",
        "https://shop.example.com",
        "https://blog.example.com",
        "https://support.example.com",
        "https://news.example.com",
    ]
    d = domains[(domain_seed + random.randint(0,5)) % len(domains)]
    path = "/"+"/".join(
        "".join(random.choices(string.ascii_lowercase, k=random.randint(4,10)))
        for _ in range(random.randint(1,3))
    )
    return d + path

def generate(rows=10000, days=60, seed=1337, out="synthetic_google_clicks.csv", verbose_columns=True):
    """Generate synthetic Google-style click data.

    Parameters
    ----------
    rows : int
        Number of event rows to generate.
    days : int
        Number of days history (uniformly sampled).
    seed : int
        RNG seed for reproducibility.
    out : str
        Output CSV path.
    verbose_columns : bool
        If True (default) use descriptive column names; if False use legacy short names.
    """
    random.seed(seed)
    np.random.seed(seed)

    today = date.today()
    start = today - timedelta(days=days-1)

    records = []
    unique_queries = max(500, int(rows*0.1))
    categories_pool = [wchoice([("navigational",0.3),("informational",0.45),("transactional",0.2),("local",0.05)]) for _ in range(unique_queries)]
    queries = [random_query(categories_pool[i]) for i in range(unique_queries)]
    queries = list(dict.fromkeys(queries))
    unique_queries = len(queries)

    idx = np.arange(1, unique_queries+1)
    zipf_weights = 1 / (idx ** 1.1)
    zipf_weights = zipf_weights / zipf_weights.sum()

    serp_bias = {q: random.choice(SERP_FEATURES) for q in queries}
    brand_terms = set(q for q in queries if "official" in q or "login" in q or "gmail" in q)

    ranks = np.arange(1, 51)
    rank_weights = np.linspace(0.25, 0.75, 50)[::-1]
    rank_weights = rank_weights / rank_weights.sum()

    for _ in range(rows):
        d = start + timedelta(days=random.randint(0, days-1))
        device = wchoice(DEVICES)
        country = wchoice(COUNTRIES)
        lang = wchoice(LANGS)
        ref = random.choice(REFERRERS)

        q = random.choices(queries, weights=zipf_weights, k=1)[0]

        if any(x in q for x in ["buy", "price", "discount", "coupon"]):
            cat = "transactional"
        elif any(x in q for x in ["near me", "open now", "hours", "directions"]):
            cat = "local"
        elif any(x in q for x in ["login", "official", "site"]):
            cat = "navigational"
        else:
            cat = "informational"

        position = int(np.random.choice(ranks, p=rank_weights))
        impressions = max(1, int(np.random.lognormal(mean=1.2, sigma=0.6)))
        base_ctr = positional_ctr(position)

        if device == "mobile":
            base_ctr *= 0.95
        if country in ["US","GB","CA"]:
            base_ctr *= 1.05

        is_brand = q in brand_terms or "official" in q or "login" in q
        if is_brand:
            base_ctr = min(0.85, base_ctr * 1.4)

        p_click = max(0.0, min(0.95, base_ctr))
        clicks = np.random.binomial(impressions, p_click)
        clicked = clicks > 0

        dwell = dwell_time_s(position, cat, clicked)
        bounce = bounce_from_dwell(dwell, device)

        page = make_page(domain_seed=hash(q) % 1000)
        title = make_title(q)
        features = serp_bias[q]

        if verbose_columns:
            records.append({
                "event_date": d.isoformat(),
                "search_query": q,
                "query_intent_category": cat,
                "landing_page_url": page,
                "page_title": title,
                "device_type": device,
                "user_country": country,
                "user_language": lang,
                "traffic_referrer": ref,
                "user_id": "u_" + str(uuid.uuid4())[:8],
                "session_id": "s_" + str(uuid.uuid4())[:8],
                "rank_position": int(position),
                "serp_impressions": int(impressions),
                "serp_clicks": int(clicks),
                "click_through_rate": round((clicks / impressions) if impressions else 0.0, 4),
                "dwell_time_seconds": int(dwell),
                "bounced_session": bool(bounce),
                "serp_feature": features,
                "is_brand_query": bool(is_brand),
            })
        else:
            # Legacy short schema
            records.append({
                "date": d.isoformat(),
                "query": q,
                "query_category": cat,
                "page": page,
                "title": title,
                "device": device,
                "country": country,
                "language": lang,
                "referrer": ref,
                "user_id": "u_" + str(uuid.uuid4())[:8],
                "session_id": "s_" + str(uuid.uuid4())[:8],
                "position": int(position),
                "impressions": int(impressions),
                "clicks": int(clicks),
                "ctr": round((clicks / impressions) if impressions else 0.0, 4),
                "dwell_time_s": int(dwell),
                "bounce": bool(bounce),
                "serp_features": features,
                "is_brand": bool(is_brand),
            })

    df = pd.DataFrame.from_records(records)
    if verbose_columns:
        df.sort_values(["event_date","search_query"], inplace=True)
    else:
        df.sort_values(["date","query"], inplace=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out, df.head(20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic search click dataset")
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", type=str, default="synthetic_google_clicks.csv")
    parser.add_argument("--legacy", action="store_true", help="Use legacy short column names")
    args = parser.parse_args()
    out, head = generate(rows=args.rows, days=args.days, seed=args.seed, out=args.out, verbose_columns=not args.legacy)
    print(f"Wrote sample to {out}")
    print(head)
