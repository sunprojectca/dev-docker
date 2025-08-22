#!/usr/bin/env python3
"""
Generate enriched synthetic search click data with labels for conversion and fraud detection.

- Adds many realistic fields: geo, device, UA, IP/ASN class, referrer, time-of-day, campaign/adgroup, costs, etc.
- Produces correlated labels:
    * y_conv (1 if at least one conversion on the row)
    * y_fraud (1 if row dominated by fraudulent clicks)
- Keeps CTR/CVR logic plausible; adjusts by intent, device, geo, position, brand, and fraudiness.

Usage:
  python generate_search_clicks_enriched.py --rows 50000 --days 90 --seed 7 --fraud_rate 0.08 --out enriched_clicks.csv

The output CSV is suitable for XGBoost / sklearn.
"""
import argparse, random, string, math, uuid, ipaddress
from datetime import date, timedelta, datetime, timezone
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

QUERY_CATEGORIES = [("navigational", 0.28), ("informational", 0.44), ("transactional", 0.23), ("local", 0.05)]
DEVICES = [("desktop", 0.46), ("mobile", 0.47), ("tablet", 0.07)]
COUNTRIES = [("US", 0.45), ("GB", 0.10), ("CA", 0.08), ("AU", 0.05), ("IN", 0.12), ("DE", 0.05), ("FR", 0.05), ("BR", 0.05), ("JP", 0.05)]
LANGS = [("en", 0.55), ("en-GB", 0.10), ("en-US", 0.20), ("de", 0.03), ("fr", 0.03), ("pt", 0.03), ("hi", 0.02), ("ja", 0.02), ("es", 0.02)]
SERP_FEATURES = ["none","sitelinks","faq","video","image","news","localpack","shopping"]
REFERRERS = ["google", "google_news", "google_images", "google_discover"]

BROWSERS = [("Chrome",0.62),("Safari",0.18),("Edge",0.10),("Firefox",0.08),("Other",0.02)]
OSES = [("Windows",0.35),("Android",0.32),("iOS",0.20),("macOS",0.10),("Linux",0.03)]

# ISP/ASN class: influences fraudiness
ISP_CLASSES = [("residential",0.74),("mobile_carrier",0.18),("datacenter",0.08)]

CAMPAIGNS = ["Brand", "Generic", "Competitor", "Retargeting", "DisplayProspecting"]
ADGROUPS = {
    "Brand": ["Brand Core", "Brand Location"],
    "Generic": ["Apartments City", "Amenities", "Near Me"],
    "Competitor": ["Comp A", "Comp B"],
    "Retargeting": ["Visitors 7d", "Abandoned Tours"],
    "DisplayProspecting": ["In-Market", "Affinity Home"],
}
CREATIVE_TYPES = ["RSA", "Text", "Display", "Video"]

def wchoice(pairs):
    items, weights = zip(*pairs)
    return random.choices(items, weights=weights, k=1)[0]

def random_query(category):
    topics = {
        "navigational": ["facebook login", "youtube", "gmail", "amazon", "wikipedia", "bank", "official site"],
        "informational": ["how to", "what is", "best way to", "benefits of", "guide to", "tutorial"],
        "transactional": ["buy", "price", "discount", "near me", "coupon", "best"],
        "local": ["near me", "closest", "open now", "hours", "directions", "map"]
    }
    nouns = ["apartments","redis","python","fpga","laptop","headphones","router","ssd","credit card","insurance","pizza","coffee","gym"]
    t = random.choice(topics[category]); n = random.choice(nouns)
    if category in ("navigational","local"): return f"{n} {t}"
    return f"{t} {n}"

def make_title(query):
    base = query.title()
    suffix = random.choice([" | Official Site", " – Ultimate Guide", " | Pricing", " – Reviews", "", " | Docs"])
    return base + suffix

def positional_ctr(position):
    # CTR drops with rank; cap to [0, 0.8]
    base = 0.38 * (1 / math.log(position + 1.8))
    noise = random.uniform(-0.03, 0.03)
    return max(0.0, min(0.8, base + noise))

def dwell_time_s(position, category, clicked, fraud_weight):
    if not clicked:
        return 0
    base = {"navigational": 35, "informational": 85, "transactional": 65, "local": 45}[category]
    pos_factor = 1.0 + (max(0, position-1) * 0.02)
    # Fraud shrinks dwell time substantially
    fraud_factor = 1.0 - 0.8*fraud_weight
    noise = random.gauss(1.0, 0.25)
    return max(3, int(base * pos_factor * fraud_factor * noise))

def bounce_from_dwell(dwell, device, fraud_weight):
    if dwell == 0:
        return True
    base = 0.22 if device == "desktop" else 0.30
    # Fraud raises bounce
    p = max(0.05, min(0.97, base + 0.6*fraud_weight - (dwell / 300.0)))
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

def random_ipv4_block(isp_class):
    # Return an IPv4 address string; rough blocks for "datacenter" vs residential/mobile
    import ipaddress
    if isp_class == "datacenter":
        base = ipaddress.IPv4Network("52.23.0.0/16")  # e.g., AWS-ish range
    elif isp_class == "mobile_carrier":
        base = ipaddress.IPv4Network("100.64.0.0/16") # CGNAT-like
    else:
        base = ipaddress.IPv4Network("73.162.0.0/16") # cable/residential-ish
    host = int(base.network_address) + random.randint(1, 65534)
    return str(ipaddress.IPv4Address(host))

def ua_string(browser, os_name, headless=False):
    base = f"{browser}/{random.randint(70,125)} ({os_name})"
    return base + (" Headless" if headless else "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50000)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="enriched_clicks.csv")
    ap.add_argument("--fraud_rate", type=float, default=0.08, help="Base rate of fraudulent rows (0..1)")
    ap.add_argument("--value_per_conv", type=float, default=300.0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    today = date.today(); start = today - timedelta(days=args.days-1)

    # Build a query universe with Zipf distribution
    unique_queries = max(500, int(args.rows*0.12))
    cats = [wchoice(QUERY_CATEGORIES) for _ in range(unique_queries)]
    queries = [random_query(cats[i]) for i in range(unique_queries)]
    queries = list(dict.fromkeys(queries)); unique_queries = len(queries)
    idx = np.arange(1, unique_queries+1)
    zipf_weights = 1 / (idx ** 1.1); zipf_weights = zipf_weights / zipf_weights.sum()

    # Random SERP features + brand detection
    serp_bias = {q: random.choice(SERP_FEATURES) for q in queries}
    brand_terms = set(q for q in queries if "official" in q or "login" in q or "gmail" in q)

    ranks = np.arange(1, 51)
    rank_weights = np.linspace(0.25, 0.75, 50)[::-1]; rank_weights = rank_weights / rank_weights.sum()

    # Campaign structure
    def pick_campaign():
        c = random.choice(CAMPAIGNS)
        g = random.choice(ADGROUPS[c])
        ad_id = random.randint(100000, 999999)
        creative = random.choice(CREATIVE_TYPES)
        return c, g, ad_id, creative

    rows = []
    for _ in range(args.rows):
        # Time & basic context
        d = start + timedelta(days=random.randint(0, args.days-1))
        hour = random.choices(range(24), weights=[3,2,2,2,2,2,3,4,5,6,6,6,6,6,6,6,7,8,7,6,5,4,3,3], k=1)[0]
        wday = d.weekday()

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
        impressions = max(1, int(np.random.lognormal(mean=1.3, sigma=0.6)))
        base_ctr = positional_ctr(position)

        # Device & geo adjustments
        if device == "mobile": base_ctr *= 0.95
        if country in ["US","GB","CA"]: base_ctr *= 1.05
        is_brand = q in brand_terms or "official" in q or "login" in q
        if is_brand: base_ctr = min(0.85, base_ctr * 1.4)

        # Campaign & economic signals
        campaign, adgroup, ad_id, creative = pick_campaign()
        bid_cpc = {"Brand":0.6,"Generic":1.6,"Competitor":2.3,"Retargeting":0.8,"DisplayProspecting":0.4}[campaign]
        est_cpc = max(0.15, np.random.normal(bid_cpc, bid_cpc*0.15))
        quality_score = max(1, min(10, int(np.random.normal(7 if is_brand else 5, 2))))

        # ISP / UA / IP features
        isp_class = wchoice(ISP_CLASSES)
        headless = (isp_class == "datacenter") and (random.random() < 0.35)
        browser = wchoice(BROWSERS); os_name = wchoice(OSES)
        ua = ua_string(browser, os_name, headless=headless)
        ip = random_ipv4_block(isp_class)
        ip_subnet24 = ".".join(ip.split(".")[:3]) + ".0/24"

        # Base fraud score from telltales
        fraud_weight = 0.0
        if isp_class == "datacenter": fraud_weight += 0.5
        if headless: fraud_weight += 0.3
        if lang not in ("en","en-US","en-GB") and country in ("US","GB","CA"): fraud_weight += 0.2
        if device == "desktop" and os_name in ("Android","iOS"): fraud_weight += 0.3  # mismatch
        fraud_weight = max(0.0, min(1.0, fraud_weight))

        # Probability of fraud for this row
        p_fraud = min(0.95, args.fraud_rate*0.8 + 0.6*fraud_weight)
        is_fraud = 1 if random.random() < p_fraud else 0

        # Clicks
        p_click = max(0.0, min(0.97, base_ctr * (0.65 if is_fraud else 1.0)))
        clicks = np.random.binomial(impressions, p_click)
        clicked = clicks > 0

        # Dwell & bounce
        dwell = dwell_time_s(position, cat, clicked, fraud_weight=(0.7 if is_fraud else 0.0))
        bounce = bounce_from_dwell(dwell, device, fraud_weight=(0.7 if is_fraud else 0.0))

        # Costs & conversions
        cpc = est_cpc * (0.8 + random.random()*0.4) * (0.7 if is_brand else 1.0)
        cost = round(clicks * cpc, 2)
        base_cvr = {"navigational":0.11, "informational":0.03, "transactional":0.07, "local":0.05}[cat]
        base_cvr *= (1.15 if country in ["US","GB","CA"] else 0.9)
        base_cvr *= (1.1 if device=="desktop" else 0.95)
        # Fraud kills conversion
        cvr = max(0.0005, min(0.35, base_cvr * (0.05 if is_fraud else 1.0)))
        conversions = int(np.random.binomial(clicks, cvr)) if clicks>0 else 0
        revenue = round(conversions * args.value_per_conv * np.random.uniform(0.9, 1.1), 2)
        y_conv = 1 if conversions > 0 else 0
        y_fraud = is_fraud

        # Session/UX features
        js_enabled = (random.random() > (0.65 if is_fraud else 0.05))
        mouse_moves = (random.random() > (0.7 if is_fraud else 0.15))
        viewport_w = random.choice([360, 390, 414, 768, 1280, 1366, 1440, 1536, 1920])
        viewport_h = random.choice([640, 780, 896, 1024, 800, 900, 1080])
        cookie_age_days = int(np.random.exponential(scale=15 if is_fraud else 60))
        session_depth = int(np.random.poisson(1 if is_fraud else 3))
        pages_viewed = max(1, session_depth + (0 if bounce else random.randint(1,3)))
        ttf_interact_ms = 0 if not clicked else int(max(50, np.random.normal(800 if is_fraud else 2500, 500)))

        page = make_page(domain_seed=hash(q) % 1000)
        title = make_title(q)
        features = serp_bias[q]

        # Build record
        event_dt = datetime(d.year, d.month, d.day, hour, random.randint(0,59), random.randint(0,59), tzinfo=timezone.utc)
        record = {
            "event_ts": int(event_dt.timestamp()*1000),
            "event_date": d.isoformat(),
            "hour_of_day": hour,
            "day_of_week": wday,

            "search_query": q,
            "query_intent_category": cat,
            "is_brand_query": bool(is_brand),

            "campaign": campaign,
            "ad_group": adgroup,
            "ad_id": ad_id,
            "creative_type": creative,
            "quality_score": quality_score,
            "bid_cpc": round(bid_cpc,2),

            "rank_position": int(position),
            "serp_impressions": int(impressions),
            "serp_clicks": int(clicks),
            "click_through_rate": round((clicks / impressions) if impressions else 0.0, 4),

            "device_type": device,
            "os": os_name,
            "browser": browser,
            "user_agent": ua,

            "user_country": country,
            "user_language": lang,
            "traffic_referrer": ref,
            "serp_feature": features,

            "landing_page_url": page,
            "page_title": title,

            "ip": ip,
            "ip_subnet24": ip_subnet24,
            "isp_class": isp_class,

            "user_id": "u_" + str(uuid.uuid4())[:8],
            "session_id": "s_" + str(uuid.uuid4())[:8],

            "dwell_time_seconds": int(dwell),
            "bounced_session": bool(bounce),
            "pages_viewed": int(pages_viewed),
            "session_depth": int(session_depth),
            "time_to_first_interaction_ms": int(ttf_interact_ms),
            "viewport_width": int(viewport_w),
            "viewport_height": int(viewport_h),
            "js_enabled": bool(js_enabled),
            "mouse_moves": bool(mouse_moves),
            "cookie_age_days": int(cookie_age_days),

            "cpc": round(cpc,2),
            "cost": float(cost),
            "conversions": int(conversions),
            "revenue": float(revenue),

            # Labels
            "y_conv": int(y_conv),
            "y_fraud": int(y_fraud),

            # Derived placeholders (filled later)
            # "ctr", "cvr", "roas"

            # Provenance
            "is_synthetic": True,
            "synthetic_source": "clicks-enriched-v1"
        }
        rows.append(record)

    df = pd.DataFrame.from_records(rows).sort_values(["event_ts","search_query"]).reset_index(drop=True)

    # Helpful derived numeric features (no leakage)
    df["ctr"] = df["click_through_rate"]
    df["cvr"] = (df["conversions"] / df["serp_clicks"].replace(0, np.nan)).fillna(0.0)
    df["roas"] = (df["revenue"] / df["cost"].replace(0, np.nan)).replace([np.inf,-np.inf], 0).fillna(0.0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows to {out}")
    print(df.head(10))

if __name__ == "__main__":
    main()
