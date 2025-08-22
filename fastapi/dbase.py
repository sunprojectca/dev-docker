# Synthetic Google Ads + GA-style dataset for 1 year
# and a lightweight "dashboard" in this notebook.
#
# - Creates a realistic apartment campaign dataset with campaigns,
#   ad groups, keywords, devices, and cities.
# - Saves CSV to /mnt/data/apartment_ga_ads_synth.csv
# - Displays: KPIs, campaign summary, device mix, and time-series charts.
#
# Notes for charts (per tool rules):
# - Use matplotlib (no seaborn), one chart per figure, and default colors.

import math, random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from caas_jupyter_tools import display_dataframe_to_user

random.seed(7)
np.random.seed(7)

# -----------------------------
# Config
# -----------------------------
START_DATE = datetime(2024, 9, 1)
DAYS = 365
VALUE_PER_CONVERSION = 300.0  # lead value approximation

CAMPAIGNS = {
    "Brand Search":     {"network": "Search",  "base_ctr": 0.065, "base_cpc": 0.65, "base_cv": 0.12},
    "Generic Search":   {"network": "Search",  "base_ctr": 0.032, "base_cpc": 1.40, "base_cv": 0.045},
    "Competitor":       {"network": "Search",  "base_ctr": 0.025, "base_cpc": 2.10, "base_cv": 0.035},
    "Retargeting":      {"network": "Display", "base_ctr": 0.008, "base_cpc": 0.45, "base_cv": 0.060},
    "Display Prospect": {"network": "Display", "base_ctr": 0.004, "base_cpc": 0.30, "base_cv": 0.020},
}

ADGROUPS = {
    "Brand Search":     ["Brand Core", "Brand Location"],
    "Generic Search":   ["City Apartments", "Amenities", "Near Me"],
    "Competitor":       ["Comp A", "Comp B"],
    "Retargeting":      ["Site Visitors 7d", "Abandoned Tours"],
    "Display Prospect": ["In-Market", "Affinity Home"],
}

DEVICES = {"mobile": 0.62, "desktop": 0.30, "tablet": 0.08}
DEVICE_EFFECTS = {
    "mobile":  {"ctr": 1.05, "cpc": 0.95, "cv": 0.95},
    "desktop": {"ctr": 0.95, "cpc": 1.05, "cv": 1.05},
    "tablet":  {"ctr": 0.90, "cpc": 0.90, "cv": 0.90},
}
CITIES = [
    ("United States", "CA", "Los Angeles"),
    ("United States", "CA", "San Francisco"),
    ("United States", "WA", "Seattle"),
    ("United States", "TX", "Austin"),
    ("United States", "NY", "New York"),
    ("United States", "IL", "Chicago"),
    ("United States", "FL", "Miami"),
    ("United States", "CO", "Denver"),
]
GEO_EFFECTS = {
    "CA": {"demand": 1.10, "cpc": 1.05},
    "NY": {"demand": 1.08, "cpc": 1.10},
    "WA": {"demand": 1.00, "cpc": 0.98},
    "TX": {"demand": 0.95, "cpc": 0.90},
    "IL": {"demand": 0.98, "cpc": 0.95},
    "FL": {"demand": 0.97, "cpc": 0.92},
    "CO": {"demand": 0.96, "cpc": 0.94},
}

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def seasonality_factor(day_of_year: int) -> float:
    # Peak in mid-July (moving season). Returns approx [0.8, 1.3]
    peak_center = 200
    width = 140
    x = (day_of_year - peak_center) / width
    bump = math.exp(-0.5 * (x * 2.0) ** 2)
    return 0.8 + 0.5 * bump

def weekday_factor(weekday: int) -> float:
    # Mon-Thu stronger for research, Sat/Sun weaker
    if weekday in (0,1,2,3):
        return 1.1
    elif weekday == 4:
        return 1.0
    return 0.85

def synthesize():
    rows = []
    cid, gid, aid = 1000, 5000, 90000

    for day_i in range(DAYS):
        day = START_DATE + timedelta(days=day_i)
        doy = int(day.strftime("%j"))
        wday = day.weekday()
        base_season = seasonality_factor(doy)
        wd = weekday_factor(wday)

        for camp, meta in CAMPAIGNS.items():
            cid += 1
            network = meta["network"]
            base_ctr, base_cpc, base_cv = meta["base_ctr"], meta["base_cpc"], meta["base_cv"]

            for ag in ADGROUPS[camp]:
                gid += 1

                for (country, region, city) in CITIES:
                    geo = GEO_EFFECTS.get(region, {"demand":1.0,"cpc":1.0})
                    geo_demand = geo["demand"]
                    geo_cpc_m = geo["cpc"]

                    # demand proxy ~ budget scaling per campaign
                    budget_scale = {
                        "Brand Search": 0.6,
                        "Generic Search": 1.3,
                        "Competitor": 0.5,
                        "Retargeting": 0.8,
                        "Display Prospect": 1.0,
                    }[camp]
                    daily_demand = 1200 * base_season * wd * geo_demand * budget_scale

                    for device, share in DEVICES.items():
                        dev_eff = DEVICE_EFFECTS[device]
                        exp_impr = daily_demand * share
                        impressions = int(np.random.lognormal(mean=math.log(max(exp_impr,1)), sigma=0.35))

                        ctr = clamp(np.random.normal(base_ctr * dev_eff["ctr"], base_ctr*0.15), 0.001, 0.25)
                        clicks = int(np.random.binomial(impressions, ctr)) if impressions>0 else 0

                        cpc = clamp(np.random.normal(base_cpc * dev_eff["cpc"] * geo_cpc_m, base_cpc*0.12), 0.15, 6.0)
                        cost = round(clicks * cpc, 2)

                        session_factor = 0.95 if network=="Search" else 0.75
                        sessions = int(max(0, np.random.normal(clicks * session_factor, max(1, clicks*0.08))))

                        users = int(max(0, sessions * np.random.uniform(0.85, 0.98)))
                        new_users = int(max(0, users * (np.random.uniform(0.55,0.80) if network=="Search" else np.random.uniform(0.70,0.92))))
                        engaged_sessions = int(max(0, sessions * np.random.uniform(0.55, 0.88)))
                        bounce_rate = clamp(np.random.normal(0.42 if network=="Search" else 0.58, 0.07), 0.15, 0.90)

                        cv_rate = clamp(np.random.normal(base_cv * dev_eff["cv"], base_cv*0.20), 0.002, 0.35)
                        conversions = int(np.random.binomial(max(clicks,0), cv_rate)) if clicks>0 else 0

                        revenue = round(conversions * VALUE_PER_CONVERSION * np.random.uniform(0.9, 1.1), 2)
                        cpa = round(cost / conversions, 2) if conversions>0 else None
                        roas = round(revenue / cost, 2) if cost>0 else None

                        aid += 1
                        rows.append({
                            "date": day.strftime("%Y-%m-%d"),
                            "campaign": camp,
                            "ad_group": ag,
                            "ad_id": aid,
                            "network": network,
                            "device_category": device,
                            "country": country,
                            "region": region,
                            "city": city,
                            "impressions": impressions,
                            "clicks": clicks,
                            "ctr": round((clicks/impressions),4) if impressions>0 else 0.0,
                            "cpc": round(cpc,2),
                            "cost": cost,
                            "sessions": sessions,
                            "users": users,
                            "new_users": new_users,
                            "engaged_sessions": engaged_sessions,
                            "bounce_rate": round(bounce_rate,4),
                            "conversions": conversions,
                            "conversion_rate": round(cv_rate,4),
                            "cost_per_conversion": cpa,
                            "revenue": revenue,
                            "roas": roas
                        })
    df = pd.DataFrame(rows)
    # Derive consistent columns
    df["ctr"] = (df["clicks"]/df["impressions"]).replace([np.inf,-np.inf],0).fillna(0).round(4)
    df["cost_per_conversion"] = np.where(df["conversions"]>0, (df["cost"]/df["conversions"]).round(2), np.nan)
    df["roas"] = np.where(df["cost"]>0, (df["revenue"]/df["cost"]).round(2), np.nan)
    return df

df = synthesize()

# Save a CSV for you to download
out_path = Path("/mnt/data/apartment_ga_ads_synth.csv")
df.to_csv(out_path, index=False)

# -----------------------------
# KPI rollups
# -----------------------------
daily = df.groupby("date", as_index=False).agg({
    "impressions":"sum","clicks":"sum","cost":"sum",
    "conversions":"sum","revenue":"sum","sessions":"sum"
})
daily["ctr"] = (daily["clicks"]/daily["impressions"]).replace([np.inf,-np.inf],0).fillna(0)
daily["cpc"] = (daily["cost"]/daily["clicks"]).replace([np.inf,-np.inf],0).fillna(0)
daily["cpa"] = (daily["cost"]/daily["conversions"]).replace([np.inf,-np.inf],np.nan)
daily["roas"] = (daily["revenue"]/daily["cost"]).replace([np.inf,-np.inf],np.nan)

camp = df.groupby("campaign", as_index=False).agg({
    "impressions":"sum","clicks":"sum","cost":"sum",
    "conversions":"sum","revenue":"sum","sessions":"sum"
})
camp["ctr"] = (camp["clicks"]/camp["impressions"]).round(4)
camp["cpc"] = (camp["cost"]/camp["clicks"]).replace([np.inf,-np.inf],0).round(2)
camp["cvr"] = (camp["conversions"]/camp["clicks"]).replace([np.inf,-np.inf],0).round(4)
camp["cpa"] = (camp["cost"]/camp["conversions"]).replace([np.inf,-np.inf],np.nan).round(2)
camp["roas"] = (camp["revenue"]/camp["cost"]).replace([np.inf,-np.inf],np.nan).round(2)

device = df.groupby("device_category", as_index=False).agg({
    "impressions":"sum","clicks":"sum","cost":"sum","conversions":"sum","revenue":"sum"
})
device["ctr"] = (device["clicks"]/device["impressions"]).round(4)
device["cpc"] = (device["cost"]/device["clicks"]).replace([np.inf,-np.inf],0).round(2)
device["cpa"] = (device["cost"]/device["conversions"]).replace([np.inf,-np.inf],np.nan).round(2)
device["roas"] = (device["revenue"]/device["cost"]).replace([np.inf,-np.inf],np.nan).round(2)

# Show summary tables in a spreadsheet-style view
display_dataframe_to_user("Daily KPI (All Campaigns)", daily)
display_dataframe_to_user("Campaign Performance Summary", camp)
display_dataframe_to_user("Device Mix Summary", device)

# -----------------------------
# "Dashboard" charts
# -----------------------------
# Line: daily clicks
plt.figure()
plt.plot(pd.to_datetime(daily["date"]), daily["clicks"])
plt.title("Daily Clicks")
plt.xlabel("Date"); plt.ylabel("Clicks")
plt.tight_layout()
plt.show()

# Line: daily cost
plt.figure()
plt.plot(pd.to_datetime(daily["date"]), daily["cost"])
plt.title("Daily Cost")
plt.xlabel("Date"); plt.ylabel("Cost")
plt.tight_layout()
plt.show()

# Line: daily conversions
plt.figure()
plt.plot(pd.to_datetime(daily["date"]), daily["conversions"])
plt.title("Daily Conversions")
plt.xlabel("Date"); plt.ylabel("Conversions")
plt.tight_layout()
plt.show()

# Line: rolling ROAS (7-day)
daily_sorted = daily.sort_values("date").copy()
daily_sorted["roas_roll7"] = daily_sorted["roas"].rolling(7, min_periods=1).mean()
plt.figure()
plt.plot(pd.to_datetime(daily_sorted["date"]), daily_sorted["roas_roll7"])
plt.title("ROAS (7-day Rolling)")
plt.xlabel("Date"); plt.ylabel("ROAS")
plt.tight_layout()
plt.show()

# Pie: device share by clicks
plt.figure()
plt.pie(device["clicks"], labels=device["device_category"], autopct="%1.1f%%")
plt.title("Device Share (Clicks)")
plt.tight_layout()
plt.show()

print(f"Saved CSV to: {out_path}")
