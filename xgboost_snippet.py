
# Example: train two XGBoost models on the enriched CSV
#   1) Fraud classifier (y_fraud)
#   2) Conversion classifier (y_conv)
import os, pandas as pd, xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

csv = os.getenv("CSV_PATH", "/shared/enriched_clicks.csv")
df = pd.read_csv(csv)

labels = {"fraud":"y_fraud", "conv":"y_conv"}
cat_cols = ["query_intent_category","device_type","os","browser","user_country","user_language",
            "traffic_referrer","serp_feature","campaign","ad_group","creative_type","isp_class"]
num_cols = ["hour_of_day","day_of_week","rank_position","serp_impressions","serp_clicks","ctr","quality_score",
            "bid_cpc","cpc","cost","dwell_time_seconds","pages_viewed","session_depth","time_to_first_interaction_ms",
            "cookie_age_days","roas"]

# One-hot cats (simple and effective for XGB too)
df_cats = pd.get_dummies(df[cat_cols], dummy_na=True)
X = pd.concat([df[num_cols], df_cats], axis=1)

for name, target in labels.items():
    y = df[target].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", eval_metric="auc"
    )
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:,1]
    pred = (proba >= 0.5).astype(int)
    print(f"== {name.upper()} ==")
    print("ROC AUC:", roc_auc_score(yte, proba))
    print(classification_report(yte, pred, digits=3))
