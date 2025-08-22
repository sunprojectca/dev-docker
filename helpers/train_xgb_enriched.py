"""Train fraud and conversion XGBoost classifiers from enriched CSV.

Usage (PowerShell):
  python helpers/train_xgb_enriched.py -i path\to\enriched_clicks.csv
"""
import argparse, os, pandas as pd, xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def load_df(path: str):
    return pd.read_csv(path)

def build_features(df: pd.DataFrame):
    cat_cols = ["query_intent_category","device_type","os","browser","user_country","user_language","traffic_referrer","serp_feature","campaign","ad_group","creative_type","isp_class"]
    num_cols = [c for c in ["hour_of_day","day_of_week","rank_position","serp_impressions","serp_clicks","ctr","quality_score","bid_cpc","cpc","cost","dwell_time_seconds","pages_viewed","session_depth","time_to_first_interaction_ms","cookie_age_days","roas"] if c in df.columns]
    df_cats = pd.get_dummies(df[cat_cols], dummy_na=True, drop_first=True)
    X = pd.concat([df[num_cols], df_cats], axis=1)
    return X

def train_model(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()>1 else None)
    model = xgb.XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.08, subsample=0.85, colsample_bytree=0.8, tree_method="hist", eval_metric="auc")
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:,1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(yte, proba)
    print("ROC AUC", round(auc,4))
    print(classification_report(yte, pred, digits=3))
    return model, list(X.columns)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--input', default=os.getenv('CSV_PATH','/shared/enriched_clicks.csv'))
    ap.add_argument('-o','--outdir', default='models')
    ap.add_argument('--no-conv', action='store_true')
    ap.add_argument('--no-fraud', action='store_true')
    args = ap.parse_args()
    df = load_df(args.input)
    X = build_features(df)
    os.makedirs(args.outdir, exist_ok=True)
    if not args.no_fraud and 'y_fraud' in df.columns:
        print('== FRAUD MODEL ==')
        m, feats = train_model(X, df['y_fraud'].astype(int))
        m.save_model(os.path.join(args.outdir, 'xgb_fraud.json'))
    if not args.no_conv and 'y_conv' in df.columns:
        print('== CONVERSION MODEL ==')
        m2, feats2 = train_model(X, df['y_conv'].astype(int))
        m2.save_model(os.path.join(args.outdir, 'xgb_conv.json'))

if __name__ == '__main__':
    main()
