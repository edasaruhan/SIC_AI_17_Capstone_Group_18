"""P2.1 — Ablasyon tablosu (gerçek veri).

Model × özellik alt kümesi karşılaştırması, 5-fold CV R² (mean).
Sonuç reports/ablation_results.csv olarak kaydedilir ve log'a basılır.

Alt kümeler:
  structural : char/word/emoji/hashtag/mention/exclamation/diversity... (yapısal)
  +affect    : yapısal + duygu/toksisite
  +action    : +affect + aciliyet/CTA/soru  (tam CONTENT_FEATURE_COLS)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.data_prep import load_and_clean
from src.features import add_features
from src.ctr_model import CONTENT_FEATURE_COLS, XGB_PARAMS

STRUCTURAL = [
    "char_count", "word_count", "emoji_count", "hashtag_count", "has_hashtag",
    "has_mention", "has_exclamation", "avg_word_len", "word_diversity",
    "all_caps_ratio", "numeric_ratio",
]
AFFECT = ["sentiment_score", "toxicity_score"]
ACTION = ["urgency_count", "has_cta", "has_question"]

SUBSETS = {
    "structural": STRUCTURAL,
    "structural+sentiment/tox": STRUCTURAL + AFFECT,
    "structural+full (tam)": STRUCTURAL + AFFECT + ACTION,
}

MODELS = {
    "Linear": LinearRegression,
    "RandomForest": lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": lambda: XGBRegressor(**XGB_PARAMS),
}

KFOLD = KFold(n_splits=5, shuffle=True, random_state=42)
SEED = 42


def main():
    df = load_and_clean()
    df = add_features(df)
    y = np.log1p(df["ctr_proxy"])

    rows = []
    for sub_name, cols in SUBSETS.items():
        X = df[cols].fillna(0)
        for model_name, builder in MODELS.items():
            scores = cross_val_score(builder(), X, y, cv=KFOLD,
                                     scoring="r2", n_jobs=-1)
            rows.append({
                "subset": sub_name,
                "model": model_name,
                "cv_r2_mean": round(float(scores.mean()), 4),
                "cv_r2_std": round(float(scores.std()), 4),
            })
            print(f"{sub_name:>28} | {model_name:>12} | R² = {scores.mean():.4f} ± {scores.std():.4f}")

    df_out = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent / "reports" / "ablation_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out, index=False)
    print(f"\nKaydedildi: {out}")

    # Basit güdü: en yüksek CV R² 'full subsets'te mi? (doğal olarak)
    best = df_out.loc[df_out["cv_r2_mean"].idxmax()]
    print(f"En iyi: {best['model']} / {best['subset']} (R²={best['cv_r2_mean']})")


if __name__ == "__main__":
    main()