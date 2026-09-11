import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from .features import extract_features
from .sentiment import analyze_sentiment, analyze_toxicity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Content-only columns: used to score NEW ad copy (no impression/buzz context exists).
# These must be exactly what `extract_features` + dynamic sentiment/toxicity produce.
CONTENT_FEATURE_COLS = [
    "char_count", "word_count", "has_question", "has_exclamation",
    "urgency_count", "sentiment_score", "toxicity_score",
    "emoji_count", "hashtag_count", "has_hashtag", "has_mention", "has_cta",
    "avg_word_len", "word_diversity", "all_caps_ratio", "numeric_ratio",
]

# Full (contextual) model: content + impression/buzz context.
EXTRA_FEATURE_COLS = CONTENT_FEATURE_COLS + [
    "log_impressions", "buzz_change_rate",
]

_XGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.03, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, gamma=0.1,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42,
)
XGB_PARAMS = _XGB_PARAMS


def _augment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_impressions"] = np.log1p(df["impressions"])
    return df


def _evaluate(model, X_test, y_test_orig, y_test_log) -> dict:
    y_pred = np.expm1(model.predict(X_test))
    return {
        "r2": float(r2_score(y_test_orig, y_pred)),
        "rmse": float(root_mean_squared_error(y_test_orig, y_pred)),
        "mae": float(mean_absolute_error(y_test_orig, y_pred)),
    }


def _cv_r2(model_cls, X_train, y_train_log) -> float:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, va_idx in kf.split(X_train):
        Xtr, Xva = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        ytr = y_train_log.iloc[tr_idx]
        yva_log = y_train_log.iloc[va_idx]
        m = model_cls()
        m.fit(Xtr, ytr)
        preds = np.expm1(m.predict(Xva))
        scores.append(r2_score(np.expm1(yva_log), preds))
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    logger.info(f"5-fold CV R2: mean={mean:.4f}, std={std:.4f}")
    return mean


# Prefix-based artifact paths (main: "", demo: "synthetic_demo_")
MAIN_CONTENT_MODEL = MODELS_DIR / "xgboost_content_model.pkl"
DEMO_CONTENT_MODEL = MODELS_DIR / "synthetic_demo_xgboost_content_model.pkl"


def train_ctr_model(df: pd.DataFrame, output_prefix: str = "") -> dict:
    """Full model (context features) + content-only model. Saved to models/ with prefix."""
    df = _augment(df)
    y_log = np.log1p(df["ctr_proxy"].copy())
    y = df["ctr_proxy"].copy()

    X_train, X_test, y_train_log, y_test_log, y_train, y_test = train_test_split(
        df[EXTRA_FEATURE_COLS].fillna(0), y_log, y, test_size=0.2, random_state=42
    )

    xgb = XGBRegressor(**_XGB_PARAMS)
    xgb.fit(X_train, y_train_log)
    cv_r2_mean = _cv_r2(lambda: XGBRegressor(**_XGB_PARAMS), X_train, y_train_log)
    xgb_metrics = _evaluate(xgb, X_test, y_test, y_test_log)
    logger.info(
        f"XGBoost (full) test R2={xgb_metrics['r2']:.4f}, RMSE={xgb_metrics['rmse']:.4f}, "
        f"MAE={xgb_metrics['mae']:.4f}"
    )

    lr = LinearRegression()
    lr.fit(X_train, y_train_log)
    lr_metrics = _evaluate(lr, X_test, y_test, y_test_log)
    logger.info(
        f"LinearReg (full) test R2={lr_metrics['r2']:.4f}, RMSE={lr_metrics['rmse']:.4f}, "
        f"MAE={lr_metrics['mae']:.4f}"
    )

    full_path = MODELS_DIR / f"{output_prefix}xgboost_ctr_model.pkl"
    joblib.dump(xgb, full_path)
    logger.info(f"Full model saved to {full_path}")

    full_importances = dict(zip(
        EXTRA_FEATURE_COLS, [float(v) for v in xgb.feature_importances_]
    ))
    (MODELS_DIR / f"{output_prefix}feature_importance_full.json").write_text(
        json.dumps(full_importances, indent=2)
    )

    # --- Content-only model (used to score new ad copy) ---------------------
    Xc = df[CONTENT_FEATURE_COLS].fillna(0)
    Xc_train, Xc_test = X_train[CONTENT_FEATURE_COLS], X_test[CONTENT_FEATURE_COLS]
    xgb_content = XGBRegressor(**_XGB_PARAMS)
    xgb_content.fit(Xc_train, y_train_log)
    content_cv = _cv_r2(
        lambda: XGBRegressor(**_XGB_PARAMS), Xc_train, y_train_log
    )
    content_metrics = _evaluate(xgb_content, Xc_test, y_test, y_test_log)
    logger.info(
        f"XGBoost (content) test R2={content_metrics['r2']:.4f}, RMSE={content_metrics['rmse']:.4f}, "
        f"MAE={content_metrics['mae']:.4f}"
    )

    content_path = MODELS_DIR / f"{output_prefix}xgboost_content_model.pkl"
    joblib.dump(xgb_content, content_path)
    logger.info(f"Content model saved to {content_path}")

    content_importances = dict(zip(
        CONTENT_FEATURE_COLS, [float(v) for v in xgb_content.feature_importances_]
    ))
    imp_path = MODELS_DIR / f"{output_prefix}feature_importance.json"
    imp_path.write_text(json.dumps(content_importances, indent=2))
    logger.info(f"Content feature importances saved to {imp_path}")

    return {
        "xgb_metrics": xgb_metrics,
        "lr_metrics": lr_metrics,
        "cv_r2_mean": cv_r2_mean,
        "cv_r2_std": 0.0,
        "feature_importances": full_importances,
        "content_metrics": content_metrics,
        "content_cv_r2_mean": content_cv,
        "content_feature_importances": content_importances,
    }


def predict_ctr(text: str, model_path: str | None = None) -> float:
    """New-ad-copy CTR proxy score, via the content-only model.

    Falls back to the full model (log_impressions=0) if the content artifact
    is missing (e.g., pre-migration checkout). Raises FileNotFoundError only
    when neither artifact exists.
    """
    content_path = Path(model_path) if model_path else MAIN_CONTENT_MODEL
    full_path = MODELS_DIR / "xgboost_ctr_model.pkl"

    feats = extract_features(text)
    feats["sentiment_score"] = analyze_sentiment(text)
    feats["toxicity_score"] = analyze_toxicity(text)

    if content_path.exists():
        model = joblib.load(content_path)
        X = pd.DataFrame([feats])[CONTENT_FEATURE_COLS]
        return float(max(0.0, np.expm1(model.predict(X)[0])))

    if full_path.exists():
        logger.warning(
            "Content model bulunamadı (%s), full model ile tahmin yapılıyor", content_path
        )
        model = joblib.load(full_path)
        feats["log_impressions"] = 0.0
        feats["buzz_change_rate"] = 0.0
        X = pd.DataFrame([feats])[EXTRA_FEATURE_COLS]
        return float(max(0.0, np.expm1(model.predict(X)[0])))

    raise FileNotFoundError(
        f"Model bulunamadı. Önce eğitim çalıştırın (train_ctr_model): {content_path}"
    )
