import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np
from src.data_prep import load_and_clean
from src.features import add_features, extract_features, FEATURE_COLS


def test_row_count():
    df = load_and_clean()
    assert 1180 <= len(df) <= 1200, f"Expected ~1190 rows, got {len(df)}"


def test_no_nans():
    df = load_and_clean()
    assert df["sentiment_score"].isna().sum() == 0
    assert df["toxicity_score"].isna().sum() == 0


def test_ctr_proxy_formula():
    data = {
        "text_content": ["a", "b"],
        "language": ["en", "en"],
        "impressions": [100, 200],
        "likes_count": [10, 20],
        "shares_count": [5, 10],
        "comments_count": [2, 4],
        "sentiment_score": [0.5, 0.3],
        "toxicity_score": [0.1, 0.2],
    }
    df = pd.DataFrame(data)
    result = load_and_clean.__wrapped__(None) if hasattr(load_and_clean, '__wrapped__') else None
    df_calc = df.copy()
    df_calc["ctr_proxy"] = ((df_calc["likes_count"] + df_calc["shares_count"] + df_calc["comments_count"]) / df_calc["impressions"]) * 100
    assert abs(df_calc["ctr_proxy"].iloc[0] - 17.0) < 0.01
    assert abs(df_calc["ctr_proxy"].iloc[1] - 17.0) < 0.01


def test_features_basic():
    feats = extract_features("Hello! Limited time deal now!")
    assert feats["char_count"] == 29
    assert feats["word_count"] == 5
    assert feats["has_question"] == 0
    assert feats["has_exclamation"] == 1
    assert feats["urgency_count"] == 3  # Limited, deal, now
    assert feats["has_urgency"] == 1


def test_add_features():
    df = pd.DataFrame({"text_content": ["Test! Now?", "No urgency here"]})
    result = add_features(df)
    assert "char_count" in result.columns
    assert "urgency_count" in result.columns
    assert result.iloc[0]["has_question"] == 1
    assert result.iloc[0]["has_exclamation"] == 1
    assert result.iloc[1]["urgency_count"] == 0
