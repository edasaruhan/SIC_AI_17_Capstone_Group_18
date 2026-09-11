import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np
import joblib
from src.features import add_features, FEATURE_COLS
from src.ctr_model import (
    train_ctr_model,
    predict_ctr,
    EXTRA_FEATURE_COLS,
    MODELS_DIR,
    MAIN_CONTENT_MODEL,
    DEMO_CONTENT_MODEL,
)


@pytest.fixture(scope="module")
def sample_df():
    from src.data_prep import load_and_clean
    # P0.1: model testleri yalnızca GERÇEK veriyle eğitilir (ana model asla
    # sentetik veriyle üzerine yazılmaz).
    df = load_and_clean()
    df = add_features(df)
    return df


def test_model_trains(sample_df):
    results = train_ctr_model(sample_df, output_prefix="test_")
    assert "xgb_metrics" in results
    assert "lr_metrics" in results
    assert results["xgb_metrics"]["r2"] is not None


def test_predict_ctr_returns_float():
    score = predict_ctr("Amazing product! Buy now!")
    assert isinstance(score, float)
    assert score >= 0


def test_predict_ctr_varies():
    short = predict_ctr("Hi")
    long = predict_ctr("This is a longer ad copy with urgency words! Limited deal now! Buy today!")
    assert short != long


def test_model_artifact_saves():
    model_path = MODELS_DIR / "xgboost_ctr_model.pkl"
    assert model_path.exists()
    model = joblib.load(model_path)
    assert hasattr(model, "predict")


def test_content_model_artifact_saves():
    assert MAIN_CONTENT_MODEL.exists()
    model = joblib.load(MAIN_CONTENT_MODEL)
    assert hasattr(model, "predict")


def test_predict_uses_dynamic_sentiment():
    pos = predict_ctr("Amazing! Best deal ever! Limited time now!")
    neg = predict_ctr("This product is terrible and broken.")
    assert pos != neg


def test_demo_model_is_a_separate_artifact():
    # P0.1: gösterim (sentetik) modeli ana modelle karışmamalı.
    assert DEMO_CONTENT_MODEL.exists()
    demo_model = joblib.load(DEMO_CONTENT_MODEL)
    main_model = joblib.load(MAIN_CONTENT_MODEL)
    assert hasattr(demo_model, "predict")
    assert hasattr(main_model, "predict")


def test_predict_with_explicit_demo_model_path():
    score = predict_ctr("Limited-time deal! Buy now, today only, don't miss it! 🔥",
                        model_path=str(DEMO_CONTENT_MODEL))
    assert isinstance(score, float)
    assert score >= 0
