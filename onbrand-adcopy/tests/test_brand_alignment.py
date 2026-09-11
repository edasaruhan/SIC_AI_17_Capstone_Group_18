import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.brand_alignment import brand_alignment_score, load_brand_data, validate_alignment


@pytest.fixture(scope="module")
def brand_data():
    return load_brand_data()


def test_own_brand_scores_higher(brand_data):
    results = validate_alignment(brand_data)
    for bid in brand_data:
        assert results[bid]["own_higher"], f"{bid}: own-brand should score higher than cross-brand"


def test_samsung_reference_high(brand_data):
    score = brand_alignment_score("Unfold what's next.", "samsung", brand_data)
    assert score >= 20, f"Samsung reference should score high, got {score}"


def test_cross_brand_lower_than_own(brand_data):
    nike_ref = brand_data["nike"]["reference_ad_copies"][0]
    nike_score = brand_alignment_score(nike_ref, "nike", brand_data)
    duolingo_score = brand_alignment_score(nike_ref, "duolingo", brand_data)
    assert nike_score > duolingo_score, "Nike reference should score higher for Nike than Duolingo"


def test_rule_based_check(brand_data):
    from src.brand_alignment import rule_based_check
    result = rule_based_check("Great phone with seamless ecosystem", "samsung", brand_data)
    assert result["passed"] is True
    result2 = rule_based_check("Maybe you should try to run?", "nike", brand_data)
    assert result2["passed"] is False


def test_score_never_exactly_zero(brand_data):
    # P1.1: gerçek metinler tam 0.0 almamalı (SCORE_FLOOR=0.1).
    for bid in brand_data:
        score = brand_alignment_score("Sıradan, markadan bağımsız bir metin burada.", bid, brand_data)
        assert score > 0, f"{bid}: alignment score must never be 0.0, got {score}"


def test_alignment_proof_table_exists():
    # P1.1: raporlanan kanıt tablosu üretilmiş olmalı ve tüm skorlar > 0 olmalı.
    import pandas as pd
    proof = Path(__file__).resolve().parent.parent / "reports" / "brand_alignment_proof.csv"
    assert proof.exists(), "scripts/p1_brand_proof.py çalıştırılarak üretilmeli"
    df = pd.read_csv(proof)
    assert len(df) == 10
    score_cols = [c for c in df.columns if c.startswith("score_")]
    assert (df[score_cols] > 0).all().all()
