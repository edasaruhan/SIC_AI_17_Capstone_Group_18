import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from src.composite import composite_score, rank_variants
from src.brand_alignment import load_brand_data


@pytest.fixture(scope="module")
def brand_data():
    return load_brand_data()


def test_composite_score_basic(brand_data):
    result = composite_score(
        "Unfold what's next.", "samsung", 0.6, brand_data=brand_data
    )
    assert "ctr_score" in result
    assert "brand_score" in result
    assert "composite_score" in result
    assert 0 <= result["composite_score"] <= 100


def test_composite_score_weight_effect(brand_data):
    high_ctr = composite_score("Test", "samsung", 0.9, ctr_score=80, brand_score=50, brand_data=brand_data)
    high_brand = composite_score("Test", "samsung", 0.1, ctr_score=80, brand_score=50, brand_data=brand_data)
    assert high_ctr["composite_score"] > high_brand["composite_score"]


def test_rank_variants_order(brand_data):
    variants = [
        {"style": "emotional", "text": "Unfold what's next."},
        {"style": "informative", "text": "Buy now limited deal today!"},
    ]
    ranked = rank_variants(variants, "samsung", 0.6, brand_data)
    assert len(ranked) == 2
    assert ranked[0]["composite_score"] >= ranked[1]["composite_score"]
    for v in ranked:
        assert "style" in v
        assert "ctr_score" in v
        assert "brand_score" in v


def test_batch_normalization_scales(monkeypatch):
    # 10/10 spec: bileşik skor, kartta görünen mutlak skorların
    # (ctr_benchmark + brand_score) ağırlıklı ortalamasıdır; batch-içi
    # göreceli uçurumlar (10/95) oluşmaz.
    import src.composite as composite_mod

    def fake_predict(text):
        map_ = {"A": 2.0, "B": 18.0}
        return map_[text]
    def fake_brand(text, brand_id, brand_data=None):
        return {"A": 80.0, "B": 20.0}[text]
    def fake_load():
        return {}

    monkeypatch.setattr(composite_mod, "predict_ctr", fake_predict)
    monkeypatch.setattr(composite_mod, "brand_alignment_score", fake_brand)
    monkeypatch.setattr(composite_mod, "load_brand_data", fake_load)

    variants = [{"style": "a", "text": "A"}, {"style": "b", "text": "B"}]
    ranked = composite_mod.rank_variants(variants, "samsung", 0.6, brand_data={})
    by_text = {v["text"]: v for v in ranked}

    # CTR benchmark: ln(1+2)/ln(36)*100=30.7, ln(19)/ln(36)*100=82.2
    assert by_text["A"]["ctr_benchmark"] == 30.7
    assert by_text["B"]["ctr_benchmark"] == 82.2
    # brand_score ham değer kullanılır: A=80.0, B=20.0
    assert by_text["A"]["brand_score"] == 80.0
    assert by_text["B"]["brand_score"] == 20.0
    # Bileşik = w*benchmark + (1-w)*brand_score (ağırlıklı ortalama, kutupsal değil)
    w = 0.6
    assert abs(by_text["A"]["composite_score"] - round(w*30.7 + (1-w)*80.0, 1)) < 0.05
    assert abs(by_text["B"]["composite_score"] - round(w*82.2 + (1-w)*20.0, 1)) < 0.05
    # sıralama B önce gelir
    assert ranked[0]["text"] == "B"
    # ham değerler korunur
    assert by_text["A"]["ctr_score"] == 2.0 and by_text["B"]["ctr_score"] == 18.0


def test_batch_normalization_equal_values():
    # Tüm skorlar eşitse normalize değer midpoint olmalı, 0/NaN olmamalı.
    from src.composite import _soft_scale
    result = _soft_scale([5.0, 5.0, 5.0])
    assert len(result) == 3
    mid = result[0]
    assert all(abs(v - mid) < 1e-9 for v in result)
    assert mid > 0  # Not zero
    assert _soft_scale([]) == []


def test_ctr_benchmark_monotonic():
    # Görev 3: benchmark artan fonksiyon olmalı ve [10, 98] aralığında olmalı.
    from src.composite import ctr_benchmark
    low = ctr_benchmark(2.0)
    mid = ctr_benchmark(10.0)
    high = ctr_benchmark(30.0)
    assert low < mid < high
    assert 10.0 <= low <= 98.0
    assert 10.0 <= high <= 98.0


def test_score_batch_includes_benchmark():
    # Görev 3: score_batch her varyanta ctr_benchmark alanı eklemeli.
    from src.composite import score_batch
    variants = [{"style": "a", "text": "Unfold what's next."}]
    ranked = score_batch(variants, "samsung", 0.6, brand_data=None)
    assert "ctr_benchmark" in ranked[0]
    assert 10.0 <= ranked[0]["ctr_benchmark"] <= 98.0


def test_composite_matches_absolute_scores(monkeypatch):
    # DoD #1: Bileşik skor, kartta görünen İKİ mutlak skorun (ctr_benchmark +
    # brand_score) ağırlıklı ortalamasına BİREBİR eşit olmalı. Kullanıcının
    # seçtiği ağırlık (0.2 testi) sabit 0.6 varsayımını kırar.
    import src.composite as composite_mod

    def fake_predict(text):
        return 8.0
    def fake_brand(text, brand_id, brand_data=None):
        return {"X": 40.0, "Y": 90.0}[text]
    def fake_load():
        return {}

    monkeypatch.setattr(composite_mod, "predict_ctr", fake_predict)
    monkeypatch.setattr(composite_mod, "brand_alignment_score", fake_brand)
    monkeypatch.setattr(composite_mod, "load_brand_data", fake_load)

    w = 0.2
    variants = [{"style": "x", "text": "X"}, {"style": "y", "text": "Y"}]
    ranked = composite_mod.score_batch(variants, "samsung", w, brand_data={})
    for v in ranked:
        expected = round(w * v["ctr_benchmark"] + (1 - w) * round(v["brand_score"], 1), 1)
        assert abs(v["composite_score"] - expected) < 0.01, (
            f"composite {v['composite_score']} != weighted avg {expected}"
        )
