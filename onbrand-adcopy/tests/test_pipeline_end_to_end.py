import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from src.brand_alignment import brand_alignment_score, load_brand_data
from src.composite import composite_score
from src.autocorrect import auto_correct


@pytest.fixture(scope="module")
def brand_data():
    return load_brand_data()


def test_autocorrect_skips_already_aligned(brand_data):
    score = brand_alignment_score("Unfold what's next.", "samsung", brand_data)
    result = auto_correct(
        {"text": "Unfold what's next.", "ctr_score": 50.0, "brand_score": score},
        "samsung",
        ctr_median=30.0,
        alignment_threshold=70,
        brand_data=brand_data,
    )
    assert result["corrected"] is False


def test_autocorrect_skips_low_ctr(brand_data):
    result = auto_correct(
        {"text": "Bad off brand text", "ctr_score": 5.0, "brand_score": 10.0},
        "samsung",
        ctr_median=50.0,
        alignment_threshold=70,
        brand_data=brand_data,
    )
    assert result["corrected"] is False


def test_pipeline_mocked():
    from src.pipeline import run_pipeline
    fake_variants = [
        {"style": "emotional", "text": "Great product for you!"},
        {"style": "informative", "text": "Discover amazing features today!"},
    ]
    with patch("src.pipeline.generate_variants", return_value=fake_variants):
        result = run_pipeline("Test product", "Test audience", "samsung", weight_ctr=0.6)
        assert "ranked_variants" in result
        assert "timings" in result
        assert len(result["ranked_variants"]) == 2
        for v in result["ranked_variants"]:
            assert "ctr_score" in v
            assert "brand_score" in v
            assert "composite_score" in v
