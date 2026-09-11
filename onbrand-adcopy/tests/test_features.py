import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np
from src.features import extract_features, add_features, FEATURE_COLS


class TestExtractFeatures:
    def test_empty_string(self):
        feats = extract_features("")
        assert feats["char_count"] == 0
        assert feats["word_count"] == 0
        assert feats["has_question"] == 0
        assert feats["has_exclamation"] == 0
        assert feats["urgency_count"] == 0
        assert feats["has_urgency"] == 0

    def test_question_and_exclamation(self):
        feats = extract_features("Really? Yes! Limited now.")
        assert feats["has_question"] == 1
        assert feats["has_exclamation"] == 1
        assert feats["urgency_count"] == 2  # "limited" and "now"

    def test_multiple_urgency_words(self):
        text = "Hurry! Today only - exclusive deal! Must have!"
        feats = extract_features(text)
        assert feats["urgency_count"] == 5  # hurry, today, exclusive, deal, must have
        assert feats["has_urgency"] == 1

    def test_none_input(self):
        feats = extract_features(None)
        assert feats["char_count"] == 0

    def test_word_count(self):
        feats = extract_features("one two three four five")
        assert feats["word_count"] == 5

    def test_caps_string(self):
        text = "THIS IS ALL CAPS!"
        feats = extract_features(text)
        assert feats["char_count"] == len(text)
        assert feats["has_exclamation"] == 1

    def test_turkish_urgency(self):
        feats = extract_features("Son şans! Bugün al, hem şimdi indirim, fırsat kaçırma!")
        assert feats["urgency_count"] == 6  # son şans, bugün, şimdi, indirim, fırsat, kaçırma
        assert feats["has_urgency"] == 1

    def test_emoji_and_hashtags(self):
        feats = extract_features("Amazing deal! 🔥 #Tech #Sale #NewRelease @Brand")
        assert feats["emoji_count"] == 1
        assert feats["hashtag_count"] == 3
        assert feats["has_hashtag"] == 1
        assert feats["has_mention"] == 1

    def test_cta_detection(self):
        assert extract_features("Grab yours today!")["has_cta"] == 1
        assert extract_features("Hemen al, bugün keşfet!")["has_cta"] == 1
        assert extract_features("Just a casual sentence.")["has_cta"] == 0

    def test_diversity_and_caps_ratio(self):
        feats = extract_features("BEST BEST BEST inventory")
        assert feats["word_diversity"] <= 0.5
        assert feats["all_caps_ratio"] > 0.5


class TestAddFeatures:
    def test_adds_all_columns(self):
        df = pd.DataFrame({"text_content": ["Hello world", "Test content"]})
        result = add_features(df)
        for col in ["char_count", "word_count", "has_question", "has_exclamation", "urgency_count", "has_urgency"]:
            assert col in result.columns

    def test_preserves_original_columns(self):
        df = pd.DataFrame({"text_content": ["Hi"], "extra_col": [1]})
        result = add_features(df)
        assert "extra_col" in result.columns
