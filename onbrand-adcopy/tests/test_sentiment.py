import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.sentiment import analyze_sentiment, analyze_toxicity


def test_empty_texts_are_neutral():
    assert analyze_sentiment("") == 0.0
    assert analyze_sentiment(None) == 0.0
    assert analyze_toxicity("") == 0.0


def test_positive_text_scores_positive():
    score = analyze_sentiment("Amazing! Best deal ever! Absolutely love it!")
    assert score >= 0.3


def test_negative_text_scores_negative():
    score = analyze_sentiment("This product is terrible, broken and useless.")
    assert score <= -0.3


def test_sentiment_range():
    for text in ["Great!", "Worst product ever.", "It is okay I guess."]:
        assert -1.0 <= analyze_sentiment(text) <= 1.0


def test_clean_text_not_toxic():
    assert analyze_toxicity("A seamless ecosystem for the future.") == 0.0


def test_toxic_text_detected():
    assert analyze_toxicity("You stupid idiot, this is a total scam.") > 0.0


def test_negation_flips_sentiment():
    neg = analyze_sentiment("This is not a great experience.")
    assert neg < 0.05


def test_turkish_positive_text():
    assert analyze_sentiment("Harika! Muhteşem bir ürün, mükemmel, bayıldım!") >= 0.3


def test_turkish_negative_text():
    assert analyze_sentiment("Berbat ve kötü bir ürün, kesinlikle iyi değil.") <= -0.1


def test_turkish_toxicity():
    assert analyze_toxicity("Bu ürün aptalca, dolandırıcı ve saçmalık.") > 0.0


def test_turkish_negation():
    neg = analyze_sentiment("Bu ürün iyi değil.")
    assert neg < 0.1