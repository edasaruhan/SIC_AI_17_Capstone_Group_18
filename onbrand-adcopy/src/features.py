import re
import pandas as pd
import numpy as np

# İngilizce + Türkçe aciliyet sözlüğü (testler İngilizce kelimeleri değişmeden bekliyor)
URGENCY_WORDS = [
    # EN
    "now", "today", "limited", "hurry", "exclusive", "last chance",
    "deal", "promo", "discount", "must have",
    # TR
    "şimdi", "hemen", "bugün", "sınırlı", "fırsat", "indirim", "acele",
    "kaçırma", "kaçırmayın", "son şans", "son gün", "stoklar tükeniyor",
    "son fırsat",
]
URGENCY_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in URGENCY_WORDS) + r")\b",
    re.IGNORECASE,
)

# Harekete geçirici ifadeler (CTA) — EN + TR
CTA_PHRASES = [
    "buy now", "shop today", "grab yours", "order today", "get it now",
    "act fast", "join now", "sign up today", "start today", "try it now",
    "hemen al", "hemen satın al", "şimdi satın al", "hemen incele",
    "fırsatı kaçırmayın", "hemen sipariş ver", "sen de katıl", "hemen dene",
    "bugün keşfet",
]
CTA_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in CTA_PHRASES) + r")\b",
    re.IGNORECASE,
)

EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B50\u2764\uFE0F]",
    re.UNICODE,
)
HASHTAG_RE = re.compile(r"#\w+")
MENTION_RE = re.compile(r"@\w+")

# EMOJI_RE üstte *MOR* (men suit) gibi kod noktası aralıklarını da yakalar; sorun değil.


def extract_features(text: str) -> dict:
    """Metinden CTR modeline girecek tüm metin özelliklerini çıkarır."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    char_count = len(text)
    word_count = len(text.split())
    has_question = 1 if "?" in text else 0
    has_exclamation = 1 if "!" in text else 0
    urgency_count = len(URGENCY_PATTERN.findall(text))
    has_urgency = 1 if urgency_count > 0 else 0

    emojis = EMOJI_RE.findall(text)
    hashtags = HASHTAG_RE.findall(text)
    mentions = MENTION_RE.findall(text)
    has_cta = 1 if CTA_PATTERN.search(text) else 0

    words = text.split()
    total_words = max(len(words), 1)
    unique_words = len(set(w.lower() for w in words))
    cap_count = sum(1 for w in words if w and w[0].isupper())
    num_count = sum(1 for w in words if any(ch.isdigit() for ch in w))

    return {
        "char_count": char_count,
        "word_count": word_count,
        "has_question": has_question,
        "has_exclamation": has_exclamation,
        "urgency_count": urgency_count,
        "has_urgency": has_urgency,
        "emoji_count": len(emojis),
        "hashtag_count": len(hashtags),
        "has_hashtag": 1 if hashtags else 0,
        "has_mention": 1 if mentions else 0,
        "has_cta": has_cta,
        "avg_word_len": round(char_count / total_words, 2),
        "word_diversity": round(unique_words / total_words, 3),
        "all_caps_ratio": round(cap_count / total_words, 3),
        "numeric_ratio": round(num_count / total_words, 3),
    }


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = df["text_content"].apply(lambda x: pd.Series(extract_features(x)))
    result = pd.concat([df, feature_df], axis=1)
    return result


FEATURE_COLS = [
    "char_count", "word_count", "has_question", "has_exclamation",
    "urgency_count", "has_urgency", "sentiment_score", "toxicity_score",
    "emoji_count", "hashtag_count", "has_hashtag", "has_mention", "has_cta",
    "avg_word_len", "word_diversity", "all_caps_ratio", "numeric_ratio",
]