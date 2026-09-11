import logging
import re

logger = logging.getLogger(__name__)

_POSITIVE_WORDS = {
    "amazing", "awesome", "best", "better", "brilliant", "excellent", "fantastic",
    "free", "great", "incredible", "love", "loved", "perfect", "win", "winning",
    "wins", "unlimited", "epic", "dream", "crazy", "excited", "happy", "beautiful",
    "smart", "intelligent", "seamless", "powerful", "battery", "future", "stunning",
    "helpful", "effective", "fun", "sleek", "premium", "value", "upgrade", "flawless",
}
_NEGATIVE_WORDS = {
    "bad", "worst", "worse", "terrible", "awful", "hate", "horrible", "poor",
    "disappoint", "disappointed", "broken", "slow", "lag", "fragile", "outdated",
    "waste", "fail", "failed", "never", "cannot", "not", "no", "nothing",
    "annoying", "useless", "expensive", "overpriced", "regret", "quit",
}
_NEGATION_WORDS = {"not", "never", "no", "hardly", "barely"}

# Türkçe sözlükler ve vurgulama
_TR_POSITIVE = {
    "harika", "muhteşem", "süper", "mükemmel", "efsane", "benzersiz", "kusursuz",
    "ücretsiz", "fırsat", "indirim", "akıllı", "hızlı", "pratik", "yenilikçi",
    "seviyorum", "bayıldım", "sevdim", "güzel", "iyi", "sınırları aş", "şaşırtıcı",
    "kolay", "taşı", "öndeyim", "lider", "kazan", "coşkulu", "hesapsız",
}
_TR_NEGATIVE = {
    "berbat", "kötü", "rezalet", "kırık", "yavaş", "beğenmedim", "pahalı",
    "iğrenç", "bozuk", "zaman kaybı", "pişman", "nefret", "vasat", "hayal kırıklığı",
    "çalışmıyor", "tükenmiş", "alma", "kandırmaca", "saçma", "geç", "karmaşık",
}
_TR_NEGATIONS = {"değil", "yok", "yoksa", "hiç"}
_TR_TOXIC = ["aptal", "salak", "dolandırıcı", "saçmalık", "rezalet", "mal", "gerizekalı"]

_TOXIC_TERMS = [
    "stupid", "idiot", "dumb", "moron", "hate", "shut up", "loser", "trash",
    "garbage", "scam", "rip-off", "sucks", "pathetic", "delusional", "fraud",
]

_NEGATION_RE = re.compile(r"\b(?:not|never|no|hardly|barely)\b", re.IGNORECASE)
_EXCLAMATION_RE = re.compile(r"!+")

# Türkçe karakter varlığına göre metin dili tespiti
_TR_CHARS_RE = re.compile(r"[ğĞışİöÖüÜçÇ]")
_TR_FUNCTION_WORDS = {"ve", "bir", "değil", "ile", "için", "çok", "ama", "var",
                      "yok", "da", "de", "bile", "ya"}


def _turkish_compound(words: list[str]) -> float:
    """Türkçe sözlük + SONRADAN gelen olumsuzlama ("iyi değil") desteği."""
    contribs: list[float] = []
    for i, w in enumerate(words):
        if w in _TR_NEGATIONS:
            if contribs and i > 0 and words[i - 1] in (_TR_POSITIVE | _TR_NEGATIVE):
                contribs[-1] = -contribs[-1]  # "iyi değil" -> iyi'yi çevir
            continue
        if w in _TR_POSITIVE:
            contribs.append(1.0)
        elif w in _TR_NEGATIVE:
            contribs.append(-1.0)
    if not contribs:
        return 0.0
    return sum(contribs) / len(contribs)


def _is_turkish(text: str) -> bool:
    if _TR_CHARS_RE.search(text):
        return True
    words = set(re.findall(r"\w+", text.lower()))
    return bool(words & _TR_FUNCTION_WORDS)


def _vader_scores(text: str) -> float | None:
    """VADER ile [-1,1] skor; sözlük yoksa None döner."""
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except Exception:
        return None
    analyzer = getattr(_vader_scores, "_analyzer", None)
    if analyzer is None:
        try:
            analyzer = SentimentIntensityAnalyzer()
            _vader_scores._analyzer = analyzer
        except Exception:
            return None
    try:
        return float(analyzer.polarity_scores(text)["compound"])
    except Exception:
        return None


def _lexicon_compound(words: list[str], positive, negative, negations) -> float:
    """Basit hâlinde sözlük + olumsuzlama desteğiyle [-1,1] bileşik skor."""
    score = 0.0
    hit = 0
    negate = False
    for w in words:
        if w in negations:
            negate = True
            continue
        pol = 0
        if w in positive:
            pol = 1
        elif w in negative:
            pol = -1
        if pol != 0:
            if negate:
                pol = -pol
            score += pol
            hit += 1
            negate = False
    if hit:
        return score / hit
    return 0.0


def _turkish_words(text: str) -> list[str]:
    # Türkçe metinde alttan üstten ayraçlara (ğ, ş, ı, ö, ü, ç) izin veren tokenizer
    return re.findall(r"[A-Za-zğĞışİöÖüÜçÇ']+", text.lower())


def analyze_sentiment(text: str) -> float:
    """Metnin duygu skoru, [-1,1]. Türkçe metinlerde Türkçe sözlük,
    İngilizce metinlerde VADER; VADER yoksa yerleşik İngilizce sözlük."""
    if not text or not text.strip():
        return 0.0

    if _is_turkish(text):
        words = _turkish_words(text)
        compound = _turkish_compound(words)
        if _EXCLAMATION_RE.search(text):
            compound = min(1.0, compound + 0.1)
        return max(-1.0, min(1.0, compound))

    vader = _vader_scores(text)
    if vader is not None:
        return max(-1.0, min(1.0, vader))

    words = re.findall(r"[A-Za-z']+", text.lower())
    compound = _lexicon_compound(words, _POSITIVE_WORDS, _NEGATIVE_WORDS, _NEGATION_WORDS)
    if _EXCLAMATION_RE.search(text) and any(w in words for w in (_POSITIVE_WORDS | _NEGATIVE_WORDS)):
        compound = min(1.0, compound + 0.1)
    return max(-1.0, min(1.0, compound))


def analyze_toxicity(text: str) -> float:
    """Metnin toksiklik skoru, [0,1]. Toksik kelime yoğunluğu bazlı (EN + TR)."""
    if not text or not text.strip():
        return 0.0
    lowered = text.lower()
    if _is_turkish(text):
        terms = _TOXIC_TERMS + _TR_TOXIC
    else:
        terms = _TOXIC_TERMS
    hits = sum(1 for term in terms if term in lowered)
    if hits == 0:
        return 0.0
    words = len(text.split())
    words = max(words, 1)
    raw = hits / min(words, 10) * 3
    return float(max(0.0, min(1.0, raw)))