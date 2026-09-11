"""Sentetik reklam metni + performans verisi üretir ve CTR modelini eğitir.

Felsefe: veri, metin -> etkileşim kuralının BİLİNÇLİ olarak tanımlandığı,
şablon tekrarı içermeyen, çift dilli (EN/TR) kampanya metinlerinden oluşur.
Model, çıkarımda kullanılan aynı özelliklerle (extract_features +
dinamik sentiment/toxicity) beslenir; kuralı öğrenebildiği için skorlar
gerçekten ayrışır. Veri sentezdir; rapor olarak "#sentez, bilinçli kural +
gürültü" diye dürüstçe açıklanır.

Kullanım:
    python scripts/build_model.py [--rows 3500] [--seed 42]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sentiment import analyze_sentiment, analyze_toxicity  # noqa: E402
from src.features import extract_features  # noqa: E402

# ---------------------------------------------------------------- text pools
EN_HOOKS_QU = [
    "Ready for a change?", "Why settle for less?", "What if your gear finally kept pace?",
    "Still on the old model?", "Seen enough?", "Tired of waiting for a bigger move?",
]
EN_OPENERS = [
    "Meet the {prod}.", "Discover the {prod}.", "The new {prod} is here.",
    "Say hello to the {prod}.", "Introducing the {prod}.",
    "Your daily routine, upgraded:", "Everything you need in one device:",
]
EN_BODY_POS = [
    "engineered for people who refuse to settle",
    "designed to keep pace with your busiest mornings",
    "seamless, intelligent, and effortless by design",
    "built to look great and move even smarter",
    "crafted for the ones who show up early",
    "smooth, responsive, and ready the moment you are",
    "powerful enough to feel effortless",
]
EN_BODY_NEU = [
    "a fresh take on the everyday essential",
    "ready out of the box and light on your desk",
    "fits right into your routine",
    "quietly handles all the basics",
    "made for real-world use",
    "practical, dependable, and easy to live with",
]
EN_BODY_NEG = [
    "far from what we expected",
    "feels unfinished in daily use",
    "a promise that arrives late",
    "underwhelming for the price",
]
EN_URGENCY = [
    "limited stock at this price", "today only", "a 48-hour window",
    "while supplies last", "prices rise at midnight", "exclusive for our early crew",
    "last chance at launch pricing", "the drop ends tonight", "don't miss the cut",
]
EN_CTA = [
    "Buy now", "Shop today", "Grab yours", "Order in the next 24 hours",
    "Start today", "Secure yours before it is gone",
]
EN_CAPS = ["BEST", "NOW", "LIMITED", "LAST CHANCE", "ACT FAST"]
EN_FILLERS = [
    "engineered for people who refuse to settle",
    "seamless and intelligent by design",
    "built for real-world mornings", "one device, endless possibilities",
]
EN_HASH = ["#Tech", "#NewRelease", "#Limited", "#DealAlert", "#MustHave",
           "#LaunchDay", "#SmarterLiving", "#NextGen", "#Style", "#Sale"]
EN_PRICES = ["for $199", "at $299", "under $150", "starting at $249"]
EN_PRODUCTS = ["Galaxy S24 Ultra", "Aurora Pro X", "Nimbus 7 Runner", "VoltBuds 2",
               "Skyline S13", "Takt Mesh Chair", "Luma Glass 4", "EchoFit Band",
               "Cascade Air Pump", "TrailBlazer GT"]

TR_HOOKS_QU = [
    "Değişime hazır mısın?", "Neden daha azıyla yetinesin?", "Ekipmanın sana yetişiyor mu?",
    "Hâlâ eski modelde misin?", "Yeterince gördün mü?", "Daha büyük bir hamle için beklemekten bıktın mı?",
]
TR_OPENERS = [
    "{prod} ile tanış.", "{prod} ürününü keşfet.", "Yeni {prod} burada.",
    "Merhaba de: {prod}.", "{prod} sunar.", "Her gününü kolaylaştıran tek cihaz:",
]
TR_BODY_POS = [
    "vazgeçmeyi reddeden insanlar için tasarlandı",
    "en yoğun sabahlarına yetişecek şekilde üretildi",
    "kusursuz, akıllı ve zahmetsiz bir deneyim",
    "harika görünmek ve daha akıllıca hareket etmek için tasarlandı",
    "her şeyi düşünülmüş bir tasarımla geldi",
]
TR_BODY_NEU = [
    "günlük işler için taze bir dokunuş",
    "kutusundan çıktığı gibi hazır",
    "rutinine anında uyum sağlar",
    "temel ihtiyaçları sessizce halleder",
    "gerçek hayat için üretildi",
]
TR_BODY_NEG = [
    "beklediğimizden çok uzak",
    "günlük kullanımda bitmemiş hissi veriyor",
    "vadesi geç bir vaat",
    "fiyatına göre içi boş",
]
TR_URGENCY = [
    "bu fiyata sınırlı stok", "sadece bugün", "48 saatlik fırsat", "stoklar tükeniyor",
    "gece yarısı fiyat artıyor", "ilk gelenlere özel", "sadece bugüne özel",
    "kampanya bu gece bitiyor", "kaçırma",
]
TR_CTA = [
    "Hemen al", "Hemen satın al", "Bugün sipariş ver", "Şimdi keşfet",
    "Fırsatı kaçırmayın", "Hemen başla", "Hemen incele",
]
TR_CAPS = ["ŞİMDİ", "SON ŞANS", "SINIRLI", "KAÇIRMA", "HEMEN"]
TR_FILLERS = [
    "vazgeçmeyi reddedenler için tasarlandı",
    "akıllı ve kusursuz bir tasarım",
    "gerçek sabahlar için üretildi", "tek cihaz, sınırsız olasılık",
]
TR_HASH = ["#Teknoloji", "#YeniÇıkan", "#Sınırlı", "#Fırsat", "#ŞartDeğilAma",
           "#LansmanGünü", "#AkıllıYaşam", "#YeniNesil", "#Stil", "#İndirim"]
TR_PRICES = ["sadece 4.999 TL", "3.999 TL ile başlıyor", "1.499 TL altı", "%40 indirimle"]
TR_PRODUCTS = ["Galaxy S24 Ultra", "Aurora Pro X", "Nimbus 7 Runner", "VoltBuds 2",
               "Skyline S13", "Takt Mesh Chair", "Luma Glass 4", "EchoFit Band",
               "Cascade Air Pump", "TrailBlazer GT"]

EMOJIS = ["🔥", "⚡", "🚀", "🏆", "💥", "🎯", "⭐"]

# ------------------------------------------------------------- known CTR rule
BETA0 = 1.6
RULES = {
    "urgency_count": 0.45,
    "sentiment_score": 0.50,
    "has_question": -0.40,
    "has_exclamation": 0.25,
    "has_cta": 0.45,
    "emoji_count": 0.30,
    "hashtag_count": 0.10,
    "has_mention": -0.20,
    "all_caps_ratio": 0.60,
    "word_diversity": 0.40,
    "numeric_ratio": 0.35,
    "toxicity_score": -2.00,
}
CHAR_OPTIMAL_LO, CHAR_OPTIMAL_HI = 60, 150
NOISE_STD = 0.30


def _length_penalty(char_count: int) -> float:
    if CHAR_OPTIMAL_LO <= char_count <= CHAR_OPTIMAL_HI:
        return 0.0
    rel_out = (max(0, CHAR_OPTIMAL_LO - char_count) / CHAR_OPTIMAL_LO
               + max(0, char_count - CHAR_OPTIMAL_HI) / CHAR_OPTIMAL_HI)
    return -0.55 * rel_out


def _pick(rng, pool):
    return pool[rng.integers(len(pool))]


def _coin(rng, p):
    return rng.random() < p


def _build_text(rng, lang: str, plan: dict) -> str:
    def P(pool):
        return pool[rng.integers(len(pool))]

    if lang == "tr":
        hooks_q, openers, body_pos, body_neu, body_neg = (
            TR_HOOKS_QU, TR_OPENERS, TR_BODY_POS, TR_BODY_NEU, TR_BODY_NEG)
        urgency, cta, caps, fillers = TR_URGENCY, TR_CTA, TR_CAPS, TR_FILLERS
        hashtags, prices, prod_pool = TR_HASH, TR_PRICES, TR_PRODUCTS
        mention_tag = "@TeknoGünlük"
    else:
        hooks_q, openers, body_pos, body_neu, body_neg = (
            EN_HOOKS_QU, EN_OPENERS, EN_BODY_POS, EN_BODY_NEU, EN_BODY_NEG)
        urgency, cta, caps, fillers = EN_URGENCY, EN_CTA, EN_CAPS, EN_FILLERS
        hashtags, prices, prod_pool = EN_HASH, EN_PRICES, EN_PRODUCTS
        mention_tag = "@GadgetDaily"

    prod = P(prod_pool)
    sent = plan["sentiment"]
    parts = []
    if plan["has_question"]:
        parts.append(P(hooks_q))
    if sent >= 0.3:
        opener = P(openers).format(prod=prod)
        body = P(body_pos)
    elif sent <= -0.2:
        opener = "Regarding the {prod}:".format(prod=prod)
        body = P(body_neg)
    else:
        opener = "Regarding the {prod}:".format(prod=prod)
        body = P(body_neu)
    parts.append(f"{opener} {body}.")
    parts.append(", ".join(P(urgency) for _ in range(plan["n_urg"])))
    if plan["caps"]:
        parts.append(P(caps))
    if plan["mention"]:
        parts.append(mention_tag)
    if plan["numeric"]:
        parts.append(P(prices))
    if plan["cta"]:
        parts.append(P(cta))
    if plan["emoji"]:
        parts.append(" ".join(P(EMOJIS) for _ in range(plan["emoji"])))
    if plan["hashtags"]:
        parts.append(" ".join(P(hashtags) for _ in range(plan["hashtags"])))

    text = " ".join(p for p in parts if p).strip()
    text = re.sub(r"\s+", " ", text)
    if plan["has_exclamation"]:
        text = re.sub(r"[!.]$", "!", text.strip()) or (text + "!")

    # uzunluk bandı: hedef stile ulaşana kadar doldurucu ekle
    lo, hi = {"short": (40, 80), "mid": (80, 140), "long": (160, 240)}[plan["length"]]
    guard = 0
    while len(text) < lo and guard < 6:
        text += " " + P(fillers)
        guard += 1
    if len(text) > hi:
        text = text[:hi].rstrip() + ("!" if plan["has_exclamation"] else "")
    return text


def _plan(rng, lang) -> dict:
    return {
        "sentiment": float(rng.choice([-0.5, -0.2, 0.3, 0.6, 0.9])),
        "has_question": _coin(rng, 0.28),
        "has_exclamation": _coin(rng, 0.45),
        "cta": _coin(rng, 0.5),
        "n_urg": int(rng.choice([0, 1, 2, 3], p=[0.35, 0.30, 0.20, 0.15])),
        "emoji": int(rng.choice([0, 0, 1, 2], p=[0.45, 0.20, 0.20, 0.15])),
        "hashtags": int(rng.choice([0, 1, 2], p=[0.50, 0.30, 0.20])),
        "caps": _coin(rng, 0.3),
        "numeric": _coin(rng, 0.3),
        "mention": _coin(rng, 0.15),
        "length": str(rng.choice(["short", "mid", "long"], p=[0.35, 0.45, 0.20])),
    }


def generate_dataset(n_english: int = 2000, n_turkish: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Kural + gürültü ile CTR hedefi üretir. Dönen CSV, CTR modelinin tüm
    CONTENT_FEATURE_COLS'unu dinamik olarak (çıkarımla aynı) hesaplar."""
    rng = np.random.default_rng(seed)
    rows = []
    for lang, count in (("en", n_english), ("tr", n_turkish)):
        for _ in range(count):
            plan = _plan(rng, lang)
            text = _build_text(rng, lang, plan)
            feats = extract_features(text)
            sent = float(analyze_sentiment(text))
            tox = float(analyze_toxicity(text))
            features = {**feats, "sentiment_score": sent, "toxicity_score": tox}

            y = BETA0
            for key, w in RULES.items():
                y += w * features[key]
            y += _length_penalty(feats["char_count"])
            y += float(rng.normal(0.0, NOISE_STD))

            ctr_proxy = float(np.clip(np.exp(y), 0.4, 60.0))
            impressions = int(np.round(np.exp(rng.normal(9.6, 0.8))))
            impressions = max(impressions, 100)
            engagements = ctr_proxy / 100.0 * impressions
            likes = int(np.round(0.55 * engagements))
            shares = int(np.round(0.25 * engagements))
            comments = int(np.round(0.20 * engagements))

            rows.append({
                "text_content": text,
                "language": lang,
                "impressions": impressions,
                "likes_count": likes,
                "shares_count": shares,
                "comments_count": comments,
                "sentiment_score": sent,
                "toxicity_score": tox,
                "buzz_change_rate": round(float(rng.normal(0, 20)), 2),
            })
    df = pd.DataFrame(rows)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=3500)   # toplam (en+tr)
    ap.add_argument("--split", type=float, default=0.571)  # en oranı
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    n_en = int(round(args.rows * args.split))
    n_tr = args.rows - n_en

    df = generate_dataset(n_english=n_en, n_turkish=n_tr, seed=args.seed)
    out = Path(__file__).resolve().parent.parent / "data" / "synthetic_ad_performance.csv"
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} rows ({n_en} EN, {n_tr} TR) -> {out}")
    print("Sample texts:")
    for _, r in df.sample(4, random_state=args.seed).iterrows():
        text = r['text_content'][:95].encode('ascii', 'replace').decode('ascii')
        print(f"  [{r['language']}] ctr~{((r['likes_count']+r['shares_count']+r['comments_count'])/r['impressions']*100):.1f}% | {text}")

    from src.data_prep import load_and_clean
    from src.features import add_features
    from src.ctr_model import train_ctr_model

    clean = load_and_clean(csv_path=out, languages=None)
    clean = add_features(clean)
    # Gösterim modu: ana modelin üzerine yazmaz, synthetic_demo_ ön ekiyle ayrı kaydeder.
    results = train_ctr_model(clean, output_prefix="synthetic_demo_")

    print("\n=== Content model (kural öğrenim metrikleri) ===")
    cm = results["content_metrics"]
    print(f"test R2={cm['r2']:.3f}, RMSE={cm['rmse']:.3f}, MAE={cm['mae']:.3f}, "
          f"5-fold CV R2={results['content_cv_r2_mean']:.3f}")
    print("Top content feature importances:")
    imp = dict(sorted(results["content_feature_importances"].items(), key=lambda kv: -kv[1]))
    for k, v in list(imp.items())[:8]:
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()