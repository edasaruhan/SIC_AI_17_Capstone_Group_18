import logging
import numpy as np
from .ctr_model import predict_ctr
from .brand_alignment import brand_alignment_score, load_brand_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Görev 3: CTR Benchmark kalibrasyon sabitleri
# Ham CTR (~5-25%) daha tanıdık bir 0-100 puanına dönüştürülür:
# clip(ln(1+raw) / ln(1+ref) * 100, floor, ceil)
_CTR_BENCH_REF = 35.0  # referans üst sınır
_CTR_BENCH_FLOOR = 10.0
_CTR_BENCH_CEIL = 98.0


def ctr_benchmark(raw_ctr: float) -> float:
    """Ham CTR skorunu 0-100 kalibre edilmiş benchmark puanına çevirir."""
    score = np.log1p(max(0.0, raw_ctr)) / np.log1p(_CTR_BENCH_REF) * 100
    return float(np.clip(score, _CTR_BENCH_FLOOR, _CTR_BENCH_CEIL))


def composite_score(
    text: str,
    brand_id: str,
    weight_ctr: float = 0.6,
    ctr_score: float | None = None,
    brand_score: float | None = None,
    brand_data: dict | None = None,
) -> dict:
    if ctr_score is None:
        ctr_score = predict_ctr(text)
    if brand_score is None:
        brand_score = brand_alignment_score(text, brand_id, brand_data)

    # 10/10 spec: CTR bileşeni kalibre edilmiş benchmark olarak kullanılır,
    # böylece karttaki CTR Skoru ile bileşik skor aynı 0-100 ölçekte tutarlıdır.
    ctr_norm = ctr_benchmark(ctr_score)
    brand_norm = max(0, min(100, brand_score))

    composite = weight_ctr * ctr_norm + (1 - weight_ctr) * brand_norm

    return {
        "text": text,
        "ctr_score": float(ctr_score),
        "brand_score": float(brand_score),
        "ctr_normalized": float(ctr_norm),
        "brand_normalized": float(brand_norm),
        "composite_score": float(composite),
    }


def _soft_scale(values: list[float], floor: float = 10.0, ceil: float = 95.0) -> list[float]:
    """Batch-aware soft normalization to [floor, ceil] range.

    Avoids misleading 0/100 extremes when batch is small (e.g. 2 variants).
    Equal values map to midpoint. With >=4 variants uses wider range.
    """
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mx - mn < 1e-9:
        mid = (floor + ceil) / 2
        return [mid] * len(values)
    # With 4+ variants, allow fuller range; with 2 use tighter range
    if len(values) <= 2:
        floor, ceil = 20.0, 85.0
    elif len(values) <= 3:
        floor, ceil = 15.0, 90.0
    return [floor + (v - mn) / (mx - mn) * (ceil - floor) for v in values]


def score_batch(
    variants: list[dict],
    brand_id: str,
    weight_ctr: float = 0.6,
    brand_data: dict | None = None,
) -> list[dict]:
    """Score a batch of variants with consistent, absolute 0-100 scores.

    10/10 spec: karttaki CTR hücresi `ctr_benchmark`'ı, marka hücresi
    `brand_score`'u gösterir; bileşik skor bu İKİ mutlak skorun ağırlıklı
    ortalamasıdır: composite = w*benchmark + (1-w)*brand_score. Batch-içi
    göreceli uçurumlar (10/95) oluşmaz; sayılar hesap makinesiyle tutarlıdır.
    """
    if brand_data is None:
        brand_data = load_brand_data()

    scored = [composite_score(v["text"], brand_id, weight_ctr, brand_data=brand_data)
              for v in variants]

    for v, s in zip(variants, scored):
        bench = round(ctr_benchmark(s["ctr_score"]), 1)
        brand_val = round(float(s["brand_score"]), 1)
        # Bileşik = kartta görünen iki skorun ağırlıklı ortalaması
        composite = weight_ctr * bench + (1 - weight_ctr) * brand_val
        s["style"] = v.get("style", "unknown")
        s["needs_human_review"] = v.get("needs_human_review", False)
        s["ctr_benchmark"] = bench
        s["ctr_scaled"] = bench
        s["brand_scaled"] = brand_val
        s["composite_score"] = round(float(composite), 1)

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    return scored


def rank_variants(
    variants: list[dict],
    brand_id: str,
    weight_ctr: float = 0.6,
    brand_data: dict | None = None,
) -> list[dict]:
    return score_batch(variants, brand_id, weight_ctr, brand_data)
