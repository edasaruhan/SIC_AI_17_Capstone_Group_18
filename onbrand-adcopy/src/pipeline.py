import time
import logging
import numpy as np
from .generator import generate_variants
from .composite import rank_variants
from .autocorrect import auto_correct
from .brand_alignment import load_brand_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(
    product_description: str,
    target_audience: str,
    brand_id: str,
    weight_ctr: float = 0.6,
    alignment_threshold: float = 70,
) -> dict:
    brand_data = load_brand_data()
    timings = {}

    t0 = time.time()
    raw_variants = generate_variants(product_description, target_audience, brand_id)
    timings["generation"] = time.time() - t0

    variants = [v for v in raw_variants if v.get("text")]
    generation_errors = [v["error"] for v in raw_variants if v.get("error")]
    if not variants:
        detail = generation_errors[0] if generation_errors else "bilinmeyen hata"
        raise RuntimeError(
            "Reklam metni üretilemedi (tüm üslup denemeleri başarısız oldu). "
            f"Son hata: {detail}"
        )
    if generation_errors:
        logger.warning(f"{len(generation_errors)} üslup üretilemedi: {generation_errors}")

    t1 = time.time()
    ranked = rank_variants(variants, brand_id, weight_ctr, brand_data)
    timings["scoring"] = time.time() - t1

    ctr_scores = [v["ctr_score"] for v in ranked]
    ctr_median = float(np.median(ctr_scores)) if ctr_scores else 0

    t2 = time.time()
    # Build a mapping: corrected_text -> correction_result for proper matching
    correction_map = {}
    corrected_variants = []
    for v in ranked:
        result = auto_correct(v, brand_id, ctr_median, alignment_threshold, brand_data=brand_data)
        final_text = result["final_text"]
        correction_map[final_text] = result
        corrected_variants.append({
            "style": v["style"],
            "text": final_text,
            "needs_human_review": result["needs_human_review"],
        })
    timings["correction"] = time.time() - t2

    t3 = time.time()
    final_ranked = rank_variants(corrected_variants, brand_id, weight_ctr, brand_data)
    timings["final_scoring"] = time.time() - t3

    timings["total"] = sum(timings.values())

    # Match correction info by text content (not by index, since final_ranked is re-sorted)
    all_correction_results = []
    for fr in final_ranked:
        cr = correction_map.get(fr["text"])
        if cr:
            fr["correction_info"] = {
                "corrected": cr["corrected"],
                "attempts": cr["attempts"],
                "alignment_uplift": cr["alignment_uplift"],
                "ctr_delta": cr["ctr_delta"],
                "original_brand_score": cr["original_brand_score"],
                "original_text": cr["original_text"],
            }
            all_correction_results.append(cr)
        else:
            fr["correction_info"] = {
                "corrected": False, "attempts": 0,
                "alignment_uplift": 0.0, "ctr_delta": 0.0,
                "original_brand_score": fr.get("brand_score", 0.0),
                "original_text": fr["text"],
            }

    return {
        "ranked_variants": final_ranked,
        "correction_results": all_correction_results,
        "timings": timings,
        "brand_id": brand_id,
        "weight_ctr": weight_ctr,
        "generation_errors": generation_errors,
    }
