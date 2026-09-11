import json
import logging
import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(os.getenv("SBERT_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"))
    return _model


def load_brand_data() -> dict:
    path = DATA_DIR / "brand_reference.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {b["brand_id"]: b for b in data["brands"]}


# Off-brand sim baseline: mean max-cosine-similarity between one brand's
# reference copies and ANOTHER brand's reference copies (~0.26-0.28 measured
# on the training references). Used to calibrate scores to an intuitive 0-100
# scale where >=70 means "strongly on-brand".
MAX_SIM_OFFBRAND_BASELINE = 0.30

# P1.1: kalibrasyon tabanına denk düşen skorlar tam 0.0 görünmesin diye
# pozitif bir taban uygulanır. "Uzak marka" metinleri 0.0 yerine çok küçük
# pozitif değerle gösterilir (0.1).
SCORE_FLOOR = 0.1


def brand_alignment_score(text: str, brand_id: str, brand_data: dict | None = None) -> float:
    """Semantic closeness of `text` to a brand's reference ad copies, 0-100.

    Uses the MAX cosine similarity to any single reference copy. Max
    (not mean) is used so a genuine brand slogan scores near 100 (it matches a
    reference copy almost exactly) while semantically off-brand text lands near
    0. The value is linearly calibrated from the measured off-brand baseline
    (~0.30 max-sim between brands) up to an exact match (1.0).
    Asla tam 0.0 dönmez (SCORE_FLOOR).
    """
    if brand_data is None:
        brand_data = load_brand_data()

    brand = brand_data[brand_id]
    refs = brand["reference_ad_copies"]

    model = _get_model()
    text_emb = model.encode([text])
    ref_embs = model.encode(refs)

    sims = cosine_similarity(text_emb, ref_embs)[0]
    best = float(np.max(sims))

    calibrated = (best - MAX_SIM_OFFBRAND_BASELINE) / (1.0 - MAX_SIM_OFFBRAND_BASELINE) * 100
    clipped = float(np.clip(calibrated, 0, 100))
    return clipped if clipped >= SCORE_FLOOR else SCORE_FLOOR


def rule_based_check(text: str, brand_id: str, brand_data: dict | None = None) -> dict:
    if brand_data is None:
        brand_data = load_brand_data()

    brand = brand_data[brand_id]
    text_lower = text.lower()
    violations = []

    preferred = brand.get("preferred_vocabulary", [])
    if preferred and not any(w in text_lower for w in preferred):
        violations.append(f"Missing preferred vocabulary: {', '.join(preferred)}")

    forbidden = brand.get("forbidden_vocabulary", [])
    hits = [w for w in forbidden if w in text_lower]
    if hits:
        violations.append(f"Contains forbidden language: {', '.join(hits)}")

    return {
        "violations": violations,
        "passed": len(violations) == 0,
        "rule_score_penalty": len(violations) * 5,
    }


def validate_alignment(brand_data: dict | None = None) -> dict:
    if brand_data is None:
        brand_data = load_brand_data()

    results = {}
    for bid, bdata in brand_data.items():
        refs = bdata["reference_ad_copies"]
        own_scores = [brand_alignment_score(ref, bid, brand_data) for ref in refs[:3]]
        other_brands = [other_bid for other_bid in brand_data if other_bid != bid]
        cross_scores = []
        for other_bid in other_brands:
            for ref in refs[:2]:
                cross_scores.append(brand_alignment_score(ref, other_bid, brand_data))

        results[bid] = {
            "own_mean": float(np.mean(own_scores)),
            "cross_mean": float(np.mean(cross_scores)),
            "own_higher": float(np.mean(own_scores)) > float(np.mean(cross_scores)),
        }
        logger.info(
            f"{bid}: own-brand mean={results[bid]['own_mean']:.2f}, "
            f"cross-brand mean={results[bid]['cross_mean']:.2f}, "
            f"own higher: {results[bid]['own_higher']}"
        )

    return results
