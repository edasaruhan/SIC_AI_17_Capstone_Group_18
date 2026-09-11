import logging
import time
from google import genai
import numpy as np
from .brand_alignment import brand_alignment_score, load_brand_data, rule_based_check
from .ctr_model import predict_ctr
from .config import get_google_api_key, get_model_name, get_fallback_models
from .cache import get_cached, set_cached

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REWRITE_SYSTEM = """You are an expert advertising copywriter specializing in brand voice alignment.
You rewrite ad copy to match a specific brand's voice and style rules.
Rules:
- No fabricated price or discount claims.
- No false scarcity or misleading claims.
- No negative mentions of competitors.
- Keep each variant under 30 words.
- Return ONLY the rewritten ad copy text, nothing else."""


def _get_client() -> genai.Client:
    return genai.Client(api_key=get_google_api_key())


def _rewrite(client: genai.Client, prompt: str) -> str:
    """Try primary model first, then fall back to alternative models on quota errors."""
    models = get_fallback_models()
    last_error = None
    for model_name in models:
        cached = get_cached(model_name, prompt)
        if cached is not None:
            return cached
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            text = response.text.strip()
            set_cached(model_name, prompt, text)
            return text
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            if "resource_exhausted" in msg or "quota" in msg or "429" in msg:
                logger.warning(f"Rewrite kota doldu ({model_name}), sonraki modele geçiliyor...")
                continue
            raise  # Non-quota error, re-raise immediately
    raise last_error


def auto_correct(
    variant: dict,
    brand_id: str,
    ctr_median: float,
    alignment_threshold: float = 70,
    max_attempts: int = 2,
    brand_data: dict | None = None,
) -> dict:
    if brand_data is None:
        brand_data = load_brand_data()

    text = variant["text"]
    ctr = variant.get("ctr_score", predict_ctr(text))
    brand_sc = variant.get("brand_score", brand_alignment_score(text, brand_id, brand_data))

    if brand_sc >= alignment_threshold or ctr <= ctr_median:
        return {
            "original_text": text,
            "final_text": text,
            "original_brand_score": brand_sc,
            "final_brand_score": brand_sc,
            "ctr_score": ctr,
            "corrected": False,
            "attempts": 0,
            "needs_human_review": False,
            "alignment_uplift": 0.0,
            "ctr_delta": 0.0,
        }

    brand = brand_data[brand_id]
    rules = brand.get("style_rules", [])
    rules_text = "\n".join(f"- {r}" for r in rules)
    refs = brand["reference_ad_copies"][:3]
    refs_text = "\n".join(f"- {r}" for r in refs)

    best_text = text
    best_brand_score = brand_sc
    attempts = 0

    try:
        client = _get_client()
    except EnvironmentError as e:
        logger.warning("GOOGLE_API_KEY bulunamadı, yeniden yazma yapılamadı")
        return {
            "original_text": text, "final_text": text,
            "original_brand_score": brand_sc, "final_brand_score": brand_sc,
            "ctr_score": ctr, "corrected": False, "attempts": 0,
            "needs_human_review": True,
            "alignment_uplift": 0.0, "ctr_delta": 0.0,
        }

    for i in range(max_attempts):
        attempts += 1
        rule_check = rule_based_check(best_text, brand_id, brand_data)
        violations = rule_check.get("violations", [])
        violation_note = f"Specific violations: {'; '.join(violations)}" if violations else "Semantic alignment is too low."

        prompt = f"""{REWRITE_SYSTEM}

Brand: {brand['brand_name']}
Tone: {brand['tone_category']}
Brand voice: {brand['brand_voice_summary']}

Style rules:
{rules_text}

Reference copies:
{refs_text}

Original ad copy:
{best_text}

Current brand alignment score: {best_brand_score:.1f}/100 (threshold: {alignment_threshold})
{violation_note}

Rewrite this ad copy to better match the brand voice. Keep the same message and approximate length.
Return ONLY the rewritten ad copy text."""

        try:
            rewritten = _rewrite(client, prompt)
            new_brand_sc = brand_alignment_score(rewritten, brand_id, brand_data)

            if new_brand_sc > best_brand_score:
                best_text = rewritten
                best_brand_score = new_brand_sc

            logger.info(
                f"Rewrite attempt {i+1}: alignment {brand_sc:.1f} -> {new_brand_sc:.1f}"
            )

            if best_brand_score >= alignment_threshold:
                break

        except Exception as e:
            logger.error(f"Rewrite error: {e}")
            break

    final_ctr = predict_ctr(best_text)
    uplift = best_brand_score - brand_sc
    ctr_delta = final_ctr - ctr

    return {
        "original_text": text,
        "final_text": best_text,
        "original_brand_score": brand_sc,
        "final_brand_score": best_brand_score,
        "ctr_score": final_ctr,
        "corrected": True,
        "attempts": attempts,
        "needs_human_review": best_brand_score < alignment_threshold,
        "alignment_uplift": float(uplift),
        "ctr_delta": float(ctr_delta),
    }
