import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from .brand_alignment import load_brand_data
from .config import get_google_api_key, get_model_name, get_fallback_models
from .cache import get_cached, set_cached

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class QuotaExceededError(Exception):
    """API günlük kotası doldu / anahtar geçersiz; yeniden denemenin faydası yok."""


def classify_api_error(message: str) -> str:
    """API hatasını Türkçe, kullanıcının anlayacağı şekilde sınıflandırır."""
    msg = message.lower()
    if "resource_exhausted" in msg or "429" in msg:
        return "API günlük kotası doldu. Bi'r saat sonra veya yeni anahtar ile tekrar deneyin."
    if "unauth" in msg or "api key" in msg or "invalid" in msg:
        return "API anahtarı geçersiz. .env dosyasındaki GOOGLE_API_KEY değerini kontrol edin."
    if "503" in msg or "unavailable" in msg:
        return "Çığır açıcı servis geçici olarak kullanılamıyor; birkaç saniye sonra yeniden deneyin."
    if "not found" in msg or "404" in msg:
        return "İstenen model bulunamadı. .env içindeki GEMINI_MODEL değerini kontrol edin."
    return message


def _is_transient(message: str) -> bool:
    msg = message.lower()
    return ("429" in msg or "503" in msg or "rate" in msg
            or "unavailable" in msg or msg.startswith("5"))


def _generate_with_retry(client: genai.Client, prompt: str, models: list[str] | None = None, max_attempts: int = 3) -> str:
    """Üstel geri çekilme + model fallback: kotaya basıldıysa sonraki modeli dene,
    geçici hatalar (429/503/5xx) için 2,4,8 sn bekle ve tekrar dene."""
    if models is None:
        models = [get_model_name()]

    last_error = None
    for model_name in models:
        for attempt in range(max_attempts):
            try:
                return _generate(client, prompt, model_name)
            except Exception as e:
                last_error = e
                msg = str(e)
                if "resource_exhausted" in msg.lower() or "quota" in msg.lower():
                    logger.warning(f"Kota doldu ({model_name}), sonraki modele geçiliyor...")
                    break  # Try next model
                if _is_transient(msg):
                    wait = 2 ** attempt * 2  # 2, 4, 8 sn
                    logger.warning(f"Geçici hata, {wait}s sonra denenecek ({attempt + 1}/{max_attempts}): {msg}")
                    time.sleep(wait)
                else:
                    logger.error(f"Kalıcı hata ({attempt + 1}/{max_attempts}): {msg}")
        else:
            continue  # All attempts for this model failed with transient errors, try next
        continue  # Quota exhausted for this model, try next
    raise last_error

STYLE_PROMPTS = {
    "emotional": "Write an emotional, heartfelt ad copy that connects with the reader on a personal level.",
    "informative": "Write an informative, fact-driven ad copy that highlights key product benefits clearly.",
    "urgency": "Write an urgent, action-driven ad copy that motivates immediate response.",
    "humorous": "Write a funny, witty ad copy that entertains while promoting the product.",
}

STYLE_POSITIVE_HINTS = {
    "emotional": "of the brand's emotionally resonant lines",
    "informative": "of the brand's benefit-driven, feature-forward lines",
    "urgency": "of the brand's short, punchy, immediate lines",
    "humorous": "of the brand's playful, witty lines",
}

SYSTEM_PROMPT = """You are an expert advertising copywriter. You write concise, compelling ad copy.
Rules:
- No fabricated price or discount claims.
- No false scarcity or misleading claims.
- No negative mentions of competitors.
- Keep each variant under 30 words.
- Never start with a generic teaser like 'Discover', 'Unleash', 'Elevate' or 'Say hello to'.
- Return ONLY the ad copy text, nothing else."""

NEGATIVE_EXAMPLES = [
    "Discover the amazing future of technology with our incredible new product!",
    "Unleash your potential with this must-have gadget, buy now!",
]


def _get_client() -> genai.Client:
    return genai.Client(api_key=get_google_api_key())


def _generate(client: genai.Client, prompt: str, model_name: str | None = None) -> str:
    if model_name is None:
        model_name = get_model_name()
    cached = get_cached(model_name, prompt)
    if cached is not None:
        return cached
    response = client.models.generate_content(model=model_name, contents=prompt)
    text = response.text.strip()
    set_cached(model_name, prompt, text)
    return text


def _build_prompt(
    brand: dict,
    style: str,
    product_description: str,
    target_audience: str,
    refs_text: str,
    style_rules_text: str,
    negative_text: str,
) -> str:
    return f"""{SYSTEM_PROMPT}

Brand: {brand['brand_name']}
Tone: {brand['tone_category']}
Brand voice: {brand['brand_voice_summary']}

Style rules:
{style_rules_text}

Reference ad copies from this brand (tone examples, pick from these):
{refs_text}

Examples of ad copy to NEVER write (generic, off-brand):
{negative_text}

Product: {product_description}
Target audience: {target_audience}

Task: {STYLE_PROMPTS[style]}

Write exactly ONE ad copy variant in the {style} style ({STYLE_POSITIVE_HINTS[style]}). Stay true to the brand voice, sound like a real human, and avoid the never-write examples. Return ONLY the ad copy text."""


def _generate_single(
    client: genai.Client,
    style: str,
    prompt: str,
    fallback_models: list[str],
) -> dict:
    """Tek bir üslup için LLM çağrısı yapar (paralel çalıştırılır)."""
    try:
        text = _generate_with_retry(client, prompt, models=fallback_models, max_attempts=2)
        logger.info(f"Generated {style} variant: {text[:60]}...")
        return {"style": style, "text": text}
    except QuotaExceededError as e:
        logger.error(f"Quota exceeded for {style}: {e}")
        return {"style": style, "text": None, "error": f"kota: {e}"}
    except Exception as e:
        logger.error(f"Error generating {style} variant: {e}")
        return {"style": style, "text": None, "error": str(e)}


def generate_variants(
    product_description: str,
    target_audience: str,
    brand_id: str,
    styles: list[str] | None = None,
    parallel: bool = True,
) -> list[dict]:
    if styles is None:
        styles = ["emotional", "informative", "urgency", "humorous"]

    brand_data = load_brand_data()
    brand = brand_data[brand_id]
    refs = brand["reference_ad_copies"][:5]
    refs_text = "\n".join(f"- {r}" for r in refs)
    style_rules_text = "\n".join(f"- {r}" for r in brand["style_rules"])
    negative_text = "\n".join(f"- {ex}" for ex in NEGATIVE_EXAMPLES)

    client = _get_client()
    fallback_models = get_fallback_models()

    # Her stil için prompt'u hazırla
    prompts = {}
    for style in styles:
        prompts[style] = _build_prompt(
            brand, style, product_description, target_audience,
            refs_text, style_rules_text, negative_text,
        )

    if parallel and len(styles) > 1:
        # Görev 1: Paralel LLM üretimi (ThreadPoolExecutor)
        results = [None] * len(styles)
        with ThreadPoolExecutor(max_workers=min(len(styles), 4)) as pool:
            future_map = {
                pool.submit(_generate_single, client, s, prompts[s], fallback_models): i
                for i, s in enumerate(styles)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                results[idx] = future.result()
    else:
        results = [_generate_single(client, s, prompts[s], fallback_models) for s in styles]

    return results