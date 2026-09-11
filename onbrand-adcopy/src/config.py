import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from a .env file into the environment (no override)."""
    target = path or ENV_PATH
    if not target.exists():
        return
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_google_api_key() -> str:
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY bulunamadı. Lütfen proje kökündeki '.env' dosyasına "
            "GOOGLE_API_KEY=... satırını ekleyin (bkz. .env.example)."
        )
    return api_key


def get_model_name() -> str:
    """Gemini model adı. Varsayılan gemini-2.5-flash; GEMINI_MODEL env ile ezilebilir."""
    load_dotenv()
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"


def get_fallback_models() -> list[str]:
    """Primary model + fallback chain. Free tier = 20 req/model/day.
    Using multiple models effectively multiplies daily quota."""
    primary = get_model_name()
    all_models = ["gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash"]
    fallbacks = [m for m in all_models if m != primary]
    return [primary] + fallbacks