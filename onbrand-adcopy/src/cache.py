"""P3.1 — LLM çağrı önbelleği (SQLite).

Aynı (model, prompt) ikilisi için SQLite'ta text önbellekleme; HTTP çağrısını
atlar. cache/generations.db içinde tutulur, çağrı başına bir satır.
Multi-thread güvenli (Streamlit çoklu iş parçacığı uyumlu).
"""
import hashlib
import sqlite3
import threading
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = CACHE_DIR / "generations.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
    cache_key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    prompt TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
"""

_init_lock = threading.Lock()
_initialized = False


def _ensure_db():
    global _initialized
    if not _initialized:
        with _init_lock:
            if not _initialized:
                with sqlite3.connect(str(DB_PATH), timeout=15.0) as conn:
                    conn.execute(_SCHEMA)
                    conn.commit()
                _initialized = True


def _connect() -> sqlite3.Connection:
    _ensure_db()
    return sqlite3.connect(str(DB_PATH), timeout=15.0, check_same_thread=False)


def cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\n{prompt}".encode("utf-8")).hexdigest()


def get_cached(model: str, prompt: str) -> str | None:
    try:
        with _connect() as conn:
            cur = conn.execute(
                "SELECT text FROM generations WHERE cache_key = ?", (cache_key(model, prompt),)
            )
            row = cur.fetchone()
            return row[0] if row else None
    except Exception:
        return None


def set_cached(model: str, prompt: str, text: str) -> None:
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO generations (cache_key, model, prompt, text) VALUES (?, ?, ?, ?)",
                (cache_key(model, prompt), model, prompt, text),
            )
            conn.commit()
    except Exception:
        pass


def clear_cache() -> int:
    try:
        with _connect() as conn:
            cur = conn.execute("DELETE FROM generations")
            conn.commit()
            return cur.rowcount or 0
    except Exception:
        return 0