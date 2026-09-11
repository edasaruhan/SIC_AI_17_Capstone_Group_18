import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.generator import (
    classify_api_error,
    QuotaExceededError,
    _generate_with_retry,
    _is_transient,
)


def test_is_transient():
    assert _is_transient("429 Too Many Requests")
    assert _is_transient("503 SERVICE_UNAVAILABLE")
    assert _is_transient("rate limit exceeded")
    assert not _is_transient("invalid api key")


def test_classify_api_error_quota():
    msg = classify_api_error("429 RESOURCE_EXHAUSTED, daily limit reached")
    assert "kota" in msg


def test_classify_api_error_key():
    msg = classify_api_error("INVALID_ARGUMENT: API key not valid")
    assert "anahtar" in msg


def test_classify_api_error_service():
    msg = classify_api_error("503 Unavailable: server overloaded")
    assert "kullanılamıyor" in msg


def test_generate_with_retry_transient_then_success(monkeypatch):
    calls = {"n": 0}

    def fake_generate(client, prompt, model_name=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("503 Unavailable")
        return "ok text"

    sleeps = []
    monkeypatch.setattr("src.generator._generate", fake_generate)
    monkeypatch.setattr("src.generator.time.sleep", lambda s: sleeps.append(s))

    out = _generate_with_retry(None, "prompt", models=["test-model"], max_attempts=4)
    assert out == "ok text"
    assert calls["n"] == 3
    assert sleeps == [2, 4]  # üstel geri çekilme


def test_generate_with_retry_quota_raises_fast(monkeypatch):
    def fake_generate(client, prompt, model_name=None):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr("src.generator._generate", fake_generate)
    # With a single model, quota exhaustion should raise after trying
    with pytest.raises(RuntimeError):
        _generate_with_retry(None, "prompt", models=["test-model"], max_attempts=2)


def test_generate_with_retry_persistent_error(monkeypatch):
    def fake_generate(client, prompt, model_name=None):
        raise RuntimeError("INVALID_ARGUMENT: model bad")

    sleeps = []
    monkeypatch.setattr("src.generator._generate", fake_generate)
    monkeypatch.setattr("src.generator.time.sleep", lambda s: sleeps.append(s))

    with pytest.raises(RuntimeError):
        _generate_with_retry(None, "prompt", models=["test-model"], max_attempts=3)