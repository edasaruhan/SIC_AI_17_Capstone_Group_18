import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.cache import get_cached, set_cached, clear_cache, DB_PATH


@pytest.fixture(autouse=True)
def _clean():
    yield
    clear_cache()


def test_set_and_get():
    set_cached("m", "prompt-A", "response-A")
    assert get_cached("m", "prompt-A") == "response-A"
    assert get_cached("m", "prompt-B") is None


def test_same_key_same_model():
    set_cached("gpt-4", "q", "ans1")
    assert get_cached("gpt-4", "q") == "ans1"
    # farklı model farklı yanıt
    set_cached("gpt-3.5", "q", "ans2")
    assert get_cached("gpt-3.5", "q") == "ans2"
    assert get_cached("gpt-4", "q") == "ans1"


def test_db_exists():
    assert DB_PATH.exists()