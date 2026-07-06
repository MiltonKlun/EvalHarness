"""Tests for the ClaudeJudge cache/JUDGE_LIVE routing — no network, no keys.

Verifies the D1 fix: with config.JUDGE_LIVE the judge calls fresh (no cache read, no cache
write); otherwise it routes through the record/replay cache. We patch the Anthropic client
so no real call happens, and watch cache.cached_call to see which path was taken.
"""

from __future__ import annotations

import types

import pytest

from evals import judge
from shared import cache, config


class _FakeMessages:
    def __init__(self, text: str):
        self._text = text

    def create(self, **kwargs):
        block = types.SimpleNamespace(text=self._text)
        return types.SimpleNamespace(content=[block])


class _FakeAnthropic:
    """Stands in for anthropic.Anthropic so _compute makes no real call."""

    last_call_count = 0

    def __init__(self, *a, **k):
        type(self).last_call_count += 1
        self.messages = _FakeMessages("safe")


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Patch anthropic.Anthropic (real module) + a dummy key so _compute runs offline.

    The raw path (schema=None) doesn't touch instructor, so patching the class on the real
    module is enough and avoids faking module internals.
    """
    import anthropic

    _FakeAnthropic.last_call_count = 0
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return _FakeAnthropic


def test_judge_live_bypasses_cache_entirely(monkeypatch, fake_anthropic):
    """JUDGE_LIVE=True: compute runs, cache.cached_call is NOT invoked (no read, no write)."""
    monkeypatch.setattr(config, "JUDGE_LIVE", True)

    called = {"cache": 0}

    def _boom(*a, **k):
        called["cache"] += 1
        raise AssertionError("cache.cached_call must not run under JUDGE_LIVE")

    monkeypatch.setattr(judge.cache, "cached_call", _boom)

    out = judge.ClaudeJudge().generate("grade this")  # raw path (schema=None)
    assert out == "safe"
    assert called["cache"] == 0
    assert fake_anthropic.last_call_count == 1  # a fresh client was built + called


def test_default_routes_through_cache(monkeypatch, fake_anthropic):
    """JUDGE_LIVE=False: the call goes through cache.cached_call (record/replay path)."""
    monkeypatch.setattr(config, "JUDGE_LIVE", False)

    seen = {"pm": None}

    def _fake_cached(provider_model, prompt, params, compute):
        seen["pm"] = provider_model
        return "cached-verdict"  # replay hit; compute not run

    monkeypatch.setattr(judge.cache, "cached_call", _fake_cached)

    out = judge.ClaudeJudge().generate("grade this")
    assert out == "cached-verdict"
    assert seen["pm"] == config.JUDGE_MODEL
    assert fake_anthropic.last_call_count == 0  # replay hit -> no live client built


def test_judge_live_writes_nothing_to_cache_dir(monkeypatch, tmp_path, fake_anthropic):
    """JUDGE_LIVE=True must not create any recording on disk."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "JUDGE_LIVE", True)

    judge.ClaudeJudge().generate("grade this")

    # No cache dir created, or if created, empty — nothing was recorded.
    cache_dir = tmp_path / "cache"
    assert not cache_dir.exists() or not any(cache_dir.iterdir())


def test_cache_miss_still_propagates_when_not_live(monkeypatch, fake_anthropic):
    """Sanity: with JUDGE_LIVE off and a real (temp) empty cache, replay misses raise."""
    monkeypatch.setattr(config, "JUDGE_LIVE", False)
    monkeypatch.setattr(config, "LIVE_LLM", False)
    # Point cache at an empty dir so any key is a miss.
    import tempfile
    from pathlib import Path

    monkeypatch.setattr(config, "CACHE_DIR", Path(tempfile.mkdtemp()))
    with pytest.raises(cache.CacheMiss):
        judge.ClaudeJudge().generate("never recorded prompt")
