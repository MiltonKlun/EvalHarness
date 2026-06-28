"""Smoke test for the record/replay cache — Phase 0 exit criterion.

Verifies the three behaviours the two-tier CI strategy depends on:
  1. live mode records the computed response to disk,
  2. replay mode returns the recording without invoking compute(),
  3. replay mode hard-fails (CacheMiss) on an unknown key.

No API keys required — compute() is a stand-in for a real LLM call.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def cache_in_tmp(tmp_path, monkeypatch):
    """Point the cache at a temp dir and reload the modules that read CACHE_DIR."""
    from shared import config

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    cache = importlib.import_module("shared.cache")
    return config, cache


def test_record_then_replay(cache_in_tmp, monkeypatch):
    config, cache = cache_in_tmp
    pm, prompt, params = "test:model", "hello", {"temperature": 0}

    # Live mode: compute() runs and the result is recorded.
    monkeypatch.setattr(config, "LIVE_LLM", True)
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return "world"

    assert cache.cached_call(pm, prompt, params, compute) == "world"
    assert calls["n"] == 1

    # Replay mode: same key returns the recording WITHOUT calling compute().
    monkeypatch.setattr(config, "LIVE_LLM", False)

    def must_not_run():
        raise AssertionError("compute() must not run in replay mode on a hit")

    assert cache.cached_call(pm, prompt, params, must_not_run) == "world"
    assert calls["n"] == 1  # unchanged


def test_replay_miss_is_hard_failure(cache_in_tmp, monkeypatch):
    config, cache = cache_in_tmp
    monkeypatch.setattr(config, "LIVE_LLM", False)

    with pytest.raises(cache.CacheMiss):
        cache.cached_call("test:model", "never recorded", {}, lambda: "x")
