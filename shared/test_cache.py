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


def test_record_missing_replays_existing_hit(cache_in_tmp, monkeypatch):
    """RECORD_MISSING + LIVE: an already-recorded key replays; compute() is NOT called."""
    config, cache = cache_in_tmp
    pm, prompt, params = "test:model", "hello", {"temperature": 0}

    # Seed a recording via a normal live call.
    monkeypatch.setattr(config, "LIVE_LLM", True)
    monkeypatch.setattr(config, "RECORD_MISSING", False)
    assert cache.cached_call(pm, prompt, params, lambda: "world") == "world"

    # Now a RECORD_MISSING live run must replay the hit without recomputing.
    monkeypatch.setattr(config, "RECORD_MISSING", True)

    def must_not_run():
        raise AssertionError("compute() must not run on a hit under RECORD_MISSING")

    assert cache.cached_call(pm, prompt, params, must_not_run) == "world"


def test_record_missing_records_a_genuine_miss(cache_in_tmp, monkeypatch):
    """RECORD_MISSING + LIVE: an unrecorded key IS computed and recorded."""
    config, cache = cache_in_tmp
    monkeypatch.setattr(config, "LIVE_LLM", True)
    monkeypatch.setattr(config, "RECORD_MISSING", True)
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return "fresh"

    assert cache.cached_call("test:model", "brand new", {}, compute) == "fresh"
    assert calls["n"] == 1
    # And it is now on disk: a subsequent replay-mode call returns it with no compute.
    monkeypatch.setattr(config, "LIVE_LLM", False)
    monkeypatch.setattr(config, "RECORD_MISSING", False)
    assert cache.cached_call("test:model", "brand new", {}, lambda: "x") == "fresh"


def test_record_missing_is_noop_in_replay_mode(cache_in_tmp, monkeypatch):
    """RECORD_MISSING has no effect when LIVE_LLM is falsy: a miss still hard-fails."""
    config, cache = cache_in_tmp
    monkeypatch.setattr(config, "LIVE_LLM", False)
    monkeypatch.setattr(config, "RECORD_MISSING", True)

    with pytest.raises(cache.CacheMiss):
        cache.cached_call("test:model", "never recorded", {}, lambda: "x")
