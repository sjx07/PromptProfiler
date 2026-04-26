"""Regression tests for run_experiment config plumbing."""
from __future__ import annotations

from run_experiment import _generator_kwargs, _llm_sampling_kwargs


def test_generator_kwargs_forwards_coalition_bounds():
    cfg = {
        "min_features": 4,
        "max_features": 4,
        "min_rules": 2,
        "max_rules": 5,
    }

    assert _generator_kwargs(cfg, n_samples=1, seed=42) == {
        "n_samples": 1,
        "seed": 42,
        "min_features": 4,
        "max_features": 4,
        "min_rules": 2,
        "max_rules": 5,
    }


def test_generator_kwargs_omits_unspecified_bounds():
    assert _generator_kwargs({}, n_samples=20, seed=7) == {
        "n_samples": 20,
        "seed": 7,
    }


def test_llm_sampling_kwargs_forwards_decoding_controls():
    cfg = {
        "temperature": 0.6,
        "top_p": 0.95,
        "sampling_top_k": 20,
        "top_k": 10,
    }

    assert _llm_sampling_kwargs(cfg) == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    }


def test_llm_sampling_kwargs_omits_unspecified_controls():
    assert _llm_sampling_kwargs({"top_k": 10}) == {}
