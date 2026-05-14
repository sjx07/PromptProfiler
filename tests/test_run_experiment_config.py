"""Regression tests for run_experiment config plumbing."""
from __future__ import annotations

from core.store import CubeStore, OnConflict
from experiment.planner import RunEntry
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


def test_scores_by_config_can_filter_dataset():
    store = CubeStore(":memory:")
    try:
        config_id = store.get_or_create_config([])
        store.upsert_queries(
            [
                {
                    "query_id": "wtq_1",
                    "dataset": "wtq",
                    "content": "wtq question",
                    "meta": {"split": "test"},
                },
                {
                    "query_id": "wtq_2",
                    "dataset": "wtq",
                    "content": "another wtq question",
                    "meta": {"split": "test"},
                },
                {
                    "query_id": "sqa_1",
                    "dataset": "sqa",
                    "content": "sqa question",
                    "meta": {"split": "test"},
                },
            ],
            on_conflict=OnConflict.SKIP,
        )
        wtq_exec = store.insert_execution(config_id, "wtq_1", "model")
        wtq_exec_2 = store.insert_execution(config_id, "wtq_2", "model")
        sqa_exec = store.insert_execution(config_id, "sqa_1", "model")
        store.upsert_evaluation(wtq_exec, "denotation_acc", 0.25)
        store.upsert_evaluation(wtq_exec_2, "denotation_acc", 0.75)
        store.upsert_evaluation(sqa_exec, "denotation_acc", 1.0)

        pooled = store.scores_by_config("model", "denotation_acc")
        wtq_only = store.scores_by_config("model", "denotation_acc", dataset="wtq")
        wtq_subset = store.scores_by_config(
            "model",
            "denotation_acc",
            dataset="wtq",
            query_ids=["wtq_1"],
        )

        assert pooled == [
            {
                "config_id": config_id,
                "n": 3,
                "avg_score": 2.0 / 3.0,
                "min_score": 0.25,
                "max_score": 1.0,
            }
        ]
        assert wtq_only == [
            {
                "config_id": config_id,
                "n": 2,
                "avg_score": 0.5,
                "min_score": 0.25,
                "max_score": 0.75,
            }
        ]
        assert wtq_subset == [
            {
                "config_id": config_id,
                "n": 1,
                "avg_score": 0.25,
                "min_score": 0.25,
                "max_score": 0.25,
            }
        ]
    finally:
        store.close()


def test_run_and_eval_plan_passes_phase_to_runner(monkeypatch):
    from experiment import loop as loop_module

    class _Task:
        scorer = "dummy"

        def bind(self, state, *, example_pool=None):
            self.state = state

    phases = []

    def fake_run_config(
        store,
        config_id,
        queries,
        task,
        model,
        llm_call,
        *,
        num_workers,
        on_conflict,
        phase=None,
    ):
        phases.append(phase)

    def fake_evaluate_config(*args, **kwargs):
        return None

    monkeypatch.setattr(loop_module, "run_config", fake_run_config)
    monkeypatch.setattr(loop_module, "evaluate_config", fake_evaluate_config)

    store = CubeStore(":memory:")
    try:
        store.upsert_queries(
            [{
                "query_id": "q1",
                "dataset": "wtq",
                "content": "question",
                "meta": {"split": "test"},
            }],
            on_conflict=OnConflict.ERROR,
        )

        loop_module._run_and_eval_plan(
            store,
            [RunEntry(config_id=1, func_ids=[], query_ids=["q1"])],
            _Task,
            "model",
            lambda *_args, **_kwargs: {},
            phase="lengthfix",
            dataset="wtq",
        )

        assert phases == ["lengthfix"]
    finally:
        store.close()
