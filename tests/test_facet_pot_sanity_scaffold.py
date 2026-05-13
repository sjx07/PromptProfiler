from __future__ import annotations

from core.feature_registry import FeatureRegistry
from core.func_registry import apply_config
from core.store import CubeStore
from tasks.sqa.sequential_qa import SequentialQA
from tasks.tabfact.fact_verification import FactVerification
from tasks.wtq.table_qa import TableQA


BASE_SECTIONS = [
    "_section_role",
    "_section_task",
    "_section_table_handling",
    "_section_reasoning",
    "_section_format_fix",
    "_section_rules",
]


def _bind_scaffold(task_name: str, task_cls: type, scaffold: str = "facet_pot_exec_scaffold"):
    reg = FeatureRegistry.load(task=task_name)
    specs, _ = reg.materialize(BASE_SECTIONS + [scaffold])
    store = CubeStore(":memory:")
    try:
        store.upsert_funcs(specs)
        state = apply_config([spec["func_id"] for spec in specs], store)
        task = task_cls()
        task.bind(state)
        return state, task, task._prompt_state._build_system_content()
    finally:
        store.close()


def test_wtq_non_pot_scaffold_contracts():
    cases = [
        ("facet_dp_scaffold", "answer", {"answer"}, "markdown"),
        ("facet_tcot_scaffold", "answer", {"reasoning", "answer"}, "markdown"),
        ("facet_scot_scaffold", "answer", {"symbolic_trace", "answer"}, "markdown"),
    ]
    for scaffold, dispatch_field, output_fields, table_format in cases:
        state, task, system_prompt = _bind_scaffold("wtq", TableQA, scaffold)

        assert task._dispatch_field() == dispatch_field
        assert set(task._prompt_state.semantic.output_fields) == output_fields
        assert state.table_format == table_format
        assert "answer:" in system_prompt


def test_sqa_non_pot_scaffold_contracts():
    cases = [
        ("facet_dp_scaffold", "answer", {"answer"}, "markdown"),
        ("facet_tcot_scaffold", "answer", {"reasoning", "answer"}, "markdown"),
        ("facet_scot_scaffold", "answer", {"symbolic_trace", "answer"}, "markdown"),
    ]
    for scaffold, dispatch_field, output_fields, table_format in cases:
        state, task, system_prompt = _bind_scaffold("sqa", SequentialQA, scaffold)

        assert task._dispatch_field() == dispatch_field
        assert set(task._prompt_state.semantic.output_fields) == output_fields
        assert state.table_format == table_format
        assert "answer:" in system_prompt


def test_tabfact_non_pot_scaffold_contracts():
    cases = [
        ("facet_dp_scaffold", "verdict", {"verdict"}, "markdown"),
        ("facet_tcot_scaffold", "verdict", {"reasoning", "verdict"}, "markdown"),
        ("facet_scot_scaffold", "verdict", {"symbolic_trace", "verdict"}, "markdown"),
    ]
    for scaffold, dispatch_field, output_fields, table_format in cases:
        state, task, system_prompt = _bind_scaffold("tabfact", FactVerification, scaffold)

        assert task._dispatch_field() == dispatch_field
        assert set(task._prompt_state.semantic.output_fields) == output_fields
        assert state.table_format == table_format
        assert "verdict:" in system_prompt


def _sample_sqa_query() -> dict:
    return {
        "query_id": "sample_sqa",
        "content": "what boats were lost on may 5?",
        "meta": {
            "_raw": {
                "question": "what boats were lost on may 5?",
                "table_file": "boats.csv",
                "history": [],
                "answer_text": ["U-638", "U-531"],
                "table": {
                    "headers": ["Date", "Number", "Type"],
                    "rows": [
                        ["4 May 1943", "U-209", "VIIC"],
                        ["5 May 1943", "U-638", "VIIC"],
                        ["5 May 1943", "U-531", "IXC/40"],
                    ],
                },
            }
        },
    }


def test_wtq_facet_pot_scaffold_contract():
    state, task, system_prompt = _bind_scaffold("wtq", TableQA)

    assert task._dispatch_field() == "code"
    assert set(task._prompt_state.semantic.output_fields) == {"code"}
    assert state.table_format == "json_records"
    assert "df is the authoritative pandas DataFrame" in system_prompt
    assert "`data` mirrors the displayed JSON records" in system_prompt
    assert "Do not rebuild the table from the displayed text" in system_prompt
    assert "Set `answer` to the final scalar or list value" in system_prompt
    assert "print(answer)" in system_prompt


def test_sqa_facet_pot_scaffold_contract():
    state, task, system_prompt = _bind_scaffold("sqa", SequentialQA)

    assert task._dispatch_field() == "code"
    assert set(task._prompt_state.semantic.output_fields) == {"code"}
    assert state.table_format == "json_records"
    assert "`history` contains prior question-answer turns" in system_prompt
    assert "`data` contains the displayed JSON records under `rows`" in system_prompt
    assert "Use `history` to resolve follow-up references" in system_prompt
    assert "Set `answer` to the final scalar or list value" in system_prompt
    assert "print(answer)" in system_prompt


def test_sqa_facet_pot_scaffold_renders_json_records():
    _state, task, _system_prompt = _bind_scaffold("sqa", SequentialQA)
    _sys, user_prompt = task.build_prompt(_sample_sqa_query())

    assert '"rows": [' in user_prompt
    assert '"Date": "5 May 1943"' in user_prompt
    assert "| Date | Number | Type |" not in user_prompt
    assert "conversation_history:" in user_prompt


def test_tabfact_facet_pot_scaffold_contract():
    state, task, system_prompt = _bind_scaffold("tabfact", FactVerification)

    assert task._dispatch_field() == "code"
    assert set(task._prompt_state.semantic.output_fields) == {"code"}
    assert state.table_format == "json_records"
    assert "df is the authoritative pandas DataFrame" in system_prompt
    assert "Set `answer` to a Python boolean" in system_prompt
    assert "print(answer)" in system_prompt
