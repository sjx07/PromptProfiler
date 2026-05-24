from __future__ import annotations

from core.store import CubeStore, OnConflict
from analyze.context_attributes import canonical_context_items, canonical_context_values
from analyze.facet_matrix_ops import query_context_sets


def test_canonical_context_values_are_shared_and_input_only():
    meta = {
        "split": "test",
        "_raw": {
            "question": "How many teams scored after 2001?",
            "table": {
                "header": ["Team", "Year", "Score"],
                "rows": [["A", "2001", "3-2"], ["B", "2002", "4-1"]],
            },
        },
    }

    values = canonical_context_values(meta, "wtq")

    assert values["intent.count"] == "yes"
    assert values["intent.temporal"] == "yes"
    assert values["schema.has_score_col"] == "yes"
    assert values["schema.has_date_col"] == "yes"
    assert values["cell.has_range_or_score"] == "yes"
    assert values["dialog.has_reference_marker"] == "na"


def test_canonical_context_items_support_shared_and_scoped_views():
    meta = {
        "split": "test",
        "_raw": {
            "question": "Which of these teams had the highest score?",
            "position": 1,
            "table": {
                "headers": ["Team", "Score"],
                "rows": [["A", "4"], ["B", "7"]],
            },
        },
    }

    shared = set(canonical_context_items(meta, "sqa", view="shared"))
    scoped = set(canonical_context_items(meta, "sqa", view="scoped"))

    assert "intent.superlative_rank=yes" in shared
    assert all(not atom.startswith("dialog.") for atom in shared)
    assert "dialog.has_reference_marker=yes" in scoped
    assert "dialog.turn_bin=early" in scoped


def test_query_context_sets_can_use_canonical_context(tmp_path):
    store = CubeStore(tmp_path / "cube.db")
    store.upsert_queries(
        [
            {
                "query_id": "q1",
                "dataset": "wtq",
                "content": "How many teams scored after 2001?",
                "meta": {
                    "split": "test",
                    "_raw": {
                        "question": "How many teams scored after 2001?",
                        "table": {
                            "header": ["Team", "Year", "Score"],
                            "rows": [["A", "2001", "3-2"], ["B", "2002", "4-1"]],
                        },
                    },
                },
            }
        ],
        on_conflict=OnConflict.SKIP,
    )

    rows = query_context_sets(
        store,
        dataset="wtq",
        split="test",
        context_view="canonical_shared",
    )

    atoms = rows.iloc[0]["context_atoms"]
    assert "intent.count=yes" in atoms
    assert "schema.has_score_col=yes" in atoms
    assert "cell.has_range_or_score=yes" in atoms
    assert all(not atom.startswith("dialog.") for atom in atoms)
    store.close()


def test_tabfact_alias_uses_statement_text():
    meta = {
        "split": "validation",
        "_raw": {
            "statement": "The team scored before 2001.",
            "table_text": "team#year\nA#1999",
        },
    }

    values = canonical_context_values(meta, "tabfact")

    assert values["intent.temporal"] == "yes"
    assert values["grounding.header_overlap"] == "yes"
