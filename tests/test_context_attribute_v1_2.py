import json

from study_layer.context_attribute.lightweight_extractors import canonical_atoms, indicator_atoms


def _meta(question, header, rows):
    return {
        "_raw": {
            "question": question,
            "table": {"header": header, "rows": rows},
        }
    }


def test_registry_v1_2_does_not_emit_entity_ontology_atoms():
    atoms = canonical_atoms(
        _meta(
            "which country had the highest score?",
            ["country", "team", "score"],
            [["spain", "a", "2-1"], ["france", "b", "1-0"]],
        ),
        "wtq",
    )
    assert "schema.has_entity_col" not in atoms
    assert "text.answer_role_entity" not in atoms
    assert atoms["schema.has_score_col"] == "yes"


def test_score_surface_split_from_generic_numeric_range():
    score_atoms = canonical_atoms(
        _meta("what was the score?", ["team", "score"], [["a", "18-12"]]),
        "wtq",
    )
    assert score_atoms["cell.has_score_surface"] == "yes"

    range_atoms = canonical_atoms(
        _meta("who served between 1999 and 2004?", ["name", "years"], [["a", "1999 - 2004"]]),
        "wtq",
    )
    assert range_atoms["cell.has_numeric_range"] == "yes"
    assert range_atoms["cell.has_score_surface"] == "no"


def test_indicator_atoms_only_marks_yes_binary_values():
    registry_atoms = {
        "cell.has_score_surface": {"value_type": "binary"},
        "table.rows_bin": {"value_type": "categorical"},
    }
    indicators = indicator_atoms(
        {"cell.has_score_surface": "yes", "table.rows_bin": "0_5"},
        registry_atoms,
    )
    assert indicators == {"cell.has_score_surface", "table.rows_bin=0_5"}
