from core.preprocess import annotate_types as generic_annotate_types
from tasks.wtq.table_transforms import (
    _detect_column_type,
    annotate_types,
    compute_column_stats,
)


def test_wtq_type_annotation_distinguishes_percent_from_int():
    header = ["win rate", "wins"]
    rows = [["10%", "10"], ["20%", "20"], ["30%", "30"]]

    assert _detect_column_type(["10%", "20%", "30%"]) == "percent"
    assert annotate_types(header, rows) == ["win rate (percent)", "wins (int)"]


def test_wtq_column_stats_preserve_percent_type():
    stats = compute_column_stats(
        ["win rate", "wins"],
        [["10%", "10"], ["20%", "20"], ["30%", "30"]],
    )

    assert "win rate (percent" in stats
    assert "10%" in stats
    assert "win rate (int" not in stats
    assert "wins (int" in stats


def test_generic_preprocess_type_annotation_distinguishes_percent_from_int():
    header, rows = generic_annotate_types(
        ["win rate", "wins"],
        [["10%", "10"], ["20%", "20"], ["30%", "30"]],
        "which team has the highest win rate",
    )

    assert header == ["win rate (percent)", "wins (int)"]
    assert rows == [["10%", "10"], ["20%", "20"], ["30%", "30"]]
