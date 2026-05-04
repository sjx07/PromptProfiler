from pathlib import Path

from tasks.tablebench import loaders


FIXTURE = Path(__file__).with_name("tablebench_tableinstruct_sample.jsonl")


def test_tableinstruct_train_loader_defaults_to_all_instruction_types():
    rows = loaders._load_tableinstruct_train_rows(
        revision=None,
        cache_dir=None,
        train_data_path=str(FIXTURE),
        include_qtypes={"FactChecking", "NumericalReasoning", "DataAnalysis"},
        instruction_types=None,
    )

    assert [row["instruction_type"] for row in rows] == [
        "DP",
        "TCoT",
        "SCoT",
        "PoT",
    ]
    assert all(row["_source_dataset"] == "TableInstruct" for row in rows)


def test_tableinstruct_train_loader_keeps_explicit_instruction_filter():
    rows = loaders._load_tableinstruct_train_rows(
        revision=None,
        cache_dir=None,
        train_data_path=str(FIXTURE),
        include_qtypes={"FactChecking", "NumericalReasoning", "DataAnalysis"},
        instruction_types=["DP", "PoT"],
    )

    assert [row["instruction_type"] for row in rows] == ["DP", "PoT"]
