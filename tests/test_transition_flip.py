import json
import sqlite3

from analyze.transition_flip import (
    AtomEditSpec,
    TransitionSpec,
    aggregate_query_effects,
    build_atom_edit_paired_contrasts,
    build_paired_contrasts,
    family_assignments,
    inventory_transitions,
    join_query_effects_with_context,
    load_config_atoms,
    load_config_family_assignments,
    load_context_atoms,
    load_scored_executions,
    score_context_rules,
    summarize_background_effects,
)


def make_cube() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE config (config_id INTEGER PRIMARY KEY, func_ids TEXT DEFAULT '[]', meta TEXT DEFAULT '{}');
        CREATE TABLE feature (feature_id TEXT PRIMARY KEY, canonical_id TEXT NOT NULL, task TEXT NOT NULL, requires_json TEXT DEFAULT '[]', conflicts_json TEXT DEFAULT '[]', primitive_spec TEXT DEFAULT '{}', semantic_labels_json TEXT DEFAULT '[]', scope_json TEXT DEFAULT '{}');
        CREATE TABLE config_feature (config_id INTEGER NOT NULL, feature_id TEXT NOT NULL, role TEXT DEFAULT 'feature', PRIMARY KEY(config_id, feature_id));
        CREATE TABLE query (query_id TEXT PRIMARY KEY, dataset TEXT NOT NULL, content TEXT NOT NULL, meta TEXT DEFAULT '{}');
        CREATE TABLE execution (execution_id INTEGER PRIMARY KEY, config_id INTEGER NOT NULL, query_id TEXT NOT NULL, model TEXT NOT NULL, error TEXT DEFAULT '');
        CREATE TABLE evaluation (eval_id INTEGER PRIMARY KEY, execution_id INTEGER NOT NULL, scorer TEXT NOT NULL, score REAL, metrics TEXT DEFAULT '{}');
        CREATE TABLE predicate (query_id TEXT NOT NULL, name TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY(query_id, name));
        """
    )
    atoms = [
        "prompt_format_plain",
        "table_serialization_json_columns_data",
        "table_serialization_html",
        "output_contract_json_answer_list",
        "reasoning_symbolic_operation",
        "input_context_column_statistics",
        "input_context_type_annotation",
    ]
    for i, atom in enumerate(atoms, 1):
        conn.execute(
            "INSERT INTO feature(feature_id, canonical_id, task) VALUES (?, ?, 'wtq')",
            (f"f{i}", atom),
        )
    configs = {
        1: ["prompt_format_plain", "table_serialization_json_columns_data", "output_contract_json_answer_list"],
        2: ["prompt_format_plain", "table_serialization_json_columns_data", "output_contract_json_answer_list", "reasoning_symbolic_operation"],
        3: ["prompt_format_plain", "table_serialization_html", "output_contract_json_answer_list"],
        4: ["prompt_format_plain", "table_serialization_html", "output_contract_json_answer_list", "reasoning_symbolic_operation"],
        5: ["prompt_format_plain", "table_serialization_json_columns_data", "output_contract_json_answer_list", "input_context_column_statistics"],
        6: ["prompt_format_plain", "table_serialization_json_columns_data", "output_contract_json_answer_list", "input_context_type_annotation"],
        7: ["prompt_format_plain", "table_serialization_json_columns_data", "output_contract_json_answer_list", "input_context_type_annotation", "input_context_column_statistics"],
    }
    atom_to_fid = {atom: f"f{i}" for i, atom in enumerate(atoms, 1)}
    for cid, c_atoms in configs.items():
        conn.execute(
            "INSERT INTO config(config_id, meta) VALUES (?, ?)",
            (cid, json.dumps({"canonical_ids": c_atoms})),
        )
        for atom in c_atoms:
            conn.execute("INSERT INTO config_feature(config_id, feature_id) VALUES (?, ?)", (cid, atom_to_fid[atom]))
    for qid in ["q1", "q2", "q3"]:
        conn.execute("INSERT INTO query(query_id, dataset, content, meta) VALUES (?, 'wtq', ?, '{}')", (qid, qid))
    predicates = {
        "q1": {"operation_type": "count", "n_cols": "5"},
        "q2": {"operation_type": "lookup", "n_cols": "5"},
        "q3": {"operation_type": "count", "n_cols": "8"},
    }
    for qid, items in predicates.items():
        for name, value in items.items():
            conn.execute("INSERT INTO predicate(query_id, name, value) VALUES (?, ?, ?)", (qid, name, value))

    scores = {
        # json_columns background: up, stable success, down
        (1, "q1"): 0, (2, "q1"): 1,
        (1, "q2"): 1, (2, "q2"): 1,
        (1, "q3"): 1, (2, "q3"): 0,
        # html background: up, stable fail, stable success
        (3, "q1"): 0, (4, "q1"): 1,
        (3, "q2"): 0, (4, "q2"): 0,
        (3, "q3"): 1, (4, "q3"): 1,
        # context stats, same background as config 1 except added stats
        (5, "q1"): 1, (5, "q2"): 1, (5, "q3"): 0,
        # type and type+stats: tests atom edits with another same-family atom fixed
        (6, "q1"): 0, (7, "q1"): 1,
        (6, "q2"): 0, (7, "q2"): 0,
        (6, "q3"): 1, (7, "q3"): 1,
    }
    eid = 1
    for (cid, qid), score in scores.items():
        conn.execute(
            "INSERT INTO execution(execution_id, config_id, query_id, model) VALUES (?, ?, ?, 'm')",
            (eid, cid, qid),
        )
        conn.execute(
            "INSERT INTO evaluation(execution_id, scorer, score) VALUES (?, 'exact', ?)",
            (eid, score),
        )
        eid += 1
    conn.commit()
    return conn


def test_family_assignments_multi_hot_and_absent():
    out = family_assignments([
        "prompt_format_plain",
        "table_serialization_json_records",
        "input_context_type_annotation",
        "input_context_column_statistics",
    ])
    assert out["prompt_format"] == "plain"
    assert out["table_serialization"] == "json_records"
    assert out["input_context"] == "column_statistics+type_annotation"
    assert out["reasoning"] == "absent"


def test_exact_paired_transition_and_background_summary():
    conn = make_cube()
    assignments = load_config_family_assignments(conn)
    rows = load_scored_executions(conn)
    spec = TransitionSpec("reasoning", "absent", "symbolic_operation")
    pairs = build_paired_contrasts(rows, assignments, spec)
    assert len(pairs) == 6
    assert {p["flip_type"] for p in pairs} == {"up_flip", "down_flip", "stable_success", "stable_fail"}

    backgrounds = summarize_background_effects(pairs)
    assert len(backgrounds) == 2
    assert sum(int(bg["n_pairs"]) for bg in backgrounds) == 6


def test_inventory_finds_supported_transitions():
    conn = make_cube()
    assignments = load_config_family_assignments(conn)
    rows = load_scored_executions(conn)
    inv = inventory_transitions(rows, assignments, families=["reasoning", "table_serialization"])
    labels = {(r["family"], r["from_value"], r["to_value"]): r for r in inv}
    assert ("reasoning", "absent", "symbolic_operation") in labels
    assert labels[("reasoning", "absent", "symbolic_operation")]["n_pairs"] == 6
    assert ("table_serialization", "json_columns_data", "html") in labels


def test_context_rule_scoring_uses_query_level_effects():
    conn = make_cube()
    assignments = load_config_family_assignments(conn)
    rows = load_scored_executions(conn)
    pairs = build_paired_contrasts(rows, assignments, TransitionSpec("reasoning", "absent", "symbolic_operation"))
    query_effects = aggregate_query_effects(pairs)
    context = load_context_atoms(conn, [r["query_id"] for r in query_effects])
    joined = join_query_effects_with_context(query_effects, context)
    pos, neg = score_context_rules(joined, min_support=1, max_len=1, top_k=10)
    assert pos or neg
    assert any("n_cols=" in r["rule"] for r in pos + neg)
    assert all(r["support"] >= 1 for r in pos + neg)


def test_atom_edit_pairs_keep_other_same_family_atoms_in_background():
    conn = make_cube()
    atoms_by_config = load_config_atoms(conn)
    rows = load_scored_executions(conn)
    pairs = build_atom_edit_paired_contrasts(
        rows,
        atoms_by_config,
        AtomEditSpec(add_atoms=("input_context_column_statistics",), family="input_context"),
    )
    assert len(pairs) == 6
    assert len({p["background_id"] for p in pairs}) == 2
    assert any(p["config_from_ids"] == "6" and p["config_to_ids"] == "7" for p in pairs)
    assert any("input_context_type_annotation" in p["background_id"] for p in pairs)
