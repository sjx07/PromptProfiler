"""Overall exact atom-edit effects for a FACET observation cube.

This module computes an overview table where each row is a prompt feature atom
under a dataset/model pair.  For each atom, it compares configs that differ only
by that atom and hold the remaining prompt atom set fixed.  This is broader than
a single transition audit and avoids treating multi-hot family values as
unrelated categorical states.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from analyze.transition_flip import (
    DEFAULT_BACKGROUND_IGNORE,
    DEFAULT_PREFIX_RULES,
    atom_family_value,
    connect,
    has_table,
    load_config_atoms,
)


def split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(',') if part.strip()]


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in keys})


def feature_family(atom: str) -> str:
    mapped = atom_family_value(atom, rules=DEFAULT_PREFIX_RULES)
    return mapped[0] if mapped else 'unknown'


def kept_atoms(atoms: Iterable[str], atom: str, ignore_families: Sequence[str]) -> Tuple[str, ...]:
    ignored = set(ignore_families)
    out = []
    for item in atoms:
        if item == atom:
            continue
        if feature_family(item) in ignored:
            continue
        out.append(item)
    return tuple(sorted(set(out)))


def discover_tasks(
    conn: sqlite3.Connection,
    *,
    datasets: Sequence[str],
    models: Sequence[str],
    min_configs: int,
) -> List[Tuple[str, str, int, int, int]]:
    params: List[Any] = []
    where = ['ev.score IS NOT NULL']
    if datasets:
        where.append('q.dataset IN (%s)' % ','.join('?' for _ in datasets))
        params.extend(datasets)
    if models:
        where.append('e.model IN (%s)' % ','.join('?' for _ in models))
        params.extend(models)
    rows = conn.execute(
        f"""
        SELECT q.dataset,
               e.model,
               COUNT(*) AS eval_rows,
               COUNT(DISTINCT e.config_id) AS configs,
               COUNT(DISTINCT e.query_id) AS queries
        FROM execution e
        JOIN query q ON q.query_id = e.query_id
        JOIN evaluation ev ON ev.execution_id = e.execution_id
        WHERE {' AND '.join(where)}
        GROUP BY q.dataset, e.model
        HAVING configs >= ?
        ORDER BY q.dataset, e.model
        """,
        tuple(params + [min_configs]),
    ).fetchall()
    return [(str(r['dataset']), str(r['model']), int(r['eval_rows']), int(r['configs']), int(r['queries'])) for r in rows]


def load_scores_frame(db_path: str, dataset: str, model: str, max_rows: int = 0) -> pd.DataFrame:
    limit_sql = ''
    params: List[Any] = [dataset, model]
    if max_rows > 0:
        limit_sql = ' LIMIT ?'
        params.append(int(max_rows))
    query = f"""
        SELECT e.query_id, e.config_id, ev.score
        FROM execution e
        JOIN query q ON q.query_id = e.query_id
        JOIN evaluation ev ON ev.execution_id = e.execution_id
        WHERE ev.score IS NOT NULL
          AND q.dataset = ?
          AND e.model = ?
        {limit_sql}
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=params)


def summarize_atom(
    pivot: pd.DataFrame,
    config_atoms: Mapping[int, Sequence[str]],
    atom: str,
    *,
    ignore_families: Sequence[str],
) -> Optional[Dict[str, Any]]:
    config_ids = [int(c) for c in pivot.columns]
    by_bg: Dict[Tuple[str, ...], Dict[str, List[int]]] = defaultdict(lambda: {'present': [], 'absent': []})
    for cid in config_ids:
        atoms = tuple(config_atoms.get(cid, ()))
        bg = kept_atoms(atoms, atom, ignore_families)
        side = 'present' if atom in atoms else 'absent'
        by_bg[bg][side].append(cid)

    n_pairs = 0
    n_backgrounds = 0
    queries = set()
    deltas: List[float] = []
    up = down = stable = 0
    background_effects = []
    for bg, sides in by_bg.items():
        present = [c for c in sides['present'] if c in pivot.columns]
        absent = [c for c in sides['absent'] if c in pivot.columns]
        if not present or not absent:
            continue
        y_to = pivot[present].mean(axis=1)
        y_from = pivot[absent].mean(axis=1)
        joined = pd.concat([y_from.rename('from'), y_to.rename('to')], axis=1).dropna()
        if joined.empty:
            continue
        bg_deltas = (joined['to'] - joined['from']).astype(float)
        n_backgrounds += 1
        n_pairs += int(len(bg_deltas))
        queries.update(str(x) for x in joined.index)
        vals = bg_deltas.tolist()
        deltas.extend(vals)
        bg_up = int((bg_deltas > 1e-12).sum())
        bg_down = int((bg_deltas < -1e-12).sum())
        bg_stable = int((bg_deltas.abs() <= 1e-12).sum())
        up += bg_up
        down += bg_down
        stable += bg_stable
        background_effects.append((float(bg_deltas.mean()), len(bg_deltas), bg))

    if not deltas:
        return None
    mean_delta = sum(deltas) / len(deltas)
    sorted_d = sorted(deltas)
    mid = len(sorted_d) // 2
    median_delta = sorted_d[mid] if len(sorted_d) % 2 else (sorted_d[mid - 1] + sorted_d[mid]) / 2
    bg_signs = [1 if x[0] > 1e-12 else -1 if x[0] < -1e-12 else 0 for x in background_effects]
    return {
        'atom': atom,
        'family': feature_family(atom),
        'n_pairs': n_pairs,
        'n_unique_queries': len(queries),
        'n_backgrounds': n_backgrounds,
        'mean_delta': mean_delta,
        'median_delta': median_delta,
        'up_count': up,
        'down_count': down,
        'stable_count': stable,
        'up_rate': up / n_pairs if n_pairs else 0.0,
        'down_rate': down / n_pairs if n_pairs else 0.0,
        'net_flip': (up - down) / n_pairs if n_pairs else 0.0,
        'background_positive_rate': sum(1 for s in bg_signs if s > 0) / len(bg_signs) if bg_signs else 0.0,
        'background_negative_rate': sum(1 for s in bg_signs if s < 0) / len(bg_signs) if bg_signs else 0.0,
    }


def run_task(args_tuple: Tuple[str, str, str, Dict[int, List[str]], Tuple[str, ...], int, int]) -> List[Dict[str, Any]]:
    db_path, dataset, model, config_atoms, ignore_families, min_pairs, max_rows = args_tuple
    df = load_scores_frame(db_path, dataset, model, max_rows=max_rows)
    if df.empty:
        return []
    pivot = df.pivot_table(index='query_id', columns='config_id', values='score', aggfunc='mean')
    present_configs = {int(c) for c in pivot.columns}
    atoms = sorted({atom for cid in present_configs for atom in config_atoms.get(cid, [])})
    rows: List[Dict[str, Any]] = []
    for atom in atoms:
        rec = summarize_atom(pivot, config_atoms, atom, ignore_families=ignore_families)
        if not rec or int(rec['n_pairs']) < min_pairs:
            continue
        rec.update({'dataset': dataset, 'model': model, 'n_configs': len(present_configs), 'n_eval_rows': len(df)})
        rows.append(rec)
    rows.sort(key=lambda r: (-abs(float(r['mean_delta'])), r['family'], r['atom']))
    return rows


def aggregate_family(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row['dataset']), str(row['model']), str(row['family']))].append(row)
    out = []
    for (dataset, model, family), items in grouped.items():
        weighted_pairs = sum(int(r['n_pairs']) for r in items)
        if weighted_pairs <= 0:
            continue
        weighted_delta = sum(float(r['mean_delta']) * int(r['n_pairs']) for r in items) / weighted_pairs
        out.append({
            'dataset': dataset,
            'model': model,
            'family': family,
            'n_atoms': len(items),
            'n_pairs': weighted_pairs,
            'weighted_mean_delta': weighted_delta,
            'positive_atoms': sum(1 for r in items if float(r['mean_delta']) > 0),
            'negative_atoms': sum(1 for r in items if float(r['mean_delta']) < 0),
            'max_positive_atom': max(items, key=lambda r: float(r['mean_delta']))['atom'],
            'max_positive_delta': max(float(r['mean_delta']) for r in items),
            'max_negative_atom': min(items, key=lambda r: float(r['mean_delta']))['atom'],
            'max_negative_delta': min(float(r['mean_delta']) for r in items),
        })
    out.sort(key=lambda r: (r['dataset'], r['model'], r['family']))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Overall exact atom-edit effects for a FACET cube')
    parser.add_argument('--db', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--datasets', default='')
    parser.add_argument('--models', default='')
    parser.add_argument('--ignore-families', default=','.join(DEFAULT_BACKGROUND_IGNORE))
    parser.add_argument('--min-configs', type=int, default=2)
    parser.add_argument('--min-pairs', type=int, default=100)
    parser.add_argument('--max-rows', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--progress', action='store_true')
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = split_csv(args.datasets)
    models = split_csv(args.models)
    ignore_families = tuple(split_csv(args.ignore_families) or DEFAULT_BACKGROUND_IGNORE)
    with connect(args.db) as conn:
        config_atoms = load_config_atoms(conn)
        tasks = discover_tasks(conn, datasets=datasets, models=models, min_configs=args.min_configs)
    task_args = [(args.db, ds, model, config_atoms, ignore_families, args.min_pairs, args.max_rows) for ds, model, _n, _c, _q in tasks]

    rows: List[Dict[str, Any]] = []
    iterator: Iterable[Any]
    max_workers = max(1, min(args.num_workers, len(task_args) or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_task, item): item for item in task_args}
        iterator = as_completed(futures)
        if args.progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=len(futures), desc='dataset/model')
        for fut in iterator:
            rows.extend(fut.result())

    rows.sort(key=lambda r: (r['dataset'], r['model'], r['family'], -abs(float(r['mean_delta'])), r['atom']))
    family_rows = aggregate_family(rows)
    write_csv(out_dir / 'overall_atom_effects.csv', rows)
    write_csv(out_dir / 'family_summary.csv', family_rows)
    write_csv(out_dir / 'top_positive_atoms.csv', sorted(rows, key=lambda r: -float(r['mean_delta']))[:100])
    write_csv(out_dir / 'top_negative_atoms.csv', sorted(rows, key=lambda r: float(r['mean_delta']))[:100])
    summary = {
        'db': args.db,
        'tasks': [{'dataset': ds, 'model': model, 'eval_rows': n, 'configs': c, 'queries': q} for ds, model, n, c, q in tasks],
        'n_rows': len(rows),
        'n_family_rows': len(family_rows),
        'min_pairs': args.min_pairs,
        'ignore_families': list(ignore_families),
        'num_workers': max_workers,
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
