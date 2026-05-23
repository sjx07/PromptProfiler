"""Per-family, per-feature exact atom-effect overview for FACET cubes.

For every feature atom, this computes the effect of adding that atom while
holding every other prompt atom fixed.  Results are grouped by dataset/model/
scorer and then aggregated to feature/family summaries.
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from analyze.transition_flip import (
    DEFAULT_BACKGROUND_IGNORE,
    DEFAULT_PREFIX_RULES,
    atom_family_value,
    connect,
    load_config_atoms,
)


EXCLUSIVE_FAMILIES = {
    'prompt_format',
    'table_serialization',
    'output_contract',
    'response_mode',
    'runtime_binding',
    'visible_reasoning',
}


def split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [p.strip() for p in value.split(',') if p.strip()]


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float('nan')


def med(xs: Sequence[float]) -> float:
    return float(median(xs)) if xs else float('nan')


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in keys})


def family_of(atom: str) -> str:
    mapped = atom_family_value(atom, rules=DEFAULT_PREFIX_RULES)
    return mapped[0] if mapped else 'unknown'


def config_has_family(atoms: Sequence[str], families: Sequence[str]) -> bool:
    blocked = set(families)
    return any(family_of(atom) in blocked for atom in atoms)


def normalized_atoms(atoms: Sequence[str], ignore_families: Sequence[str]) -> Tuple[str, ...]:
    ignored = set(ignore_families)
    return tuple(sorted(a for a in set(atoms) if family_of(a) not in ignored))


def background_without(atom: str, atoms: Sequence[str], ignore_families: Sequence[str]) -> Tuple[str, ...]:
    target_family = family_of(atom)
    if target_family in EXCLUSIVE_FAMILIES:
        return tuple(a for a in normalized_atoms(atoms, ignore_families) if family_of(a) != target_family)
    return tuple(a for a in normalized_atoms(atoms, ignore_families) if a != atom)


def discover_tasks(
    db_path: str,
    datasets: Sequence[str],
    models: Sequence[str],
    scorers: Sequence[str],
    min_configs: int,
) -> List[Dict[str, Any]]:
    # Fast setup path: avoid scanning evaluation during discovery.  Scorers are
    # loaded inside each task; the current table cubes use one scorer per
    # dataset/model in normal runs.
    params: List[Any] = []
    where = ['1=1']
    if datasets:
        where.append('q.dataset IN (%s)' % ','.join('?' for _ in datasets))
        params.extend(datasets)
    if models:
        where.append('e.model IN (%s)' % ','.join('?' for _ in models))
        params.extend(models)
    sql = f"""
        SELECT q.dataset, e.model, COUNT(DISTINCT e.config_id) AS configs
        FROM execution e
        JOIN query q ON q.query_id = e.query_id
        WHERE {' AND '.join(where)}
        GROUP BY q.dataset, e.model
        HAVING configs >= ?
        ORDER BY q.dataset, e.model
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = []
        for r in conn.execute(sql, tuple(params + [min_configs])).fetchall():
            rows.append({'dataset': str(r['dataset']), 'model': str(r['model']), 'scorer': '*', 'configs': int(r['configs'])})
    return rows


def load_task_scores(db_path: str, dataset: str, model: str, scorer: str) -> Tuple[Dict[int, Dict[str, float]], int, str]:
    params: List[Any] = [dataset, model]
    scorer_clause = ''
    if scorer and scorer != '*':
        scorer_clause = 'AND ev.scorer = ?'
        params.append(scorer)
    sql = f"""
        SELECT e.config_id, e.query_id, AVG(ev.score) AS score,
               GROUP_CONCAT(DISTINCT ev.scorer) AS scorers
        FROM evaluation ev
        JOIN execution e ON e.execution_id = ev.execution_id
        JOIN query q ON q.query_id = e.query_id
        WHERE ev.score IS NOT NULL
          AND q.dataset = ?
          AND e.model = ?
          {scorer_clause}
        GROUP BY e.config_id, e.query_id
    """
    scores: Dict[int, Dict[str, float]] = defaultdict(dict)
    scorer_names = set()
    n = 0
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for r in conn.execute(sql, tuple(params)):
            scores[int(r['config_id'])][str(r['query_id'])] = float(r['score'])
            for name in str(r['scorers'] or '').split(','):
                if name:
                    scorer_names.add(name)
            n += 1
    return dict(scores), n, '|'.join(sorted(scorer_names)) or scorer


def config_groups_for_atom(
    atom: str,
    config_ids: Iterable[int],
    atoms_by_config: Mapping[int, Sequence[str]],
    ignore_families: Sequence[str],
) -> Dict[Tuple[str, ...], Dict[str, List[int]]]:
    groups: Dict[Tuple[str, ...], Dict[str, List[int]]] = defaultdict(lambda: {'absent': [], 'present': []})
    target_family = family_of(atom)
    for cid in config_ids:
        atoms = atoms_by_config.get(cid, ())
        bg = background_without(atom, atoms, ignore_families)
        atom_set = set(atoms)
        if target_family in EXCLUSIVE_FAMILIES:
            # For choice families, compare this value against other values in
            # the same family under the same outside-family background.
            family_atoms = [a for a in atom_set if family_of(a) == target_family]
            side = 'present' if atom in atom_set else 'absent' if family_atoms else 'absent'
        else:
            # For additive/multi-hot families, keep other same-family atoms in
            # the background and compare atom absent vs present.
            side = 'present' if atom in atom_set else 'absent'
        groups[bg][side].append(cid)
    return groups


def summarize_atom(
    atom: str,
    scores: Mapping[int, Mapping[str, float]],
    atoms_by_config: Mapping[int, Sequence[str]],
    ignore_families: Sequence[str],
) -> Optional[Dict[str, Any]]:
    groups = config_groups_for_atom(atom, scores.keys(), atoms_by_config, ignore_families)
    deltas: List[float] = []
    up = down = stable = 0
    qids = set()
    bg_deltas: List[float] = []
    comparable_backgrounds = 0
    for bg, sides in groups.items():
        present = [cid for cid in sides['present'] if cid in scores]
        absent = [cid for cid in sides['absent'] if cid in scores]
        if not present or not absent:
            continue
        all_qids = set()
        for cid in present + absent:
            all_qids.update(scores[cid].keys())
        local: List[float] = []
        for qid in all_qids:
            pvals = [scores[cid][qid] for cid in present if qid in scores[cid]]
            avals = [scores[cid][qid] for cid in absent if qid in scores[cid]]
            if not pvals or not avals:
                continue
            delta = mean(pvals) - mean(avals)
            deltas.append(delta)
            local.append(delta)
            qids.add(qid)
            if delta > 1e-12:
                up += 1
            elif delta < -1e-12:
                down += 1
            else:
                stable += 1
        if local:
            comparable_backgrounds += 1
            bg_deltas.append(mean(local))
    if not deltas:
        return None
    n = len(deltas)
    return {
        'family': family_of(atom),
        'feature_atom': atom,
        'effect_kind': 'family_choice' if family_of(atom) in EXCLUSIVE_FAMILIES else 'atom_add',
        'n_pairs': n,
        'n_unique_queries': len(qids),
        'n_backgrounds': comparable_backgrounds,
        'mean_delta': mean(deltas),
        'median_delta': med(deltas),
        'up_count': up,
        'down_count': down,
        'stable_count': stable,
        'up_rate': up / n,
        'down_rate': down / n,
        'net_flip': (up - down) / n,
        'background_positive_rate': sum(1 for x in bg_deltas if x > 1e-12) / len(bg_deltas) if bg_deltas else 0.0,
        'background_negative_rate': sum(1 for x in bg_deltas if x < -1e-12) / len(bg_deltas) if bg_deltas else 0.0,
    }


def run_task(
    task: Mapping[str, Any],
    db_path: str,
    atoms_by_config: Mapping[int, Sequence[str]],
    families: Sequence[str],
    ignore_families: Sequence[str],
    exclude_config_families: Sequence[str],
    min_pairs: int,
) -> List[Dict[str, Any]]:
    dataset = str(task['dataset'])
    model = str(task['model'])
    scorer = str(task['scorer'])
    print(f"[load] {dataset} | {model} | {scorer}", flush=True)
    scores, n_loaded, scorer_label = load_task_scores(db_path, dataset, model, scorer)
    if exclude_config_families:
        scores = {
            cid: vals for cid, vals in scores.items()
            if not config_has_family(atoms_by_config.get(cid, ()), exclude_config_families)
        }
    config_ids = set(scores)
    atoms = sorted({a for cid in config_ids for a in atoms_by_config.get(cid, ())})
    if families:
        allowed = set(families)
        atoms = [a for a in atoms if family_of(a) in allowed]
    print(f"[compute] {dataset} | {model} | {scorer}: {len(atoms)} atoms, {len(config_ids)} configs, {n_loaded} rows", flush=True)
    rows: List[Dict[str, Any]] = []
    for atom in atoms:
        rec = summarize_atom(atom, scores, atoms_by_config, ignore_families)
        if not rec or int(rec['n_pairs']) < min_pairs:
            continue
        rec.update({
            'dataset': dataset,
            'model': model,
            'scorer': scorer_label,
            'n_configs': len(config_ids),
            'n_loaded_rows': n_loaded,
        })
        rows.append(rec)
    rows.sort(key=lambda r: (r['family'], -abs(float(r['mean_delta'])), r['feature_atom']))
    return rows


def aggregate_feature(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(str(r['family']), str(r['feature_atom']))].append(r)
    out = []
    for (family, atom), items in grouped.items():
        n = sum(int(r['n_pairs']) for r in items)
        if not n:
            continue
        out.append({
            'family': family,
            'feature_atom': atom,
            'n_dataset_model_scorers': len(items),
            'n_pairs': n,
            'weighted_mean_delta': sum(float(r['mean_delta']) * int(r['n_pairs']) for r in items) / n,
            'mean_of_means': mean([float(r['mean_delta']) for r in items]),
            'positive_tasks': sum(1 for r in items if float(r['mean_delta']) > 1e-12),
            'negative_tasks': sum(1 for r in items if float(r['mean_delta']) < -1e-12),
            'max_positive_delta': max(float(r['mean_delta']) for r in items),
            'max_negative_delta': min(float(r['mean_delta']) for r in items),
            'tasks': '|'.join(sorted(f"{r['dataset']}:{r['model']}:{r['scorer']}" for r in items)),
        })
    out.sort(key=lambda r: (r['family'], -abs(float(r['weighted_mean_delta'])), r['feature_atom']))
    return out


def aggregate_family(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[str(r['family'])].append(r)
    out = []
    for family, items in grouped.items():
        n = sum(int(r['n_pairs']) for r in items)
        if not n:
            continue
        out.append({
            'family': family,
            'n_feature_rows': len(items),
            'n_unique_features': len({r['feature_atom'] for r in items}),
            'n_pairs': n,
            'weighted_mean_delta': sum(float(r['mean_delta']) * int(r['n_pairs']) for r in items) / n,
            'positive_rows': sum(1 for r in items if float(r['mean_delta']) > 1e-12),
            'negative_rows': sum(1 for r in items if float(r['mean_delta']) < -1e-12),
            'top_positive_feature': max(items, key=lambda r: float(r['mean_delta']))['feature_atom'],
            'top_positive_delta': max(float(r['mean_delta']) for r in items),
            'top_negative_feature': min(items, key=lambda r: float(r['mean_delta']))['feature_atom'],
            'top_negative_delta': min(float(r['mean_delta']) for r in items),
        })
    out.sort(key=lambda r: -abs(float(r['weighted_mean_delta'])))
    return out


def write_report(out_dir: Path, rows: Sequence[Mapping[str, Any]], feature_rows: Sequence[Mapping[str, Any]], family_rows: Sequence[Mapping[str, Any]]) -> None:
    lines = ['# Overall Feature Atom Effects', '']
    lines.append('## Family Summary')
    for r in family_rows:
        lines.append(f"- `{r['family']}` rows={r['n_feature_rows']} features={r['n_unique_features']} weighted_delta={float(r['weighted_mean_delta']):+.4f} pos={r['positive_rows']} neg={r['negative_rows']}")
    lines.extend(['', '## Top Positive Features'])
    for r in sorted(feature_rows, key=lambda x: -float(x['weighted_mean_delta']))[:20]:
        lines.append(f"- `{r['family']}::{r['feature_atom']}` weighted_delta={float(r['weighted_mean_delta']):+.4f} tasks={r['positive_tasks']}/{r['n_dataset_model_scorers']}+ pairs={r['n_pairs']}")
    lines.extend(['', '## Top Negative Features'])
    for r in sorted(feature_rows, key=lambda x: float(x['weighted_mean_delta']))[:20]:
        lines.append(f"- `{r['family']}::{r['feature_atom']}` weighted_delta={float(r['weighted_mean_delta']):+.4f} tasks={r['negative_tasks']}/{r['n_dataset_model_scorers']}- pairs={r['n_pairs']}")
    out_dir.joinpath('report.md').write_text('\n'.join(lines) + '\n')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Per-family per-feature exact atom-effect overview')
    p.add_argument('--db', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--datasets', default='')
    p.add_argument('--models', default='')
    p.add_argument('--scorers', default='')
    p.add_argument('--families', default='')
    p.add_argument('--ignore-families', default=','.join(DEFAULT_BACKGROUND_IGNORE))
    p.add_argument('--exclude-config-families', default='')
    p.add_argument('--min-configs', type=int, default=2)
    p.add_argument('--min-pairs', type=int, default=100)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--progress', action='store_true')
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = split_csv(args.datasets)
    models = split_csv(args.models)
    scorers = split_csv(args.scorers)
    families = split_csv(args.families)
    ignore_families = tuple(split_csv(args.ignore_families) or DEFAULT_BACKGROUND_IGNORE)
    exclude_config_families = tuple(split_csv(args.exclude_config_families))
    print('[discover] tasks', flush=True)
    tasks = discover_tasks(args.db, datasets, models, scorers, args.min_configs)
    with connect(args.db) as conn:
        atoms_by_config = load_config_atoms(conn)
    print(f"[tasks] {len(tasks)} dataset/model/scorer tasks", flush=True)
    for t in tasks:
        print(f"  - {t['dataset']} | {t['model']} configs={t.get('configs', '')}", flush=True)

    all_rows: List[Dict[str, Any]] = []
    max_workers = max(1, min(args.num_workers, len(tasks) or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                run_task,
                t,
                args.db,
                atoms_by_config,
                families,
                ignore_families,
                exclude_config_families,
                args.min_pairs,
            )
            for t in tasks
        ]
        iterator = as_completed(futures)
        if args.progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=len(futures), desc='tasks')
        for fut in iterator:
            all_rows.extend(fut.result())
    all_rows.sort(key=lambda r: (r['dataset'], r['model'], r['scorer'], r['family'], -abs(float(r['mean_delta'])), r['feature_atom']))
    feature_rows = aggregate_feature(all_rows)
    family_rows = aggregate_family(all_rows)
    write_csv(out_dir / 'feature_effects_by_task.csv', all_rows)
    write_csv(out_dir / 'feature_effects_overall.csv', feature_rows)
    write_csv(out_dir / 'family_effects_overall.csv', family_rows)
    write_csv(out_dir / 'top_positive_feature_rows.csv', sorted(all_rows, key=lambda r: -float(r['mean_delta']))[:200])
    write_csv(out_dir / 'top_negative_feature_rows.csv', sorted(all_rows, key=lambda r: float(r['mean_delta']))[:200])
    summary = {
        'db': args.db,
        'n_tasks': len(tasks),
        'n_feature_task_rows': len(all_rows),
        'n_feature_overall_rows': len(feature_rows),
        'n_family_rows': len(family_rows),
        'datasets': datasets,
        'models': models,
        'families': families,
        'min_pairs': args.min_pairs,
        'ignore_families': list(ignore_families),
        'exclude_config_families': list(exclude_config_families),
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    write_report(out_dir, all_rows, feature_rows, family_rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
