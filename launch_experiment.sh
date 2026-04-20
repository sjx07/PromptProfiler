#!/usr/bin/env bash
# Launch experiment under nohup with full reproducibility snapshot.
#
# Usage:
#   bash launch_experiment.sh <preflight.md> <config.json> [--cli-overrides ...]
#
# Both args REQUIRED and independent. Preflight.md is a pure reasoning
# document (hypothesis, confounds, decision log); config.json is the pure
# execution artifact consumed by run_experiment.py. They stay decoupled
# on disk and in the run snapshot.
#
# As a convenience, if the preflight happens to contain a
# `- **Config:** <path>` line, it is verified to match the config arg —
# a mismatch is a hard error (catches "wrong preflight for this config"
# mistakes).
#
# Run artifacts land under `experiments/<task>_<ts>/`:
#   preflight.md         — copied from arg 1
#   config.json          — copied from arg 2
#   used_features.json   — canonical_ids + feature_id + primitive_edits
#                          for every base + experiment feature
#   meta.json            — git hash/branch, dirty flag, PID, python version,
#                          conda env, hostname, cube path, launch command,
#                          started_at, CLI overrides
#   pip_freeze.txt       — `pip freeze` snapshot
#   uncommitted.diff     — saved iff the work tree was dirty at launch
#   nohup.log            — stdout+stderr of the run
set -euo pipefail

# ── arg parsing + dual-required check ────────────────────────────────

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <preflight.md> <config.json> [--cli-overrides ...]" >&2
    echo "  Both preflight.md and config.json are REQUIRED." >&2
    exit 1
fi

PREFLIGHT="$1"
CONFIG="$2"
shift 2
CLI_OVERRIDES="$*"

if [[ "$PREFLIGHT" != *.md ]]; then
    echo "ERROR: first arg must be a preflight .md (got '$PREFLIGHT')." >&2
    exit 1
fi
if [[ ! -f "$PREFLIGHT" ]]; then
    echo "ERROR: Preflight file '$PREFLIGHT' does not exist." >&2
    exit 1
fi
if [[ "$CONFIG" != *.json ]]; then
    echo "ERROR: second arg must be a config .json (got '$CONFIG')." >&2
    exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config file '$CONFIG' does not exist." >&2
    exit 1
fi

# Optional sanity check: if preflight declares a config path, it must match.
DECLARED="$(sed -n 's/^- \*\*Config:\*\* //p' "$PREFLIGHT" | head -1 | tr -d '[:space:]' || true)"
if [[ -n "${DECLARED:-}" ]]; then
    # Resolve both to absolute paths for comparison.
    DECLARED_ABS="$(cd "$(dirname "$DECLARED")" 2>/dev/null && pwd)/$(basename "$DECLARED")" || DECLARED_ABS="$DECLARED"
    CONFIG_ABS="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
    if [[ "$DECLARED_ABS" != "$CONFIG_ABS" ]]; then
        echo "ERROR: preflight declares Config: '$DECLARED' but arg says '$CONFIG'." >&2
        echo "  Remove the 'Config:' line from preflight OR fix the mismatch." >&2
        exit 1
    fi
fi

# ── parse config for task + db path ──────────────────────────────────

TASK="$(python3 -c "import json; print(json.load(open('$CONFIG')).get('task', 'experiment'))")"
DB_PATH="$(python3 -c "import json; print(json.load(open('$CONFIG')).get('db_path', ''))")"

# ── create run directory under ./experiments/ ────────────────────────

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="experiments/${TASK}_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

# ── snapshot inputs: preflight + config verbatim ─────────────────────

cp "$PREFLIGHT" "$RUN_DIR/preflight.md"
cp "$CONFIG"    "$RUN_DIR/config.json"

# ── snapshot used_features.json (canonical_id + feature_id + primitive_spec) ──

python3 <<PYEOF
import json
from pathlib import Path
from core.feature_registry import FeatureRegistry

with open("$CONFIG") as f:
    cfg = json.load(f)

task = cfg.get("task", "")
base_ids = list(cfg.get("base_features", []))
exp_ids  = list(cfg.get("experiment_features", []))
all_ids  = sorted(set(base_ids + exp_ids))

reg = FeatureRegistry.load(task=task)
by_cid = {s["canonical_id"]: s for s in reg.all_specs()}
resolved = []
for canonical in all_ids:
    spec = by_cid.get(canonical)
    if spec is None:
        resolved.append({
            "canonical_id":   canonical,
            "role":           ("base" if canonical in base_ids else "experiment"),
            "resolve_error":  f"canonical_id not found in task '{task}'",
        })
        continue
    resolved.append({
        "canonical_id":    canonical,
        "feature_id":      spec.get("feature_id"),
        "role":            ("base" if canonical in base_ids else "experiment"),
        "primitive_edits": spec.get("primitive_edits"),
        "conflicts_with":  list(spec.get("conflicts_with", []) or []),
        "requires":        list(spec.get("requires", []) or []),
        "source_path":     spec.get("_source_path"),
    })

out = {
    "task":                task,
    "base_features":       base_ids,
    "experiment_features": exp_ids,
    "resolved":            resolved,
}
Path("$RUN_DIR/used_features.json").write_text(json.dumps(out, indent=2))
PYEOF

# ── git state (hash, branch, dirty?, uncommitted.diff) ───────────────

GIT_HASH="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="false"
if [[ "$GIT_HASH" != "unknown" ]] && [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
    GIT_DIRTY="true"
    # Save both unstaged and staged diffs side by side for reproducibility.
    {
        echo "# === git status ==="
        git status --short
        echo
        echo "# === git diff (unstaged) ==="
        git diff
        echo
        echo "# === git diff --cached (staged) ==="
        git diff --cached
    } > "$RUN_DIR/uncommitted.diff" 2>/dev/null || true
    echo "WARNING: working tree is dirty — saved diff to $RUN_DIR/uncommitted.diff" >&2
fi

# ── pip freeze snapshot (lightweight env record) ─────────────────────

python3 -m pip freeze > "$RUN_DIR/pip_freeze.txt" 2>/dev/null || true

# ── meta.json (comprehensive) ────────────────────────────────────────

PY_VERSION="$(python3 -c 'import sys; print(sys.version.replace(chr(10)," "))')"
LAUNCH_CMD="bash $0 $PREFLIGHT $CLI_OVERRIDES"

cat > "$RUN_DIR/meta.json" <<METAEOF
{
  "run_dir":        "$RUN_DIR",
  "preflight":      "$PREFLIGHT",
  "config":         "$CONFIG",
  "task":           "$TASK",
  "cube_db_path":   "$DB_PATH",
  "cli_overrides":  "$CLI_OVERRIDES",
  "launch_cmd":     "$LAUNCH_CMD",
  "git_hash":       "$GIT_HASH",
  "git_branch":     "$GIT_BRANCH",
  "git_dirty":      $GIT_DIRTY,
  "hostname":       "$(hostname)",
  "conda_env":      "${CONDA_DEFAULT_ENV:-none}",
  "python_version": "$PY_VERSION",
  "started_at":     "$(date -Iseconds)",
  "pid":            null
}
METAEOF

# ── launch under nohup ───────────────────────────────────────────────

nohup bash -c "
    set -euo pipefail
    python3 -m run_experiment '$CONFIG' $CLI_OVERRIDES
    echo '=== Done ==='
" > "$RUN_DIR/nohup.log" 2>&1 &

PID=$!

# Patch PID into meta.json now that we have it.
python3 -c "
import json, sys
path = f'{sys.argv[1]}/meta.json'
with open(path) as f:
    meta = json.load(f)
meta['pid'] = int(sys.argv[2])
with open(path, 'w') as f:
    json.dump(meta, f, indent=2)
" "$RUN_DIR" "$PID"

# ── banner ───────────────────────────────────────────────────────────

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Experiment launched                                     ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  %-55s ║\n" "Preflight: $PREFLIGHT"
printf "║  %-55s ║\n" "Config:    $CONFIG"
printf "║  %-55s ║\n" "Task:      $TASK"
printf "║  %-55s ║\n" "Cube:      $DB_PATH"
printf "║  %-55s ║\n" "PID:       $PID"
printf "║  %-55s ║\n" "Git:       ${GIT_HASH:0:12} ($GIT_BRANCH, dirty=$GIT_DIRTY)"
printf "║  %-55s ║\n" "Run dir:   $RUN_DIR"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  %-55s ║\n" "Tail log:  tail -f $RUN_DIR/nohup.log"
printf "║  %-55s ║\n" "Stop:      kill $PID"
echo "╚══════════════════════════════════════════════════════════╝"
