#!/usr/bin/env bash
# Launch experiment under nohup with metadata and PID tracking.
#
# Usage:
#   bash prompt_profiler/launch_experiment.sh \
#       prompt_profiler/runs/exp_wtq_7b_code.json
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.json> [--cli-overrides ...]" >&2
    exit 1
fi

INPUT="$1"
shift
CLI_OVERRIDES="$*"

# ── resolve config: accept .md (experiment doc) or .json ──────────
if [[ "$INPUT" == *.md ]]; then
    # Extract Config: line from experiment doc
    CONFIG="$(grep -oP '^\- \*\*Config:\*\* \K.+' "$INPUT" | tr -d '[:space:]')"
    if [[ -z "$CONFIG" ]]; then
        echo "ERROR: No '- **Config:** <path>' found in $INPUT" >&2
        exit 1
    fi
    if [[ ! -f "$CONFIG" ]]; then
        echo "ERROR: Config '$CONFIG' referenced in $INPUT does not exist" >&2
        exit 1
    fi
    EXPERIMENT_DOC="$INPUT"
    echo "Experiment doc: $INPUT"
    echo "Config:         $CONFIG"
else
    CONFIG="$INPUT"
    EXPERIMENT_DOC=""
fi

# ── extract task name from config for run dir naming ─────────────
TASK="$(python3 -c "import json; print(json.load(open('$CONFIG')).get('task', 'experiment'))")"

# ── create run directory ─────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="runs/${TASK}_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

# ── collect metadata ─────────────────────────────────────────────
GIT_HASH="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"

cat > "$RUN_DIR/meta.json" <<METAEOF
{
  "config": "$CONFIG",
  "experiment_doc": "$EXPERIMENT_DOC",
  "cli_overrides": "$CLI_OVERRIDES",
  "git_hash": "$GIT_HASH",
  "git_branch": "$GIT_BRANCH",
  "hostname": "$(hostname)",
  "conda_env": "${CONDA_DEFAULT_ENV:-none}",
  "started_at": "$(date -Iseconds)",
  "pid": null
}
METAEOF

# ── launch ───────────────────────────────────────────────────────
nohup bash -c "
    set -euo pipefail
    cd '$REPO_ROOT'
    python3 -m prompt_profiler.run_experiment \
        '$CONFIG' $CLI_OVERRIDES
    echo '=== Done ==='
" > "$RUN_DIR/nohup.log" 2>&1 &

PID=$!

# Patch PID into metadata
python3 -c "
import json, sys
path = f'{sys.argv[1]}/meta.json'
with open(path) as f:
    meta = json.load(f)
meta['pid'] = int(sys.argv[2])
with open(path, 'w') as f:
    json.dump(meta, f, indent=2)
" "$RUN_DIR" "$PID"

# ── banner ───────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Experiment launched                                    ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  %-54s  ║\n" "Config: $CONFIG"
printf "║  %-54s  ║\n" "Task:   $TASK"
printf "║  %-54s  ║\n" "PID:    $PID"
printf "║  %-54s  ║\n" "Git:    ${GIT_HASH:0:12} ($GIT_BRANCH)"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  %-54s  ║\n" "Log:  tail -f $RUN_DIR/nohup.log"
printf "║  %-54s  ║\n" "Stop: kill $PID"
echo "╚══════════════════════════════════════════════════════════╝"
