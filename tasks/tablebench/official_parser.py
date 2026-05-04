"""Official-style TableBench prediction parsing and code execution.

This mirrors the public TableBench parser while keeping generated code in an
isolated working directory under /data rather than writing table.csv in repo
root. It is intentionally narrow: enough for DP/TCoT/SCoT/PoT reproduction.
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


_FINAL_ANSWER_RE = re.compile(r"Final Answer\s*[:：]\s*(.+)", re.IGNORECASE)
_ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:Final\s+Answer|Final|Answer)\s*[:：]\s*",
    re.IGNORECASE,
)
_PYTHON_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_DEFAULT_EXEC_ROOT = Path("/data/users/jsu323/facet/tablebench_exec")


def parse_final_answer(prediction: str) -> str:
    """Match TableBench's DP parser: first single-line Final Answer wins."""
    try:
        match = _FINAL_ANSWER_RE.search(prediction or "")
    except Exception:
        return ""
    return _strip_answer_prefix(match.group(1)) if match else ""


def parse_python_code(prediction: str) -> str:
    try:
        matches = _PYTHON_BLOCK_RE.findall(prediction or "")
    except Exception:
        return ""
    return matches[-1] if matches else ""


def parse_code_output_prediction(output: str) -> str:
    parsed = parse_final_answer(output)
    if parsed:
        return parsed
    return _strip_answer_prefix(_last_nonempty_line(output))


def _last_nonempty_line(text: str) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _strip_answer_prefix(text: str) -> str:
    return _ANSWER_PREFIX_RE.sub("", str(text or "").strip()).strip()


def parse_general_code_then_exec(
    prediction: str,
    table: dict[str, Any],
    *,
    timeout_s: int = 15,
) -> tuple[str, bool]:
    """Parse PoT text answer by executing the final python block."""
    python_code = parse_python_code(prediction)
    if not python_code:
        return "", False

    result = _run_python_code(python_code, table, timeout_s=timeout_s)
    ecr_1 = result.returncode == 0 and not result.timed_out
    output_value = result.stdout if ecr_1 else ""
    parsed_prediction = parse_code_output_prediction(output_value) if output_value else ""
    if parsed_prediction == "" and output_value:
        parsed_prediction = output_value.strip()
    return parsed_prediction, ecr_1


def parse_chart_code_then_exec(
    prediction: str,
    table: dict[str, Any],
    answer: str,
    chart_type: str,
    *,
    timeout_s: int = 15,
) -> tuple[bool | str, bool]:
    """Execute generated chart code and evaluate official y-value tests."""
    python_code = parse_python_code(prediction)
    if not python_code:
        return "", False

    ecr_result = _run_python_code(python_code, table, timeout_s=timeout_s)
    ecr_1 = ecr_result.returncode == 0 and not ecr_result.timed_out

    eval_result = _run_chart_eval_code(
        python_code,
        table,
        answer=answer,
        chart_type=chart_type,
        timeout_s=timeout_s,
    )
    output_value = eval_result.stdout.strip() if eval_result.returncode == 0 else ""
    return {"True": True, "False": False}.get(output_value, ""), ecr_1


class _RunResult:
    def __init__(self, returncode: int, stdout: str, stderr: str, timed_out: bool) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out


def _run_python_code(
    python_code: str,
    table: dict[str, Any],
    *,
    timeout_s: int,
) -> _RunResult:
    payload = {
        "mode": "exec",
        "python_code": python_code,
    }
    return _run_payload(payload, table, timeout_s=timeout_s)


def _run_chart_eval_code(
    python_code: str,
    table: dict[str, Any],
    *,
    answer: str,
    chart_type: str,
    timeout_s: int,
) -> _RunResult:
    payload = {
        "mode": "chart_eval",
        "python_code": python_code,
        "answer": answer,
        "chart_type": chart_type,
    }
    return _run_payload(payload, table, timeout_s=timeout_s)


def _run_payload(
    payload: dict[str, Any],
    table: dict[str, Any],
    *,
    timeout_s: int,
) -> _RunResult:
    exec_root = Path(os.environ.get("TABLEBENCH_EXEC_DIR", str(_DEFAULT_EXEC_ROOT)))
    exec_root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]

    with tempfile.TemporaryDirectory(prefix="tb_", dir=exec_root) as workdir_s:
        workdir = Path(workdir_s)
        _write_table_csv(table, workdir / "table.csv")
        (workdir / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
        (workdir / "runner.py").write_text(_RUNNER_CODE, encoding="utf-8")

        env = dict(os.environ)
        env["MPLBACKEND"] = "Agg"
        env["PYTHONPATH"] = (
            str(repo_root)
            if not env.get("PYTHONPATH")
            else f"{repo_root}{os.pathsep}{env['PYTHONPATH']}"
        )
        try:
            completed = subprocess.run(
                [sys.executable, "runner.py"],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return _RunResult(
                completed.returncode,
                completed.stdout,
                completed.stderr,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            return _RunResult(
                returncode=124,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                timed_out=True,
            )


def _write_table_csv(table: dict[str, Any], path: Path) -> None:
    columns = table.get("columns") or table.get("header") or []
    rows = table.get("data") or table.get("rows") or []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


_RUNNER_CODE = r'''
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tasks.tablebench.chart_metric_utils import (
    compute_general_chart_metric,
    compute_pie_chart_metric,
    get_area_y_predictions,
    get_bar_y_predictions,
    get_hbar_y_predictions,
    get_line_y_predictions,
    get_pie_y_predictions,
    get_radar_y_predictions,
    get_scatter_y_predictions,
    get_waterfall_y_predictions,
)


def surround_pycode_with_main(pycode):
    start_line = "\nif __name__ == '__main__':\n"
    for line in pycode.strip().split("\n"):
        start_line += f"    {line}\n"
    return start_line


def execute(code):
    namespace = {"__name__": "__main__"}
    exec(code, namespace, namespace)


def chart_eval_code(chart_type):
    return f"""
if chart_type == 'line':
    y_predictions = get_line_y_predictions(plt)
if chart_type == 'bar':
    y_predictions = get_bar_y_predictions(plt)
if chart_type == 'hbar':
    y_predictions = get_hbar_y_predictions(plt)
if chart_type == 'pie':
    y_predictions = get_pie_y_predictions(plt)
if chart_type == 'area':
    y_predictions = get_area_y_predictions(plt)
if chart_type == 'radar':
    y_predictions = get_radar_y_predictions(plt)
if chart_type == 'scatter':
    y_predictions = get_scatter_y_predictions(plt)
if chart_type == 'waterfall':
    y_predictions = get_waterfall_y_predictions(plt)

if chart_type == 'pie':
    print(compute_pie_chart_metric(y_references, y_predictions))
else:
    print(compute_general_chart_metric(y_references, y_predictions))
"""


with open("payload.json", encoding="utf-8") as f:
    payload = json.load(f)

if payload["mode"] == "exec":
    execute(surround_pycode_with_main(payload["python_code"]))
elif payload["mode"] == "chart_eval":
    combined = "\n".join([
        "from tasks.tablebench.chart_metric_utils import *",
        "import matplotlib.pyplot as plt",
        payload["python_code"],
        payload["answer"],
        f"chart_type = {payload['chart_type']!r}",
        chart_eval_code(payload["chart_type"]),
    ])
    execute(surround_pycode_with_main(combined))
    plt.close("all")
else:
    raise ValueError(f"unknown mode: {payload['mode']}")
'''
