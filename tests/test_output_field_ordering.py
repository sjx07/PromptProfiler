"""Output field ordering should support explicit feature-authored ordinals."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_TOOL_DIR = str(Path(__file__).parent.parent.parent.parent)
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from core.func_registry import ROOT_ID, apply_config, make_func_id
from core.store import CubeStore, OnConflict


def _temp_store():
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return f.name


def _output_field(name: str, ordinal: int) -> dict:
    return {
        "node_type": "output_field",
        "parent_id": ROOT_ID,
        "payload": {
            "name": name,
            "description": f"{name} field",
            "ordinal": ordinal,
        },
    }


def test_markdown_output_field_ordinals_control_template_order():
    db = _temp_store()
    try:
        store = CubeStore(db)
        fields = [
            _output_field("step_2", 20),
            _output_field("step_1", 10),
            _output_field("step_4", 40),
            _output_field("step_3", 30),
        ]
        specs = [
            {
                "func_id": make_func_id("insert_node", params),
                "func_type": "insert_node",
                "params": params,
                "meta": {},
            }
            for params in fields
        ]
        store.upsert_funcs(specs, on_conflict=OnConflict.SKIP)

        state = apply_config([s["func_id"] for s in specs], store)
        state.format_style = "markdown"
        rendered = state.to_prompt_state()._build_system_content()
        store.close()

        positions = [
            rendered.index("### **step_1**"),
            rendered.index("### **step_2**"),
            rendered.index("### **step_3**"),
            rendered.index("### **step_4**"),
        ]
        assert positions == sorted(positions), rendered
    finally:
        os.unlink(db)
