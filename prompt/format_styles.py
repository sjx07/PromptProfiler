"""Format styles - the HOW of prompts.

Format styles define how to format semantic content (rules, fields, etc.)
in different styles: Plain, YAML, Markdown, JSON.

This is purely syntactic - no semantic content here, just formatting functions.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging

from prompt.semantic_content import Context, SemanticContent
from prompt.rules import RuleItem, RuleGroup, RuleSection, RuleTree, _get_node_id

logger = logging.getLogger(__name__)


# ── soft dependency: json_repair ─────────────────────────────────────
# `json-repair` (pip install json-repair) forgives unescaped newlines,
# trailing commas, unquoted keys, truncated JSON, etc.  When installed,
# JSONStyle uses it as the primary parser.  When absent, we fall back
# to the built-in custom repair below (_repair_unescaped_newlines_in_json).
try:
    from json_repair import loads as _json_repair_loads  # type: ignore
    _HAS_JSON_REPAIR = True
except ImportError:   # pragma: no cover
    _json_repair_loads = None
    _HAS_JSON_REPAIR = False
    logger.info(
        "json-repair not installed; falling back to built-in repair. "
        "For robust LLM-JSON parsing install: pip install json-repair"
    )

def _repair_unescaped_newlines_in_json(text: str) -> str:
    """Convert raw newlines/tabs inside JSON string values into escape sequences.

    Many LLMs (Llama, Qwen, ...) emit code inside JSON string values with
    *literal* newlines instead of ``\\n`` escapes, which makes the result
    invalid JSON. This helper walks the string, tracks whether we're inside
    a JSON string, and escapes raw whitespace that would otherwise break
    ``json.loads``.

    Idempotent: valid JSON passes through unchanged.
    """
    out: List[str] = []
    in_string = False
    prev_backslash = False
    for ch in text:
        if prev_backslash:
            out.append(ch)
            prev_backslash = False
            continue
        if ch == "\\":
            out.append(ch)
            prev_backslash = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string:
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
        out.append(ch)
    return "".join(out)


def _extract_quoted_field(text: str, field_name: str) -> Optional[str]:
    """Extract the value of ``"field_name": "..."`` where the value may contain
    unescaped newlines.  Returns the (JSON-decoded) string value or None.

    Accepts:
      "field": "single line value"
      "field": "multi
                line
                value"
    """
    # Capture everything up to the unescaped closing quote.
    pattern = rf'"{re.escape(field_name)}"\s*:\s*"((?:[^"\\]|\\.)*)"'
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        return None
    raw = m.group(1)
    # Interpret standard JSON escapes (\n, \", \\, etc.).
    try:
        return json.loads('"' + raw + '"')
    except json.JSONDecodeError:
        return raw


def fallback_parse_output(response: str, output_fields: Dict[str, str]) -> Dict[str, Any]:
    """Fallback parser that tries multiple parsing strategies.

    Tries in order:
    0. JSON with newline-repair, then direct quoted-field extraction
       (handles multi-line code strings that LLMs emit unescaped).
    1. JSON parsing - looks for JSON objects/strings in response
    2. Key-value pairs with various delimiters (: = ->)
    3. YAML-style lists (- key: value)
    4. Returns all fields with entire response if all else fails

    Args:
        response: Raw model response string
        output_fields: Expected output fields

    Returns:
        Dictionary mapping field names to extracted values
    """
    result = {}
    response_stripped = response.strip()

    # Strategy 0a: If json-repair is installed, let it handle everything
    # (unescaped newlines, trailing commas, unquoted keys, ...).
    if _HAS_JSON_REPAIR:
        try:
            parsed = _json_repair_loads(response_stripped)
            if isinstance(parsed, dict):
                for field_name in output_fields.keys():
                    if field_name in parsed:
                        result[field_name] = parsed[field_name]
                if result:
                    logger.debug("Parsed output via json-repair (fallback)")
                    return result
        except Exception:
            pass

    # Strategy 0b: Built-in newline-repair.
    try:
        repaired = _repair_unescaped_newlines_in_json(response_stripped)
        json_match = re.search(r'\{.*\}', repaired, re.DOTALL)
        if json_match:
            json_obj = json.loads(json_match.group(0))
            if isinstance(json_obj, dict):
                for field_name in output_fields.keys():
                    if field_name in json_obj:
                        result[field_name] = json_obj[field_name]
                if result:
                    logger.debug("Parsed output via built-in newline-repair")
                    return result
    except (json.JSONDecodeError, AttributeError):
        pass

    # Strategy 0c: Quoted-field extraction — handles {"OutputFields": {...}}
    # wrappers and trailing prose that the full-object path can't salvage.
    for field_name in output_fields.keys():
        val = _extract_quoted_field(response_stripped, field_name)
        if val is not None:
            result[field_name] = val
    if result:
        logger.debug("Parsed output via quoted-field extraction")
        return result

    # Strategy 1: Try to parse as JSON (original, simple object match)
    try:
        json_match = re.search(r'\{[^{}]*\}', response_stripped, re.DOTALL)
        if json_match:
            json_obj = json.loads(json_match.group(0))
            for field_name in output_fields.keys():
                if field_name in json_obj:
                    result[field_name] = json_obj[field_name]
            if result:
                logger.debug("Successfully parsed output using JSON strategy")
                return result
    except (json.JSONDecodeError, AttributeError):
        pass

    # Strategy 2: Try key-value pairs with various separators
    for field_name in output_fields.keys():
        patterns = [
            rf"['\"]?{re.escape(field_name)}['\"]?\s*:\s*['\"]?(.+?)['\"]?(?:\n|$|,)",  # field: value or "field": "value"
            rf"{re.escape(field_name)}\s*=\s*(.+?)(?:\n|$)",  # field = value
            rf"{re.escape(field_name)}\s*->\s*(.+?)(?:\n|$)",  # field -> value
            rf"\*\*{re.escape(field_name)}\*\*\s*:\s*(.+?)(?:\n|$)",  # **field**: value
            rf"-\s+{re.escape(field_name)}\s*:\s*(.+?)(?:\n|$)",  # - field: value (YAML style)
        ]

        for pattern in patterns:
            match = re.search(pattern, response_stripped, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1).strip()
                # Clean up quotes if present
                value = value.strip('"\'')
                result[field_name] = value
                break

    if result:
        logger.debug("Successfully parsed output using key-value strategy")
        return result

    # Strategy 3: If only one field, return entire response
    if len(output_fields) == 1:
        field_name = list(output_fields.keys())[0]
        # Try to parse as JSON first (the response might be the JSON value itself)
        try:
            parsed_value = json.loads(response_stripped)
            result[field_name] = parsed_value
            logger.debug("Returned parsed JSON response for single output field")
        except json.JSONDecodeError:
            result[field_name] = response_stripped
            logger.debug("Returned entire response for single output field")
        return result

    # Strategy 4: Last resort - return all fields with entire response
    logger.debug("Using fallback: returning all fields with entire response")
    for field_name in output_fields.keys():
        result[field_name] = response_stripped

    return result



def _safe_strip(s: Optional[str]) -> str:
    return (s or "").strip()


def _node_id_from_meta(meta: Dict[str, Any]) -> Optional[str]:
    if not meta:
        return None
    return meta.get("id") or meta.get("path")


def _fmt_rule_item_text(
    ri,
    *,
    include_ids: bool,
    include_kinds: bool,
) -> str:
    """
    ri: RuleItem
    """
    parts: List[str] = []
    if include_ids and getattr(ri, "rule_id", None):
        parts.append(f"[{ri.rule_id}]")
    if include_kinds and getattr(ri, "rule_kind", None):
        parts.append(f"({ri.rule_kind})")
    txt = _safe_strip(getattr(ri, "text", ""))
    parts.append(txt)
    return " ".join(p for p in parts if p).strip()


def _fmt_group_title(g, *, include_ids: bool) -> str:
    """
    g: RuleGroup
    """
    title = _safe_strip(getattr(g, "title", ""))
    if include_ids and getattr(g, "group_id", None):
        return f"{title} [{g.group_id}]".strip()
    return title


def _fmt_section_title(sec, *, include_ids: bool) -> str:
    """
    sec: RuleSection
    """
    title = _safe_strip(getattr(sec, "title", "")) or "Untitled Section"
    if include_ids:
        sid = _node_id_from_meta(getattr(sec, "metadata", {}) or {})
        if sid:
            return f"{title} [{sid}]"
    return title


def _partition_rules_by_kind(rule_items) -> Dict[str, List[Any]]:
    """
    Return buckets: {"number": [...], "other": [...]}
    - 'number' means procedural ordering
    - everything else treated as unordered in text styles
    """
    numbered = []
    other = []
    for ri in rule_items or []:
        if getattr(ri, "rule_kind", None) == "number":
            numbered.append(ri)
        else:
            other.append(ri)
    return {"number": numbered, "other": other}

def _node_kind(node) -> str:
    """
    Return rendering kind for RuleItem/RuleGroup.
    priority:
      1) node.rule_kind if RuleItem
      2) node.metadata["rule_kind"] if RuleGroup
      3) default "bullet"
    """
    if isinstance(node, RuleItem):
        return (getattr(node, "rule_kind", None) or "bullet").strip()
    if isinstance(node, RuleGroup):
        meta = getattr(node, "metadata", {}) or {}
        return (meta.get("rule_kind") or "bullet").strip()
    return "bullet"


def _emit_prefix(kind: str, idx: int | None = None) -> str:
    """prefix token for a list item."""
    if kind == "number":
        return f"{idx}."
    # treat everything else as bullet-like in text renderers
    return "-"

class FormatStyle(ABC):
    """Abstract base for format styles.

    Each format style knows how to format semantic content
    (rules, fields, examples) in its specific style.
    """

    @abstractmethod
    def format_system_message(self, semantic: "SemanticContent") -> str:
        """Format complete system message from semantic content.

        This is the main entry point - takes all semantic content
        and formats it according to this style.
        """
        pass
    
    @abstractmethod
    def format_user_message(self, record: Dict[str, Any], semantic: "SemanticContent") -> str:
        """Format user message for a given record.

        By default, just returns the input fields formatted.
        Subclasses can override for more complex behavior.
        """
        pass

    @abstractmethod
    def format_rule_sections(self, sections: List[RuleSection], tree: "RuleTree") -> str:
        """Format hierarchical rule sections."""
        pass

    @abstractmethod
    def format_field_descriptions(self, fields: Dict[str, str], label: str) -> str:
        """Format field descriptions (e.g., input fields, output fields)."""
        pass

    @abstractmethod
    def format_structure_template(self, input_fields: Dict[str, str], output_fields: Dict[str, str]) -> str:
        """Format the structure template showing expected I/O format."""
        pass

    @abstractmethod
    def render_output(self, output: Dict[str, Any]) -> str:
        """Render output values in the format the LLM is expected to produce."""
        pass

    @abstractmethod
    def format_contexts(self, contexts: List[Context]) -> str:
        """Format context blocks."""
        pass

    @abstractmethod
    def get_field_delimiter(self) -> str:
        """Get field delimiter for this format."""
        pass

    @abstractmethod
    def get_section_delimiter(self) -> str:
        """Get section delimiter for this format."""
        pass

    @abstractmethod
    def parse_output(self, response: str, output_fields: Dict[str, str]) -> Dict[str, Any]:
        """Parse model response to extract output field values.

        Args:
            response: Raw model response string
            output_fields: Expected output fields and their descriptions

        Returns:
            Dictionary mapping field names to extracted values
        """
        pass

    def build_messages(self, record: Dict[str, Any], semantic: "SemanticContent") -> List[Dict[str, str]]:
        """Build messages in plain text style with System:/User: labels."""
        # Plain style: combine system and user in a single message with labels
        system_content = self.format_system_message(semantic)
        user_content = self.format_user_message(record, semantic)

        combined_content = f"System:\n{system_content}\n\nUser:\n{user_content}"

        return [{"role": "user", "content": combined_content}]

class PlainStyle(FormatStyle):
    """Plain text formatting - natural language paragraphs."""

    def format_system_message(self, semantic: "SemanticContent") -> str:
        """Format complete system message in plain text style."""
        parts = []

        # Instruction
        if semantic.instruction:
            parts.append(f"Instruction:\n{semantic.instruction.text}")

        # Field descriptions
        if semantic.input_fields:
            parts.append(self.format_field_descriptions(semantic.input_fields, "Input Fields"))
        if semantic.output_fields:
            parts.append(self.format_field_descriptions(semantic.output_fields, "Output Fields"))

        # Rules — each section rendered as a sibling block
        if semantic.rule_sections:
            rules_text = self.format_rule_sections(semantic.rule_sections, semantic.tree)
            if rules_text.strip():
                parts.append(rules_text)

        # Contexts
        if semantic.contexts:
            parts.append(self.format_contexts(semantic.contexts))

        # Structure template with output instructions
        if semantic.output_fields:
            structure = self.format_structure_template(semantic.input_fields, semantic.output_fields)
            parts.append(structure)

        # # Final instruction
        # instruction_text = semantic.instruction.text if semantic.instruction else "complete the task"
        # parts.append(f'Instruction: {instruction_text}')

        return self.get_section_delimiter().join(parts)

    def format_user_message(self, record: Dict[str, Any], semantic: "SemanticContent") -> str:
        """Format user message for a given record in plain text."""
        blocks = []
        for field_name in semantic.input_fields.keys():
            value = record.get(field_name, "Not provided")
            blocks.append(f"{field_name}: {value}\n")
        return self.get_field_delimiter().join(blocks)
            

    def format_rule_sections(self, sections: List["RuleSection"], tree: "RuleTree") -> str:
        return self._format_rule_sections_impl(sections, level=1, tree=tree)

    def _format_rule_sections_impl(
        self,
        sections: List["RuleSection"],
        tree: "RuleTree",
        *,
        level: int,
    ) -> str:
        lines: List[str] = []

        def fmt_section_title(sec: "RuleSection", lvl: int) -> str:
            title = (getattr(sec, "title", "") or "").strip() or "Untitled Section"
            return f"{title}"
        
        def render_rule_nodes(nodes: List[Any], indent: str) -> List[str]:
            # number needs local counter per block
            out: List[str] = []
            num = 1
            for node in nodes:
                kind = _node_kind(node)
                if isinstance(node, RuleItem):
                    if not tree.is_enabled(_get_node_id(node) or ""):
                        continue
                    txt = _safe_strip(getattr(node, "text", ""))
                    if not txt:
                        continue
                    if kind == "number":
                        out.append(f"{indent}{_emit_prefix('number', num)} {txt}")
                        num += 1
                    else:
                        out.append(f"{indent}{_emit_prefix(kind)} {txt}")

                elif isinstance(node, RuleGroup):
                    if node.group_id and not tree.is_enabled(node.group_id):
                        continue
                    title = _safe_strip(getattr(node, "title", "")) or "Untitled Group"
                    # group itself is also an item in the same environment
                    if kind == "number":
                        out.append(f"{indent}{_emit_prefix('number', num)} {title}")
                        num += 1
                    else:
                        out.append(f"{indent}{_emit_prefix(kind)} {title}")

                    # children indented one step
                    child_rules = node.children
                    if child_rules:
                        out.extend(render_rule_nodes(child_rules, indent + "  "))
            return out

        for sec in sections or []:
            sid = sec.metadata.get("id") or sec.metadata.get("path") if sec.metadata else None
            if sid and not tree.is_enabled(sid):
                continue

            content = (getattr(sec, "content", "") or "").strip()
            subs = [c for c in sec.children if isinstance(c, RuleSection)]
            sub_block = self._format_rule_sections_impl(subs, level=level + 1, tree=tree) if subs else ""
            rule_lines = render_rule_nodes(list(sec.iter_rule_children() or []), indent="  ")

            if not rule_lines and not content and not sub_block.strip():
                continue

            lines.append(fmt_section_title(sec, level))
            if content:
                lines.append(f"  {content}")
            if sub_block.strip():
                lines.append(sub_block)
            if rule_lines:
                lines.extend(rule_lines)
                
            lines.append("\n")

        return "\n".join(lines)

    def format_field_descriptions(self, fields: Dict[str, str], label: str) -> str:
        """Format as natural language list."""
        field_parts = []
        for name, desc in fields.items():
            field_parts.append(f"{name} ({desc})")
        return f"{label}: " + ", ".join(field_parts) + "."

    def format_structure_template(self, input_fields: Dict[str, str], output_fields: Dict[str, str]) -> str:
        """Format as plain field: value pairs with explicit output instructions."""
        lines = ["Output Format Instructions:"]
        lines.append("You must respond with ONLY the output fields in the following format:")
        lines.append("")

        # Show output format template
        for name in output_fields.keys():
            lines.append(f"{name}: <your_answer_here>")

        # lines.append("")
        # lines.append("Important: Provide only the output fields above, nothing else.")
        return "\n".join(lines)

    def render_output(self, output: Dict[str, Any]) -> str:
        return "\n".join(f"{k}: {v}" for k, v in output.items())

    def format_contexts(self, contexts: List[Context]) -> str:
        """Format contexts inline."""
        return "\n\n".join(ctx.content for ctx in contexts)

    def get_field_delimiter(self) -> str:
        return "\n"

    def get_section_delimiter(self) -> str:
        return "\n\n"

    def parse_output(self, response: str, output_fields: Dict[str, str]) -> Dict[str, Any]:
        """Parse plain text response where values extend until the next field."""
        result = {}
        response_stripped = response.strip()
        field_names = list(output_fields.keys())

        if not field_names:
            return result

        # Build pattern to match any expected field name followed by colon
        field_pattern = '|'.join(re.escape(f) for f in field_names)
        delimiter_pattern = rf'(?:^|\n)\s*(?:-\s+)?(?:\*\*)?({field_pattern})(?:\*\*)?\s*:\s*'

        matches = list(re.finditer(delimiter_pattern, response_stripped, re.IGNORECASE))

        if not matches:
            return fallback_parse_output(response, output_fields)

        # Extract value for each field until next field or end
        for i, match in enumerate(matches):
            field_name_matched = match.group(1)
            actual_field_name = next((f for f in field_names if f.lower() == field_name_matched.lower()), None)

            if not actual_field_name:
                continue

            value_start = match.end()
            value_end = matches[i + 1].start() if i + 1 < len(matches) else len(response_stripped)

            value = response_stripped[value_start:value_end].strip().rstrip('\n').strip('"\'')
            if value:
                result[actual_field_name] = value

        return result if result else fallback_parse_output(response, output_fields)



class YAMLStyle(FormatStyle):
    """YAML formatting - hierarchical with indentation."""

    def format_system_message(self, semantic: "SemanticContent") -> str:
        """Format complete system message in YAML style."""
        lines = []

        # Instruction
        if semantic.instruction:
            lines.append("Task:")
            lines.append(f"- {semantic.instruction.text}")
            lines.append("")

        # Field descriptions
        if semantic.input_fields:
            lines.append(self.format_field_descriptions(semantic.input_fields, "Input_Fields"))
            lines.append("")
        if semantic.output_fields:
            lines.append(self.format_field_descriptions(semantic.output_fields, "Output_Fields"))
            lines.append("")

        # Rules — each section rendered as a sibling block
        if semantic.rule_sections:
            rules_text = self.format_rule_sections(semantic.rule_sections, semantic.tree)
            lines.append(rules_text)
            lines.append("")

        # Contexts
        if semantic.contexts:
            contexts_text = self.format_contexts(semantic.contexts)
            lines.append(contexts_text)
            lines.append("")

        # Structure template with output instructions
        if semantic.output_fields:
            structure = self.format_structure_template(semantic.input_fields, semantic.output_fields)
            lines.append(structure)
            lines.append("")

        return "\n".join(lines)
    
    def format_user_message(self, record: Dict[str, Any], semantic: SemanticContent) -> str:
        """Format user message for a given record in YAML style with dashes."""
        blocks = []
        for field_name in semantic.input_fields.keys():
            value = record.get(field_name, "Not provided")
            blocks.append(f"- {field_name}: {value}\n")
        return self.get_field_delimiter().join(blocks)

    def format_rule_sections(self, sections: List["RuleSection"], tree: "RuleTree") -> str:
        return self._format_rule_sections_impl(sections, indent=0, tree=tree)


    def _format_rule_sections_impl(
        self,
        sections: List["RuleSection"],
        tree: "RuleTree",
        *,
        indent: int,
    ) -> str:
        def pad(n: int) -> str:
            return "  " * n

        def clean(s: str) -> str:
            return (s or "").strip()

        def emit_rule_item(ri: "RuleItem", *, base_indent: int, num_state: Dict[str, int]) -> List[str]:
            if not tree.is_enabled(_get_node_id(ri) or ""):
                return []
            txt = clean(getattr(ri, "text", ""))
            if not txt:
                return []
            kind = getattr(ri, "rule_kind", None)
            if kind == "number":
                k = num_state["k"]
                num_state["k"] += 1
                return [f"{pad(base_indent)}- {k}. {txt}"]
            return [f"{pad(base_indent)}- {txt}"]

        def emit_group(g: "RuleGroup", *, base_indent: int) -> List[str]:
            if g.group_id and not tree.is_enabled(g.group_id):
                return []
            title = clean(getattr(g, "title", "")) or "Untitled Group"
            out = [f"{pad(base_indent)}- {title}:"]  
            num_state = {"k": 1}
            for ri in (g.children):
                out.extend(emit_rule_item(ri, base_indent=base_indent + 2, num_state=num_state))
            return out

        def emit_items_in_order(sec: "RuleSection", *, base_indent: int) -> List[str]:
            out: List[str] = []
            num_state = {"k": 1}  
            for node in (sec.iter_rule_children() or []):
                if isinstance(node, RuleItem):
                    out.extend(emit_rule_item(node, base_indent=base_indent, num_state=num_state))
                elif isinstance(node, RuleGroup):
                    out.extend(emit_group(node, base_indent=base_indent))
            return out

        def render_section(sec: "RuleSection", *, base_indent: int) -> List[str]:
            out: List[str] = []
            title = clean(getattr(sec, "title", "")) or "Untitled Section"
            content = clean(getattr(sec, "content", ""))
            subs = list([c for c in sec.children if isinstance(c, RuleSection)])

            out.append(f"{pad(base_indent)}{title}:")

            items = emit_items_in_order(sec, base_indent=base_indent + 1)

            if content:
                out.append(f"{pad(base_indent + 1)}- {content}")

            if items:
                out.extend(items)

            if subs:
                out.append(f"{pad(base_indent + 1)}Subsections:")
                for s in subs:
                    out.extend(render_section(s, base_indent=base_indent + 2))

            return out

        lines: List[str] = []
        for sec in (sections or []):
            sid = sec.metadata.get("id") or sec.metadata.get("path") if sec.metadata else None
            if sid and not tree.is_enabled(sid):
                continue
            lines.extend(render_section(sec, base_indent=indent))
            lines.append("\n")

        return "\n".join(lines)

    def format_field_descriptions(self, fields: Dict[str, str], label: str) -> str:
        """Format as YAML fields with dashes."""
        title = label.replace(' ', '_')
        field_parts = [f"{title}:"]
        for name, desc in fields.items():
            field_parts.append(f"- {name}: {desc}")
        return "\n".join(field_parts)

    def format_structure_template(self, input_fields: Dict[str, str], output_fields: Dict[str, str]) -> str:
        """Format as YAML structure with explicit output instructions."""
        lines = ["Output_Format_Instructions:"]
        lines.append("- You must respond in YAML format with ONLY the output fields")
        lines.append("- Use the exact structure shown below:")
        lines.append("")

        # Show output format template
        for name in output_fields.keys():
            lines.append(f"- {name}: <your_answer_here>")

        lines.append("")
        lines.append("Important:")
        lines.append("- Follow YAML syntax exactly")
        lines.append("- Provide only the output fields listed above")
        return "\n".join(lines)

    def render_output(self, output: Dict[str, Any]) -> str:
        return "\n".join(f"{k}: {v}" for k, v in output.items())

    def format_contexts(self, contexts: List[Context]) -> str:
        """Format contexts as YAML sections."""
        lines = []
        for ctx in contexts:
            lines.append(f"{ctx.name}:")
            for line in ctx.content.split('\n'):
                if line.strip():
                    lines.append(f"  {line}")
            lines.append("")
        return "\n".join(lines)

    def get_field_delimiter(self) -> str:
        return "\n"

    def get_section_delimiter(self) -> str:
        return "\n\n"

    def build_messages(self, record: Dict[str, Any], semantic: "SemanticContent") -> List[Dict[str, str]]:
        """Build messages in YAML style with structured formatting."""
        system_content = self.format_system_message(semantic)
        user_content = self.format_user_message(record, semantic)

        # YAML style: combine system and user with YAML-like structure
        combined_content = f"System:\n{system_content}\n\nUser:\n{user_content}"

        return [{"role": "user", "content": combined_content}]

    def parse_output(self, response: str, output_fields: Dict[str, str]) -> Dict[str, Any]:
        """Parse YAML-style response where values extend until the next field."""
        result = {}
        response_stripped = response.strip()
        field_names = list(output_fields.keys())

        if not field_names:
            return result

        # Build pattern to match any expected field name followed by colon
        field_pattern = '|'.join(re.escape(f) for f in field_names)
        delimiter_pattern = rf'(?:^|\n)\s*(?:-\s+)?(?:\*\*)?({field_pattern})(?:\*\*)?\s*:\s*'

        matches = list(re.finditer(delimiter_pattern, response_stripped, re.IGNORECASE))

        if not matches:
            return fallback_parse_output(response, output_fields)

        # Extract value for each field until next field or end
        for i, match in enumerate(matches):
            field_name_matched = match.group(1)
            actual_field_name = next((f for f in field_names if f.lower() == field_name_matched.lower()), None)

            if not actual_field_name:
                continue

            value_start = match.end()
            value_end = matches[i + 1].start() if i + 1 < len(matches) else len(response_stripped)

            value = response_stripped[value_start:value_end].strip().rstrip('\n').strip('"\'')
            if value:
                result[actual_field_name] = value

        return result if result else fallback_parse_output(response, output_fields)


class MarkdownStyle(FormatStyle):
    """Markdown formatting - headers and lists."""

    def format_system_message(self, semantic: "SemanticContent") -> str:
        """Format complete system message in Markdown style."""
        lines = []

        # Instruction
        if semantic.instruction:
            lines.append("## Task")
            lines.append(f"- {semantic.instruction.text}")
            lines.append("")

        # Field descriptions
        if semantic.input_fields:
            lines.append(self.format_field_descriptions(semantic.input_fields, "Input Fields"))
            lines.append("")
        if semantic.output_fields:
            lines.append(self.format_field_descriptions(semantic.output_fields, "Output Fields"))
            lines.append("")

        # Rules — each section rendered as a sibling block
        if semantic.rule_sections:
            lines.append(self.format_rule_sections(semantic.rule_sections, semantic.tree))
            lines.append("")

        # Contexts
        if semantic.contexts:
            lines.append(self.format_contexts(semantic.contexts))
            lines.append("")

        # Structure template with output instructions
        if semantic.output_fields:
            structure = self.format_structure_template(semantic.input_fields, semantic.output_fields)
            lines.append(structure)
            lines.append("")

        return "\n".join(lines)

    def format_rule_sections(self, sections: List["RuleSection"], tree: "RuleTree") -> str:
        return self._format_rule_sections_impl(sections, level=1, tree=tree)

    def _format_rule_sections_impl(
        self,
        sections: List["RuleSection"],
        tree: "RuleTree",
        *,
        level: int,
    ) -> str:
        def fmt_section_title(sec: "RuleSection", lvl: int) -> str:
            title = _safe_strip(getattr(sec, "title", "")) or "Untitled Section"
            return f"{'#' * (lvl + 1)} {title}"

        def render_rule_nodes(nodes: List[Any], indent: str) -> List[str]:
            out: List[str] = []
            num = 1
            for node in nodes:
                kind = _node_kind(node)

                if isinstance(node, RuleItem):
                    if not tree.is_enabled(_get_node_id(node) or ""):
                        continue
                    txt = _safe_strip(getattr(node, "text", ""))
                    if not txt:
                        continue
                    if kind == "number":
                        out.append(f"{indent}{num}. {txt}")
                        num += 1
                    else:
                        out.append(f"{indent}- {txt}")

                elif isinstance(node, RuleGroup):
                    if node.group_id and not tree.is_enabled(node.group_id):
                        continue
                    title = _safe_strip(getattr(node, "title", "")) or "Untitled Group"
                    if kind == "number":
                        out.append(f"{indent}{num}. {title}")
                        num += 1
                    else:
                        out.append(f"{indent}- {title}")

                    child_rules = node.children
                    if child_rules:
                        out.extend(render_rule_nodes(child_rules, indent + "  "))
            return out

        def render_section(sec: "RuleSection", lvl: int) -> List[str]:
            content = _safe_strip(getattr(sec, "content", ""))
            subs = [c for c in sec.children if isinstance(c, RuleSection)]
            nodes = list(sec.iter_rule_children() or [])

            rule_lines = render_rule_nodes(nodes, indent="")
            sub_blocks = [
                block
                for sub in subs
                if (block := render_section(sub, lvl + 1))
            ]

            if not rule_lines and not content and not sub_blocks:
                return []

            out = [fmt_section_title(sec, lvl)]
            if content:
                out.append(content)
            if rule_lines:
                out.extend(rule_lines)

            for block in sub_blocks:
                out.append("")
                out.extend(block)
            return out

        lines: List[str] = []
        for sec in sections or []:
            block = render_section(sec, level)
            if not block:
                continue
            lines.extend(block)
            lines.append("")  # spacing

        return "\n".join(lines).rstrip()

    def format_field_descriptions(self, fields: Dict[str, str], label: str) -> str:
        """Format as Markdown list."""
        title = f"## {label}"
        field_parts = [title]
        for name, desc in fields.items():
            field_parts.append(f"- **{name}**: {desc}")
        return "\n".join(field_parts)

    def format_structure_template(self, input_fields: Dict[str, str], output_fields: Dict[str, str]) -> str:
        """Format as Markdown structure with explicit output instructions."""
        lines = ["## Output Format Instructions"]
        lines.append("")
        lines.append("You must respond in Markdown format with ONLY the output fields.")
        lines.append("")
        lines.append("Use the exact structure shown below:")
        lines.append("")

        # Show output format template
        for name in output_fields.keys():
            lines.append(f"### **{name}**")
            lines.append("<your_answer_here>")
            lines.append("")

        lines.append("**Important:**")
        lines.append("- Use Markdown headers (###) for each field")
        lines.append("- Provide only the output fields listed above")
        lines.append("- Do not include any other text or explanations")
        return "\n".join(lines).rstrip()

    def render_output(self, output: Dict[str, Any]) -> str:
        return "\n".join(f"**{k}**: {v}" for k, v in output.items())

    def format_contexts(self, contexts: List[Context]) -> str:
        """Format contexts as Markdown sections."""
        lines = []
        for ctx in contexts:
            lines.append(f"## {ctx.name}")
            lines.append(ctx.content)
            lines.append("")
        return "\n".join(lines)

    def format_user_message(self, record: Dict[str, Any], semantic: "SemanticContent") -> str:
        """Format user message for a given record in Markdown style."""
        blocks = []
        for field_name in semantic.input_fields.keys():
            value = record.get(field_name, "Not provided")
            blocks.append(f"### **{field_name}**\n")
            blocks.append(f"{value}\n")
        return self.get_field_delimiter().join(blocks)

    def get_field_delimiter(self) -> str:
        return "\n\n"

    def get_section_delimiter(self) -> str:
        return "\n\n"

    def build_messages(self, record: Dict[str, Any], semantic: "SemanticContent") -> List[Dict[str, str]]:
        """Build messages in Markdown style with headers and structure."""
        system_content = self.format_system_message(semantic)
        user_content = self.format_user_message(record, semantic)

        # Markdown style: combine system and user with Markdown headers
        combined_content = f"# System \n\n{system_content}\n\n# User \n\n{user_content}"

        return [{"role": "user", "content": combined_content}]

    def parse_output(self, response: str, output_fields: Dict[str, str]) -> Dict[str, Any]:
        """Parse Markdown-style response to extract output field values.

        Expects format like:
        ### **field_name**
        value

        or

        ### field_name
        value
        """
        result = {}
        response_stripped = response.strip()

        # Try to find Markdown headers followed by content (multi-line)
        for field_name in output_fields.keys():
            # Patterns for multi-line format (header on one line, value on next)
            # Fixed: Lookahead now properly handles end-of-string without requiring \n before $
            patterns = [
                # ### **field**\nvalue - match until next markdown element or end
                rf"###\s+\*\*{re.escape(field_name)}\*\*\s*\n+(.+?)(?=\n+(?:###|##|\*\*)|$)",
                # ### field\nvalue
                rf"###\s+{re.escape(field_name)}\s*\n+(.+?)(?=\n+(?:###|##|\*\*)|$)",
                # ## field\nvalue
                rf"##\s+{re.escape(field_name)}\s*\n+(.+?)(?=\n+(?:###|##|\*\*)|$)",
                # **field**\nvalue
                rf"\*\*{re.escape(field_name)}\*\*\s*\n+(.+?)(?=\n+(?:###|##|\*\*)|$)",
                # Single-line patterns as fallback
                rf"\*\*{re.escape(field_name)}\*\*\s*:\s*(.+?)(?:\n|$)",  # **field**: value
                rf"###\s+{re.escape(field_name)}\s*:\s*(.+?)(?:\n|$)",  # ### field: value
                rf"{re.escape(field_name)}\s*:\s*(.+?)(?:\n|$)",  # field: value
            ]

            for pattern in patterns:
                match = re.search(pattern, response_stripped, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                if match:
                    value = match.group(1).strip()
                    # Clean up extra whitespace and newlines
                    value = ' '.join(value.split())
                    result[field_name] = value
                    break

        # If found some fields, return them
        if result:
            logger.debug("Successfully parsed output using Markdown patterns")
            return result

        # Always use fallback if nothing found
        logger.debug("Markdown parsing failed, using fallback parser")
        return fallback_parse_output(response, output_fields)


class JSONStyle(FormatStyle):
    """JSON formatting - structured objects."""

    def format_system_message(self, semantic: "SemanticContent") -> str:
        """Format complete system message in JSON style."""
        obj = {}

        # Instruction
        if semantic.instruction:
            obj["Task"] = semantic.instruction.text

        # Field descriptions
        if semantic.input_fields:
            obj["InputFields"] = semantic.input_fields
        if semantic.output_fields:
            obj["OutputFields"] = semantic.output_fields

        # Rules — merge each section as a sibling key, skip empty sections
        if semantic.rule_sections:
            rules_dict = json.loads(self.format_rule_sections(semantic.rule_sections, semantic.tree))
            obj.update({k: v for k, v in rules_dict.items() if v})

        # Contexts
        if semantic.contexts:
            obj["Contexts"] = {ctx.name: ctx.content for ctx in semantic.contexts}

        # Structure template with output instructions
        if semantic.output_fields:
            # Parse the structure template JSON and add it to the main object
            structure_json = self.format_structure_template(semantic.input_fields, semantic.output_fields)
            structure_obj = json.loads(structure_json)
            obj.update(structure_obj)

        return json.dumps(obj, indent=2)

    def format_rule_sections(self, sections: List["RuleSection"], tree: "RuleTree") -> str:
        return self._format_rule_sections_impl(sections, level=1, tree=tree)

    def _format_rule_sections_impl(
        self,
        sections: List["RuleSection"],
        tree: "RuleTree",
        *,
        level: int,
    ) -> str:
        def clean(x: Any) -> str:
            return (x or "").strip()

        def rule_item_text(ri: "RuleItem") -> str:
            if not tree.is_enabled(_get_node_id(ri) or ""):
                return ""
            return clean(getattr(ri, "text", ""))

        def group_item_obj(g: "RuleGroup") -> Dict[str, Any]:
            if g.group_id and not tree.is_enabled(g.group_id):
                return {}
            title = clean(getattr(g, "title", ""))
            items: List[Any] = []
            for node in g.children:
                if isinstance(node, RuleItem):
                    txt = rule_item_text(node)
                    if txt:
                        items.append(txt)
                elif isinstance(node, RuleGroup):
                    nested = group_item_obj(node)
                    if nested:
                        items.append(nested)
            if items:
                return {title: items}
            return {}

        def section_obj(sec: "RuleSection") -> Dict[str, Any]:
            obj: Dict[str, Any] = {"title": clean(getattr(sec, "title", ""))}
            content = clean(getattr(sec, "content", ""))
            subs = [c for c in sec.children if isinstance(c, RuleSection)]

            items: List[Any] = []
            for node in (sec.iter_rule_children() or []):
                if isinstance(node, RuleItem):
                    txt = rule_item_text(node)
                    if txt:
                        items.append(txt)
                elif isinstance(node, RuleGroup):
                    gobj = group_item_obj(node)
                    if gobj:
                        items.append(gobj)

            if content:
                obj["content"] = content

            if items and subs:
                obj["items"] = items
                obj["subsections"] = [section_obj(s) for s in subs]
            elif items:
                obj["items"] = items
            elif subs:
                obj["subsections"] = [section_obj(s) for s in subs]

            return obj

        rendered: Dict[str, Any] = {}
        for s in (sections or []):
            sid = s.metadata.get("id") or s.metadata.get("path") if s.metadata else None
            if sid and not tree.is_enabled(sid):
                continue
            sobj = section_obj(s)
            title = sobj.pop("title", "Untitled Section")
            # Merge all remaining keys into the section value
            if len(sobj) == 1:
                # Single key (items / content / subsections) — unwrap
                rendered[title] = next(iter(sobj.values()))
            elif sobj:
                # Multiple keys (items + subsections) — keep as dict
                rendered[title] = sobj
            else:
                rendered[title] = []
        return json.dumps(rendered, ensure_ascii=False, indent=2)
    
    def format_field_descriptions(self, fields: Dict[str, str], label: str) -> str:
        """Format as JSON object."""
        return json.dumps({label: fields}, indent=2)

    def format_structure_template(self, input_fields: Dict[str, str], output_fields: Dict[str, str]) -> str:
        """Format as JSON object with explicit output instructions."""
        instructions = {
            "OutputFormatInstructions": "You must respond with ONLY a valid JSON object containing the output fields",
            "FormatExample": "Use the exact JSON structure shown below",
            "OutputTemplate": {name: "<your_answer_here>" for name in output_fields.keys()},
            "Important": [
                "Return ONLY valid JSON",
                "Include only the output fields specified",
                "Do not include explanations or additional text outside the JSON object"
            ]
        }
        return json.dumps(instructions, indent=2)

    def render_output(self, output: Dict[str, Any]) -> str:
        return json.dumps(output, indent=2)

    def format_contexts(self, contexts: List[Context]) -> str:
        """Format contexts as JSON object."""
        result = {ctx.name: ctx.content for ctx in contexts}
        return json.dumps(result, indent=2)

    def format_user_message(self, record: Dict[str, Any], semantic: "SemanticContent") -> str:
        """Format user message for a given record in JSON style."""
        obj = {}
        for field_name in semantic.input_fields.keys():
            obj[field_name] = record.get(field_name, "Not provided")
        return json.dumps(obj, indent=2)

    def get_field_delimiter(self) -> str:
        return ",\n"

    def get_section_delimiter(self) -> str:
        return ",\n"

    def build_messages(self, record: Dict[str, Any], semantic: "SemanticContent") -> List[Dict[str, str]]:
        """Build messages in JSON style with structured JSON format."""
        system_content = self.format_system_message(semantic)
        user_content = self.format_user_message(record, semantic)

        # JSON style: wrap system and user in a JSON structure
        # Note: system_content and user_content are already JSON strings
        combined_obj = {
            "System": json.loads(system_content),
            "User": json.loads(user_content)
        }
        combined_content = json.dumps(combined_obj, indent=2)

        return [{"role": "user", "content": combined_content}]

    def parse_output(self, response: str, output_fields: Dict[str, str]) -> Dict[str, Any]:
        """Parse JSON-style response to extract output field values.

        Expects JSON object format:
        {
          "field_name": "value",
          "another_field": "value"
        }
        """
        response_stripped = response.strip()

        def _pick(parsed: Any) -> Optional[Dict[str, Any]]:
            """From a parsed JSON value, pick out requested output_fields."""
            if not isinstance(parsed, dict):
                return None
            hit = {k: parsed[k] for k in output_fields if k in parsed}
            return hit or None

        # 1. Strict json.loads on the raw response (fast path for valid JSON).
        try:
            hit = _pick(json.loads(response_stripped))
            if hit:
                return hit
        except json.JSONDecodeError:
            pass

        # 2. Delegate to json-repair when available; it handles unescaped
        #    newlines, trailing commas, unquoted keys, truncated JSON, etc.
        if _HAS_JSON_REPAIR:
            try:
                hit = _pick(_json_repair_loads(response_stripped))
                if hit:
                    logger.debug("Parsed output via json-repair")
                    return hit
            except Exception:   # json-repair rarely raises, but be defensive
                pass

        # 3. Built-in fallback: escape raw newlines inside strings, then retry.
        repaired = _repair_unescaped_newlines_in_json(response_stripped)
        try:
            hit = _pick(json.loads(repaired))
            if hit:
                logger.debug("Parsed output after built-in newline-repair")
                return hit
        except json.JSONDecodeError:
            pass

        # 4. Fenced/embedded JSON block extraction.
        json_patterns = [
            r'```json\s*\n(.*?)\n```',  # ```json ... ```
            r'```\s*\n(\{.*?\})\n```',  # ``` {...} ```
            r'(\{[^{}]*\})',            # simple {...} (no nested braces)
            r'(\{.*\})',                # greedy
        ]
        for pattern in json_patterns:
            match = re.search(pattern, response_stripped, re.DOTALL)
            if not match:
                continue
            inner = match.group(1).strip()
            for payload in (inner, _repair_unescaped_newlines_in_json(inner)):
                try:
                    hit = _pick(json.loads(payload))
                    if hit:
                        logger.debug("Parsed embedded JSON via pattern=%r", pattern)
                        return hit
                except json.JSONDecodeError:
                    pass
                if _HAS_JSON_REPAIR:
                    try:
                        hit = _pick(_json_repair_loads(payload))
                        if hit:
                            return hit
                    except Exception:
                        pass

        # 5. Final fallback — multi-strategy key/value extractor.
        return fallback_parse_output(response, output_fields)


class CodeBlockStyle(FormatStyle):
    """Fenced markdown code block — designed for single-field code/sql output.

    Instructs the model to reply with exactly one fenced block like::

        ```python
        # ... code here ...
        ```

    The parser extracts the LAST fenced block in the response (so any
    preamble or <think> prose is ignored) and assigns its inner text to
    the single declared output field.

    Contract
    ────────
    * Exactly one output_field must be declared (``code`` or ``sql`` in
      practice). Declaring two raises at parse time — there's no natural
      way to attribute the single block to multiple fields.
    * No JSON escape handling needed; newlines / quotes / backslashes inside
      code pass through verbatim. This is the point.
    * If the model omits the fence, the entire response is treated as the
      code block (charitable fallback — lets smaller models that forget
      the fence still score).
    """

    _LANG_HINTS = {"code": "python", "sql": "sql"}

    def _language_hint(self, output_fields: Dict[str, str]) -> str:
        if not output_fields:
            return ""
        field = next(iter(output_fields))
        return self._LANG_HINTS.get(field, "")

    def format_system_message(self, semantic: "SemanticContent") -> str:
        parts: List[str] = []
        if semantic.instruction:
            parts.append(_safe_strip(getattr(semantic.instruction, "text", "")))
        if semantic.rule_sections:
            parts.append(self.format_rule_sections(semantic.rule_sections, semantic.tree))
        if semantic.input_fields:
            parts.append(self.format_field_descriptions(semantic.input_fields, "Input Fields"))
        parts.append(self.format_structure_template(
            semantic.input_fields or {}, semantic.output_fields or {},
        ))
        return "\n\n".join(p for p in parts if p).strip()

    def format_user_message(self, record: Dict[str, Any], semantic: "SemanticContent") -> str:
        lines: List[str] = []
        for name in (semantic.input_fields or {}).keys():
            if name in record:
                lines.append(f"{name}:\n{record[name]}")
        return "\n\n".join(lines).strip()

    def format_rule_sections(self, sections: List[RuleSection], tree: "RuleTree") -> str:
        out: List[str] = []
        for s in sections:
            sid = _node_id_from_meta(getattr(s, "metadata", {}) or {}) or getattr(s, "node_id", None)
            if sid and not tree.is_enabled(sid):
                continue

            title = _safe_strip(getattr(s, "title", ""))
            content = _safe_strip(getattr(s, "content", ""))
            rule_lines: List[str] = []
            for child in getattr(s, "children", []) or []:
                if isinstance(child, RuleItem):
                    cid = _get_node_id(child)
                    if cid and not tree.is_enabled(cid):
                        continue
                    txt = _safe_strip(getattr(child, "text", ""))
                    if txt:
                        rule_lines.append(f"- {txt}")
                elif isinstance(child, RuleGroup):
                    gid = _get_node_id(child)
                    if gid and not tree.is_enabled(gid):
                        continue
                    group_title = _safe_strip(getattr(child, "title", ""))
                    if group_title:
                        rule_lines.append(f"- {group_title}")
                    for group_child in getattr(child, "children", []) or []:
                        if not isinstance(group_child, RuleItem):
                            continue
                        cid = _get_node_id(group_child)
                        if cid and not tree.is_enabled(cid):
                            continue
                        txt = _safe_strip(getattr(group_child, "text", ""))
                        if txt:
                            rule_lines.append(f"  - {txt}")

            if not content and not rule_lines:
                continue

            if title:
                out.append(f"## {title}")
            if content:
                out.append(content)
            out.extend(rule_lines)
            out.append("")
        return "\n".join(out).strip()

    def format_field_descriptions(self, fields: Dict[str, str], label: str) -> str:
        lines = [f"## {label}"]
        for name, desc in fields.items():
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def format_structure_template(
        self, input_fields: Dict[str, str], output_fields: Dict[str, str],
    ) -> str:
        if len(output_fields) != 1:
            raise ValueError(
                f"CodeBlockStyle requires exactly one output_field; got "
                f"{sorted(output_fields)!r}. Declare conflicts_with between the "
                f"offending features, or use a different format style."
            )
        lang = self._language_hint(output_fields)
        fence_open = f"```{lang}" if lang else "```"
        return (
            "## Output Format\n"
            f"Reply with a single fenced code block:\n\n"
            f"{fence_open}\n"
            "<your code here>\n"
            "```\n\n"
            "Do not include any text outside the fenced block."
        )

    def render_output(self, output: Dict[str, Any]) -> str:
        if not output:
            return ""
        # For rendered few-shot examples.
        value = next(iter(output.values()))
        lang = self._language_hint(output)
        fence = f"```{lang}" if lang else "```"
        return f"{fence}\n{value}\n```"

    def format_contexts(self, contexts: List[Context]) -> str:
        return "\n\n".join(c.content for c in contexts)

    def get_field_delimiter(self) -> str:
        return "\n"

    def get_section_delimiter(self) -> str:
        return "\n\n"

    def parse_output(self, response: str, output_fields: Dict[str, str]) -> Dict[str, Any]:
        if len(output_fields) != 1:
            raise ValueError(
                f"CodeBlockStyle.parse_output requires exactly one output_field; "
                f"got {sorted(output_fields)!r}."
            )
        field = next(iter(output_fields))
        text = response.strip()

        # Strip <think>...</think> blocks (thinking models).
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

        # Prefer the LAST fenced block — models often think before the block.
        matches = list(re.finditer(
            r"```(?:[a-zA-Z0-9_+-]*)\s*\n?(.*?)```",
            text, re.DOTALL,
        ))
        if matches:
            return {field: matches[-1].group(1).strip()}

        # No fence — assume the whole response is the code (charitable fallback).
        return {field: text}


# Format style registry
FORMAT_STYLES: Dict[str, FormatStyle] = {
    "plain":      PlainStyle(),
    "yaml":       YAMLStyle(),
    "markdown":   MarkdownStyle(),
    "json":       JSONStyle(),
    "code_block": CodeBlockStyle(),
}


def get_format_style(name: str) -> FormatStyle:
    """Get a format style by name."""
    if name not in FORMAT_STYLES:
        raise ValueError(f"Unknown format style: {name}. Available: {list(FORMAT_STYLES.keys())}")
    return FORMAT_STYLES[name]
