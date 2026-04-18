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

def fallback_parse_output(response: str, output_fields: Dict[str, str]) -> Dict[str, Any]:
    """Fallback parser that tries multiple parsing strategies.

    Tries in order:
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

    # Strategy 1: Try to parse as JSON
    try:
        # Look for JSON object in response
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
            parts.append(rules_text)

        # Contexts
        if semantic.contexts:
            parts.append(self.format_contexts(semantic.contexts))

        # Structure template with output instructions
        if semantic.output_fields:
            structure = self.format_structure_template(semantic.input_fields, semantic.output_fields)
            parts.append(structure)

        # Final instruction
        instruction_text = semantic.instruction.text if semantic.instruction else "complete the task"
        parts.append(f'Instruction: {instruction_text}')

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
        
        def emit_rule_nodes(nodes: List[Any], indent: str) -> None:
            # number needs local counter per block
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
                        lines.append(f"{indent}{_emit_prefix('number', num)} {txt}")
                        num += 1
                    else:
                        lines.append(f"{indent}{_emit_prefix(kind)} {txt}")

                elif isinstance(node, RuleGroup):
                    if node.group_id and not tree.is_enabled(node.group_id):
                        continue
                    title = _safe_strip(getattr(node, "title", "")) or "Untitled Group"
                    # group itself is also an item in the same environment
                    if kind == "number":
                        lines.append(f"{indent}{_emit_prefix('number', num)} {title}")
                        num += 1
                    else:
                        lines.append(f"{indent}{_emit_prefix(kind)} {title}")

                    # children indented one step
                    child_rules = node.children
                    if child_rules:
                        emit_rule_nodes(child_rules, indent + "  ")

        for sec in sections or []:
            sid = sec.metadata.get("id") or sec.metadata.get("path") if sec.metadata else None
            if sid and not tree.is_enabled(sid):
                continue
            lines.append(fmt_section_title(sec, level))

            content = (getattr(sec, "content", "") or "").strip()
            subs = [c for c in sec.children if isinstance(c, RuleSection)]

            if subs:
                lines.append(self._format_rule_sections_impl(subs, level=level + 1, tree=tree))
            
            nodes = list(sec.iter_rule_children() or [])
            if nodes:
                emit_rule_nodes(nodes, indent="  ")
            elif content and not subs:
                lines.append(f"  {content}")
                
            lines.append("\n")

        return "\n".join(lines)

    def format_field_descriptions(self, fields: Dict[str, str], label: str) -> str:
        """Format as natural language list."""
        title = f"{label}"

        field_parts = [title]
        for name, desc in fields.items():
            field_parts.append(f"{name} ({desc})")
        return f" {label}: " + ", ".join(field_parts) + "."

    def format_structure_template(self, input_fields: Dict[str, str], output_fields: Dict[str, str]) -> str:
        """Format as plain field: value pairs with explicit output instructions."""
        lines = ["Output Format Instructions:"]
        lines.append("You must respond with ONLY the output fields in the following format:")
        lines.append("")

        # Show output format template
        for name in output_fields.keys():
            lines.append(f"{name}: <your_answer_here>")

        lines.append("")
        lines.append("Important: Provide only the output fields above, nothing else.")
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

            if items:
                out.extend(items)
            elif content and not subs:
                out.append(f"{pad(base_indent + 1)}- {content}")

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
        lines: List[str] = []

        def fmt_section_title(sec: "RuleSection", lvl: int) -> str:
            title = _safe_strip(getattr(sec, "title", "")) or "Untitled Section"
            return f"{'#' * (lvl + 1)} {title}"

        def emit_rule_nodes(nodes: List[Any], indent: str) -> None:
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
                        lines.append(f"{indent}{num}. {txt}")
                        num += 1
                    else:
                        lines.append(f"{indent}- {txt}")

                elif isinstance(node, RuleGroup):
                    if node.group_id and not tree.is_enabled(node.group_id):
                        continue
                    title = _safe_strip(getattr(node, "title", "")) or "Untitled Group"
                    if kind == "number":
                        lines.append(f"{indent}{num}. {title}")
                        num += 1
                    else:
                        lines.append(f"{indent}- {title}")

                    child_rules = node.children
                    if child_rules:
                        emit_rule_nodes(child_rules, indent + "  ")

        for sec in sections or []:
            sid = sec.metadata.get("id") or sec.metadata.get("path") if sec.metadata else None
            if sid and not tree.is_enabled(sid):
                continue
            lines.append(fmt_section_title(sec, level))

            content = _safe_strip(getattr(sec, "content", ""))
            subs = [c for c in sec.children if isinstance(c, RuleSection)]

            nodes = list(sec.iter_rule_children() or [])
            if nodes:
                emit_rule_nodes(nodes, indent="")
            elif content and not subs:
                lines.append(content)

            if subs:
                lines.append(self._format_rule_sections_impl(subs, level=level + 1, tree=tree))

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

            if items and subs:
                obj["items"] = items
                obj["subsections"] = [section_obj(s) for s in subs]
            elif items:
                obj["items"] = items
            elif subs:
                obj["subsections"] = [section_obj(s) for s in subs]
            elif content:
                obj["content"] = content

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
        result = {}
        response_stripped = response.strip()

        # Try to parse as complete JSON first
        try:
            parsed = json.loads(response_stripped)
            if isinstance(parsed, dict):
                for field_name in output_fields.keys():
                    if field_name in parsed:
                        result[field_name] = parsed[field_name]
                if result:
                    logger.debug("Successfully parsed output as complete JSON")
                    return result
        except json.JSONDecodeError:
            pass

        # Try to find JSON object within response (e.g., wrapped in markdown code blocks)
        json_patterns = [
            r'```json\s*\n(.*?)\n```',  # ```json ... ```
            r'```\s*\n(\{.*?\})\n```',  # ``` {...} ```
            r'(\{[^{}]*\})',  # Simple {...} object
            r'(\{.*\})',  # Any JSON-like object (greedy)
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response_stripped, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1).strip()
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        for field_name in output_fields.keys():
                            if field_name in parsed:
                                result[field_name] = parsed[field_name]
                        if result:
                            logger.debug("Successfully parsed embedded JSON in response")
                            return result
                except json.JSONDecodeError:
                    continue

        # If JSON parsing failed, use fallback
        return fallback_parse_output(response, output_fields)


# Format style registry
FORMAT_STYLES: Dict[str, FormatStyle] = {
    "plain": PlainStyle(),
    "yaml": YAMLStyle(),
    "markdown": MarkdownStyle(),
    "json": JSONStyle(),
}


def get_format_style(name: str) -> FormatStyle:
    """Get a format style by name."""
    if name not in FORMAT_STYLES:
        raise ValueError(f"Unknown format style: {name}. Available: {list(FORMAT_STYLES.keys())}")
    return FORMAT_STYLES[name]
