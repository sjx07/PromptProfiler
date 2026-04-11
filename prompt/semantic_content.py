from __future__ import annotations

"""Semantic content structures - the WHAT of prompts.

Semantic content represents the meaning/content that goes into prompts:
- Rules and rule sections
- Instructions
- Few-shot examples
- Context blocks
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from prompt_profiler.prompt.rules import RuleSection, RuleTree


@dataclass
class Instruction:
    """Task instruction/objective."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Example:
    """Few-shot example."""

    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    """Context block (e.g., schema, background info)."""

    name: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticContent:
    """Complete semantic content of a prompt.

    This is the WHAT - the actual content/meaning, independent of formatting.
    """

    # Task I/O specification (what the task needs)
    input_fields: Dict[str, str] = field(default_factory=dict)
    output_fields: Dict[str, str] = field(default_factory=dict)

    # Semantic blocks
    instruction: Optional[Instruction] = None
    tree: RuleTree = field(default_factory=RuleTree)
    rule_sections: List[RuleSection] = field(default_factory=list)
    examples: List[Example] = field(default_factory=list)
    contexts: List[Context] = field(default_factory=list)

    def add_rule_section(self, section: RuleSection) -> None:
        """Add a rule section."""
        self.rule_sections.append(section)

    def add_example(self, example: Example) -> None:
        """Add a few-shot example."""
        self.examples.append(example)

    def add_context(self, context: Context) -> None:
        """Add a context block."""
        self.contexts.append(context)
