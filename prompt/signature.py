"""Feature Signature - defines the I/O contract for a single LLM call.

Similar to dspy.Signature, but wraps a PromptState which already contains
all the semantic content, formatting, and tree_index for attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from copy import deepcopy

from .prompt_state import PromptState
from .semantic_content import SemanticContent, Instruction, Example, Context
from .rules import RuleSection, RuleTree, RuleItem, parse_sections_with_index, load_sections_from_classification_results, _get_node_id
from .format_styles import get_format_style

import logging
logger = logging.getLogger(__name__)

@dataclass
class FeatureSignature:
    """Defines the I/O contract for a single pipeline step.

    This wraps a PromptState which contains all the semantic content,
    formatting style, and tree_index for attribution. The signature
    just adds a name for identification.

    Example:
        # Create from scratch
        semantic = SemanticContent(
            input_fields={"schema": "Database schema", "question": "User question"},
            output_fields={"sql_query": "Generated SQL"},
            instruction=Instruction(text="Generate SQL from the question"),
        )
        sig = FeatureSignature(
            name="sql_generator",
            prompt_state=PromptState(semantic=semantic, format_style_name="yaml"),
        )

        # Or use the convenience constructor
        sig = FeatureSignature.create(
            name="sql_generator",
            instruction="Generate SQL from the question",
            input_fields={"schema": "Database schema", "question": "User question"},
            output_fields={"sql_query": "Generated SQL"},
        )
    """

    name: str
    prompt_state: PromptState

    # ----- Convenience accessors (delegate to prompt_state.semantic) -----

    @property
    def semantic(self) -> SemanticContent:
        return self.prompt_state.semantic

    @property
    def input_fields(self) -> Dict[str, str]:
        return self.semantic.input_fields

    @property
    def output_fields(self) -> Dict[str, str]:
        return self.semantic.output_fields

    @property
    def instruction(self) -> Optional[Instruction]:
        return self.semantic.instruction

    @property
    def rule_sections(self) -> List[RuleSection]:
        return self.semantic.rule_sections

    @property
    def tree(self) -> RuleTree:
        return self.semantic.tree

    @property
    def examples(self) -> List[Example]:
        return self.semantic.examples

    @property
    def contexts(self) -> List[Context]:
        return self.semantic.contexts

    @property
    def format_style_name(self) -> str:
        return self.prompt_state.format_style_name

    # ----- Attribution helpers -----

    @property
    def has_rules(self) -> bool:
        """Check if this signature has any rules."""
        return len(self.rule_sections) > 0

    @property
    def rule_ids(self) -> List[str]:
        """Get all rule IDs from the tree_index."""
        return [
            nid for nid, ref in self.tree._index.items()
            if isinstance(ref, RuleItem)
        ]

    @property
    def section_ids(self) -> List[str]:
        """Get all section IDs (roots of the tree)."""
        return [s.node_id for s in self.tree.roots]

    def get_enabled_rule_ids(self) -> List[str]:
        """Get IDs of currently enabled rules."""
        return self.tree.mask.enabled_rule_ids(self.tree)

    def toggle_section(self, section_id: str, enabled: bool) -> None:
        """Enable/disable all rules in a section (for attribution)."""
        self.tree.mask.toggle_subtree(self.tree, section_id, enabled)

    def apply_rule_mask(self, mask: Dict[str, bool]) -> None:
        """Apply a mask to enable/disable specific rules (for attribution)."""
        self.tree.mask.apply(mask)

    # ----- Message building (delegate to prompt_state) -----

    def build_messages(self, record: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build chat messages for a given input record."""
        return self.prompt_state.build_messages(record)

    def parse_output(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract output fields."""
        logger.debug(f"Parsing response: {response}")
        
        return self.prompt_state.parse_output(response)

    # ----- Builders for creating modified signatures -----

    def clone(self, **updates) -> "FeatureSignature":
        """Create a deep copy with optional updates."""
        new_state = deepcopy(self.prompt_state)
        for key, value in updates.items():
            if hasattr(new_state, key):
                setattr(new_state, key, value)
        return FeatureSignature(name=self.name, prompt_state=new_state)

    def with_examples(self, examples: List[Example]) -> "FeatureSignature":
        """Return a new signature with additional examples."""
        new_state = deepcopy(self.prompt_state)
        new_state.semantic.examples = self.examples + examples
        return FeatureSignature(name=self.name, prompt_state=new_state)

    def with_rules(
        self,
        sections: List[RuleSection],
        tree: Optional[RuleTree] = None
    ) -> "FeatureSignature":
        """Return a new signature with additional rules."""
        new_state = deepcopy(self.prompt_state)
        new_state.semantic.rule_sections = self.rule_sections + sections
        if tree is not None:
            new_state.semantic.tree = tree
        return FeatureSignature(name=self.name, prompt_state=new_state)

    def with_rules_from_json(self, data: Dict[str, Any]) -> "FeatureSignature":
        """Return a new signature with rules loaded from JSON data."""
        sections, tree = parse_sections_with_index(data)
        return self.with_rules(sections, tree)

    def with_format_style(self, style_name: str) -> "FeatureSignature":
        """Return a new signature with a different format style."""
        new_state = deepcopy(self.prompt_state)
        new_state.format_style_name = style_name
        new_state._format_style = None  # Reset cached style
        return FeatureSignature(name=self.name, prompt_state=new_state)

    # ----- Class methods for convenient creation -----

    @classmethod
    def create(
        cls,
        name: str,
        instruction: str,
        input_fields: Dict[str, str],
        output_fields: Dict[str, str],
        rule_sections: Optional[List[RuleSection]] = None,
        tree: Optional[RuleTree] = None,
        examples: Optional[List[Example]] = None,
        contexts: Optional[List[Context]] = None,
        format_style: str = "plain",
    ) -> "FeatureSignature":
        """Convenience constructor to create a signature from components.

        Args:
            name: Signature name
            instruction: Task instruction text
            input_fields: Input field definitions
            output_fields: Output field definitions
            rule_sections: Optional rule sections
            tree_index: Optional tree index for attribution
            examples: Optional few-shot examples
            contexts: Optional context blocks
            format_style: Format style name (plain, yaml, markdown, json)

        Returns:
            A new FeatureSignature
        """
        semantic = SemanticContent(
            input_fields=input_fields,
            output_fields=output_fields,
            instruction=Instruction(text=instruction),
            rule_sections=rule_sections or [],
            tree=tree or RuleTree(),
            examples=examples or [],
            contexts=contexts or [],
        )
        prompt_state = PromptState(
            semantic=semantic,
            format_style_name=format_style,
        )
        return cls(name=name, prompt_state=prompt_state)

    @classmethod
    def from_json(
        cls,
        name: str,
        instruction: str,
        input_fields: Dict[str, str],
        output_fields: Dict[str, str],
        rules_json_path: str,
        format_style: str = "plain",
        load_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "FeatureSignature":
        """Create a signature with rules loaded from a JSON file.

        Uses load_sections_from_classification_results which handles
        different JSON formats (list with scores, dict with sections, etc.).

        Args:
            name: Signature name
            instruction: Task instruction
            input_fields: Input field definitions
            output_fields: Output field definitions
            rules_json_path: Path to rules JSON file (classification results)
            format_style: Format style name

        Returns:
            FeatureSignature with rules and tree_index loaded
        """
        tree = load_sections_from_classification_results(
            rules_json_path,
            **(load_kwargs or {}),
        )

        return cls.create(
            name=name,
            instruction=instruction,
            input_fields=input_fields,
            output_fields=output_fields,
            rule_sections=tree.roots,
            tree=tree,
            format_style=format_style,
        )


__all__ = [
    "FeatureSignature",
]
