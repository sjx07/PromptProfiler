"""Clean PromptState with ontological separation.

This version separates:
- Semantic content (WHAT): rules, examples, contexts
- Syntactic style (HOW): format style for rendering
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Callable

from .semantic_content import SemanticContent, RuleSection, Example, Context, Instruction
from .format_styles import FormatStyle, get_format_style


@dataclass
class PromptState:
    """Prompt state with two ontological dimensions.

    Two ontological dimensions:
    1. semantic: WHAT to say (content, meaning)
    2. format_style: HOW to say it (formatting)

    Everything else is utility/configuration.
    """

    # ========== ONTOLOGICAL DIMENSION 1: Semantic (WHAT) ==========
    semantic: SemanticContent = field(default_factory=SemanticContent)

    # ========== ONTOLOGICAL DIMENSION 2: Syntactic (HOW) ==========
    format_style_name: str = "plain"
    _format_style: Optional[FormatStyle] = field(default=None, repr=False)

    # ========== Utility/Configuration (Non-ontological) ==========
    custom_system_template: Optional[str] = None
    custom_user_template: Optional[str] = None
    demo_selector: Optional[Callable[[Dict[str, Any], List[Dict[str, Any]]], List[Dict[str, Any]]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def format_style(self) -> FormatStyle:
        """Get the format style instance."""
        if self._format_style is None:
            self._format_style = get_format_style(self.format_style_name)
        return self._format_style

    @property
    def effective_output_fields(self) -> Dict[str, str]:
        """Get output fields as declared in the config (no boolean injection)."""
        return self.semantic.output_fields

    @property
    def effective_semantic(self) -> SemanticContent:
        """Get semantic content as declared in the config."""
        return self.semantic

    def clone(self, **updates: Any) -> PromptState:
        """Create a copy with specified updates."""
        return replace(self, **updates)

    def build_messages(self, record: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build chat messages for this record."""
        messages = []

        # System message
        # if self.use_system_message:
        system_content = self._build_system_content()
        messages.append({"role": "system", "content": system_content})

        # Demo messages (few-shot examples)
        if self.semantic.examples or self.demo_selector:
            demo_messages = self._build_demo_messages(record)
            messages.extend(demo_messages)

        # User message
        user_content = self._build_user_content(record)
        messages.append({"role": "user", "content": user_content})

        return messages
        # return self.format_style.build_messages(record, self.effective_semantic)

    def parse_output(self, response_text: str) -> Dict[str, Any]:
        """Parse model response text into output fields.

        Strips <think>...</think> blocks from thinking models (e.g. Qwen3-Coder-Next).
        """
        response_text = re.sub(r"<think>.*?</think>\s*", "", response_text, flags=re.DOTALL)
        return self.format_style.parse_output(response_text, self.effective_output_fields)
    
    def _build_system_content(self) -> str:
        """Build system message using format style.

        The format style handles ALL formatting - no hardcoded markers here!
        """
        if self.custom_system_template:
            return self.custom_system_template

        # Delegate completely to format style, using effective semantic
        return self.format_style.format_system_message(self.effective_semantic)

    def _build_user_content(self, record: Dict[str, Any]) -> str:
        """Build user message content."""
        return self.format_style.format_user_message(record, self.semantic)

    def _build_demo_messages(self, record: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build few-shot demo messages."""
        messages = []

        # Select examples
        examples = self.semantic.examples
        if self.demo_selector:
            # Convert to old format for compatibility
            examples_dicts = [{"inputs": ex.inputs, "outputs": ex.outputs} for ex in examples]
            selected = self.demo_selector(record, examples_dicts)
            examples = [Example(inputs=ex["inputs"], outputs=ex["outputs"]) for ex in selected]

        for example in examples:
            # User message with inputs (only non-empty fields)
            filtered_inputs = {k: v for k, v in example.inputs.items() if v}
            user_msg = self._format_style.format_user_message(
                filtered_inputs,
                SemanticContent(input_fields=filtered_inputs),
            )
            messages.append({"role": "user", "content": user_msg})

            # Assistant message with outputs (same format LLM is expected to produce)
            assistant_msg = self._format_style.render_output(example.outputs)
            messages.append({"role": "assistant", "content": assistant_msg})

        return messages

    # Convenience methods for adding semantic content

    def add_rule_section(self, section: RuleSection) -> PromptState:
        """Add a rule section (returns new state)."""
        new_semantic = SemanticContent(
            instruction=self.semantic.instruction,
            rule_sections=self.semantic.rule_sections + [section],
            examples=self.semantic.examples,
            contexts=self.semantic.contexts
        )
        return self.clone(semantic=new_semantic)

    def add_example(self, example: Example) -> PromptState:
        """Add a few-shot example (returns new state)."""
        new_semantic = SemanticContent(
            instruction=self.semantic.instruction,
            rule_sections=self.semantic.rule_sections,
            examples=self.semantic.examples + [example],
            contexts=self.semantic.contexts
        )
        return self.clone(semantic=new_semantic)

    def add_context(self, context: Context) -> PromptState:
        """Add a context block (returns new state)."""
        new_semantic = SemanticContent(
            instruction=self.semantic.instruction,
            rule_sections=self.semantic.rule_sections,
            examples=self.semantic.examples,
            contexts=self.semantic.contexts + [context]
        )
        return self.clone(semantic=new_semantic)

    def set_instruction(self, instruction: str) -> PromptState:
        """Set the instruction (returns new state)."""
        new_semantic = SemanticContent(
            instruction=Instruction(text=instruction),
            rule_sections=self.semantic.rule_sections,
            examples=self.semantic.examples,
            contexts=self.semantic.contexts
        )
        return self.clone(semantic=new_semantic)

    def set_format_style(self, style_name: str) -> PromptState:
        """Change the format style (returns new state)."""
        return self.clone(format_style_name=style_name, _format_style=None)
