import json

from core.func_registry import PromptBuildState, REGISTRY, make_func_id


SECTION_CONTENT = "You should follow the answer-format requirements below."
RULE_TEXT = "Give the final answer directly without any explanation."


def _prompt_state(style: str):
    state = PromptBuildState()
    state.format_style = style

    section_params = {
        "node_type": "section",
        "parent_id": "__root__",
        "payload": {
            "title": "format_fix",
            "content": SECTION_CONTENT,
            "ordinal": 30,
            "is_system": True,
            "min_rules": 0,
            "max_rules": 4,
        },
    }
    section_id = make_func_id("insert_node", section_params)
    REGISTRY["insert_node"](state, {
        "_func_id": section_id,
        **section_params,
    })

    REGISTRY["insert_node"](state, {
        "node_type": "rule",
        "parent_id": section_id,
        "payload": {"content": RULE_TEXT},
    })

    if style == "code_block":
        REGISTRY["insert_node"](state, {
            "node_type": "output_field",
            "parent_id": "__root__",
            "payload": {
                "name": "code",
                "description": "Executable code that prints the final answer.",
            },
        })

    return state.to_prompt_state()


def test_plain_section_content_renders_before_rules():
    rendered = _prompt_state("plain")._build_system_content()

    assert rendered.splitlines().count("format_fix") == 1
    assert SECTION_CONTENT in rendered
    assert RULE_TEXT in rendered
    assert rendered.index(SECTION_CONTENT) < rendered.index(RULE_TEXT)


def test_json_section_content_is_preserved_with_items():
    rendered = _prompt_state("json")._build_system_content()
    obj = json.loads(rendered)

    assert obj["format_fix"]["content"] == SECTION_CONTENT
    assert obj["format_fix"]["items"] == [RULE_TEXT]


def test_markdown_section_content_renders_before_rules():
    rendered = _prompt_state("markdown")._build_system_content()

    assert "## format_fix" in rendered
    assert SECTION_CONTENT in rendered
    assert RULE_TEXT in rendered
    assert rendered.index(SECTION_CONTENT) < rendered.index(RULE_TEXT)


def test_yaml_section_content_renders_before_rules():
    rendered = _prompt_state("yaml")._build_system_content()

    assert "format_fix:" in rendered
    assert f"- {SECTION_CONTENT}" in rendered
    assert f"- {RULE_TEXT}" in rendered
    assert rendered.index(SECTION_CONTENT) < rendered.index(RULE_TEXT)


def test_code_block_section_content_renders_before_rules():
    rendered = _prompt_state("code_block")._build_system_content()

    assert "## format_fix" in rendered
    assert SECTION_CONTENT in rendered
    assert RULE_TEXT in rendered
    assert rendered.index(SECTION_CONTENT) < rendered.index(RULE_TEXT)
