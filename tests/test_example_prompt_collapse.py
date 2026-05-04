from task import _collapse_messages_to_system_user


def test_demo_turns_are_preserved_when_collapsing_to_two_message_prompt():
    system, user = _collapse_messages_to_system_user([
        {"role": "system", "content": "system rules"},
        {"role": "user", "content": "demo input"},
        {"role": "assistant", "content": "demo output"},
        {"role": "user", "content": "current input"},
    ])

    assert system == "system rules"
    assert "## Example 1 input" in user
    assert "demo input" in user
    assert "## Example 1 output" in user
    assert "demo output" in user
    assert "## Current input" in user
    assert "current input" in user
