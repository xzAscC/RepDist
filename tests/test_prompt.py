from repdist.extract import prompt_messages


def test_prompt_messages_drops_assistant():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "<think>x</think> answer"},
    ]
    prompt = prompt_messages(messages)
    assert prompt == [{"role": "user", "content": "hello"}]


def test_prompt_messages_keeps_system_and_user():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    prompt = prompt_messages(messages)
    assert [m["role"] for m in prompt] == ["system", "user"]
