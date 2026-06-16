from task5lib.dpo import fallback_rejected, make_dpo_prompt


def test_make_dpo_prompt():
    prompt = make_dpo_prompt("热显热")
    assert "### 指令" in prompt
    assert "请将以下温州话翻译成普通话。" in prompt
    assert "热显热" in prompt


def test_fallback_rejected_changes_answer():
    chosen = "今天很热，我不想出去。"
    rejected, tag = fallback_rejected(chosen)
    assert rejected != chosen
    assert tag in {"antonym_swap", "drop_content", "reverse_chars", "delete_random_char", "repeat_first_char"}
