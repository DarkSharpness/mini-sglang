from unittest.mock import Mock, call

import torch
from minisgl.message import TokenizeMsg
from minisgl.tokenizer.tokenize import TokenizeManager


def _msg(uid: int, text: str | list[dict[str, str]]) -> TokenizeMsg:
    return TokenizeMsg(uid=uid, text=text, sampling_params=Mock())


def test_tokenize_batches_prompts() -> None:
    tokenizer = Mock()
    tokenizer.return_value = {"input_ids": [[1, 2, 3], [4, 5]]}
    manager = TokenizeManager(tokenizer)

    results = manager.tokenize([_msg(1, "first prompt"), _msg(2, "second prompt")])

    tokenizer.assert_called_once_with(
        ["first prompt", "second prompt"],
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    assert len(results) == 2
    assert torch.equal(results[0], torch.tensor([1, 2, 3], dtype=torch.int32))
    assert torch.equal(results[1], torch.tensor([4, 5], dtype=torch.int32))


def test_tokenize_applies_chat_templates_before_batching() -> None:
    tokenizer = Mock()
    tokenizer.apply_chat_template.side_effect = ["formatted first", "formatted second"]
    tokenizer.return_value = {"input_ids": [[1], [2], [3]]}
    manager = TokenizeManager(tokenizer)
    first_chat = [{"role": "user", "content": "first"}]
    second_chat = [{"role": "user", "content": "second"}]

    manager.tokenize([_msg(1, first_chat), _msg(2, "plain"), _msg(3, second_chat)])

    assert tokenizer.apply_chat_template.call_args_list == [
        call(first_chat, tokenize=False, add_generation_prompt=True),
        call(second_chat, tokenize=False, add_generation_prompt=True),
    ]
    assert tokenizer.call_args.args[0] == ["formatted first", "plain", "formatted second"]


def test_tokenize_empty_batch_does_not_call_tokenizer() -> None:
    tokenizer = Mock()
    manager = TokenizeManager(tokenizer)

    assert manager.tokenize([]) == []
    tokenizer.assert_not_called()
