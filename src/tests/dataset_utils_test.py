import json
import pytest

from ..training.data.prepare_dataset import (
    _target_change,
    _make_messages,
)
from ..model.prompts import EXTRACTION_PROMPT


INNER_JSON = {
    "store_name": "Магнит",
    "tax": "10",
    "subtotal": "100",
    "ignore": "x",
    "tips": "5",
    "total": "115",
    "line_items": [
        {"item_name": "Хлеб", "item_key": "1", "item_value": "30", "item_quantity": "1"},
        {"item_name": "Молоко", "item_key": "2", "item_value": "80", "item_quantity": "2"},
    ],
}
WRAPPED_JSON = json.dumps(json.dumps(INNER_JSON, ensure_ascii=False), ensure_ascii=False)


class TestTargetChange:
    def test_returns_valid_json(self):
        result = _target_change(WRAPPED_JSON)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_removes_top_level_fields(self):
        result = json.loads(_target_change(WRAPPED_JSON))
        assert 'tax' not in result
        assert 'subtotal' not in result
        assert 'ignore' not in result
        assert 'tips' not in result
        assert 'total' in result
        assert 'store_name' in result

    def test_removes_item_key_from_line_items(self):
        result = json.loads(_target_change(WRAPPED_JSON))
        for item in result['line_items']:
            assert 'item_key' not in item
            assert 'item_name' in item
            assert 'item_value' in item
            assert 'item_quantity' in item

    def test_preserves_cyrillic(self):
        result = _target_change(WRAPPED_JSON)
        assert 'Магнит' in result
        assert 'Хлеб' in result


class TestMakeMessages:
    def test_returns_messages_key(self):
        sample = {'image': 'img_data', 'text': '{"total": "10"}'}
        result = _make_messages(sample)
        assert 'messages' in result

    def test_two_messages_user_and_assistant(self):
        sample = {'image': 'img_data', 'text': '{"total": "10"}'}
        messages = _make_messages(sample)['messages']
        assert len(messages) == 2
        assert messages[0]['role'] == 'user'
        assert messages[1]['role'] == 'assistant'

    def test_user_has_image_and_text(self):
        sample = {'image': 'img_data', 'text': '{"total": "10"}'}
        content = _make_messages(sample)['messages'][0]['content']
        assert len(content) == 2
        assert content[0]['type'] == 'image'
        assert content[0]['image'] == 'img_data'
        assert content[1]['type'] == 'text'

    def test_user_text_is_extraction_prompt(self):
        sample = {'image': 'img_data', 'text': '{"total": "10"}'}
        content = _make_messages(sample)['messages'][0]['content']
        assert content[1]['text'] == EXTRACTION_PROMPT

    def test_assistant_has_target_text(self):
        sample = {'image': 'img_data', 'text': '{"total": "10"}'}
        content = _make_messages(sample)['messages'][1]['content']
        assert len(content) == 1
        assert content[0]['type'] == 'text'
        assert content[0]['text'] == '{"total": "10"}'