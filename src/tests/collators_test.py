import torch
import pytest
from unittest.mock import MagicMock, patch

from ..model.data.processed.prepare_dataset import (
    TrainQwenCollator,
    EvalQwenCollator,
    BaseQwenCollator,
)


class MockTokenizer:
    """Мок токенизатора, имитирующий поведение Qwen2-VL токенизатора."""

    def __init__(self):
        self.pad_token_id = 151643
        self.image_token = "<|image_pad|>"
        self.image_token_id = 151655
        # <|im_start|>assistant\n = [151644, 8948, 13]
        self.assistant_tokens = [151644, 8948, 13]

    def convert_tokens_to_ids(self, token):
        if token == self.image_token:
            return self.image_token_id
        return 0

    def encode(self, text, return_tensors=None):
        if text == "<|im_start|>assistant\n" or text == "assistant\n":
            tokens = self.assistant_tokens
        else:
            tokens = [1, 2, 3]
        if return_tensors == "pt":
            return torch.tensor([tokens])
        return tokens


class MockProcessor:
    """Мок процессора, имитирующий поведение Qwen2-VL процессора."""

    def __init__(self):
        self.tokenizer = MockTokenizer()
        self.image_token = "<|image_pad|>"
        self.call_count = 0
        self.last_call_kwargs = None
        self.apply_chat_template = MagicMock(
            side_effect=lambda messages, tokenize=False, add_generation_prompt=False: (
                "user: <|image_pad|> prompt\nassistant\n"
            )
        )

    def __call__(self, text, images, return_tensors=None, padding=False):
        self.call_count += 1
        self.last_call_kwargs = {
            "text": text,
            "images": images,
            "return_tensors": return_tensors,
            "padding": padding,
        }
        batch_size = len(text)
        input_ids = [
            [
                151644,  # <|im_start|>
                151655,  # image_token
                151645,  # <|im_end|>
                151644,  # <|im_start|>
                8948,    # assistant
                13,      # \n
                151645,  # <|im_end|>
                151643,  # pad
            ]
            for _ in range(batch_size)
        ]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.ones(batch_size, 8, dtype=torch.long),
            "pixel_values": torch.zeros(batch_size, 3, 4, 4),
        }


@pytest.fixture
def processor():
    return MockProcessor()


@pytest.fixture
def examples():
    return [
        {
            "messages": [
                {"role": "user", "content": [{"type": "image", "image": "img1"}, {"type": "text", "text": "prompt"}]},
                {"role": "assistant", "content": [{"type": "text", "text": '{"total": "10"}'}]},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": [{"type": "image", "image": "img2"}, {"type": "text", "text": "prompt"}]},
                {"role": "assistant", "content": [{"type": "text", "text": '{"total": "20"}'}]},
            ]
        },
    ]


class _ConcreteCollator(BaseQwenCollator):
    """Конкретный подкласс для тестирования абстрактного BaseQwenCollator."""

    def __init__(self, processor, add_generation_prompt=False):
        super().__init__(processor)
        self.add_generation_prompt = add_generation_prompt

    def __call__(self, examples):
        return self._prepare_inputs(
            examples=examples, add_generation_prompt=self.add_generation_prompt
        )


class TestBaseQwenCollator:
    def test_prepare_inputs_calls_process_vision_info(self, processor, examples):
        collator = _ConcreteCollator(processor)
        with patch(
            "src.model.data.processed.prepare_dataset.process_vision_info",
            return_value=("img", None),
        ) as mock_vision:
            batch = collator._prepare_inputs(examples, add_generation_prompt=False)

        assert mock_vision.call_count == len(examples)
        assert "input_ids" in batch
        assert "attention_mask" in batch

    def test_prepare_inputs_calls_processor_once(self, processor, examples):
        collator = _ConcreteCollator(processor)
        with patch(
            "src.model.data.processed.prepare_dataset.process_vision_info",
            return_value=("img", None),
        ):
            collator._prepare_inputs(examples, add_generation_prompt=False)

        assert processor.call_count == 1

    def test_prepare_inputs_calls_processor_with_correct_args(self, processor, examples):
        collator = _ConcreteCollator(processor)
        with patch(
            "src.model.data.processed.prepare_dataset.process_vision_info",
            return_value=("img", None),
        ):
            collator._prepare_inputs(examples, add_generation_prompt=False)

        kwargs = processor.last_call_kwargs
        assert kwargs["return_tensors"] == "pt"
        assert kwargs["padding"] is True
        assert len(kwargs["text"]) == len(examples)
        assert len(kwargs["images"]) == len(examples)

    def test_prepare_inputs_apply_chat_template_false(self, processor, examples):
        collator = _ConcreteCollator(processor)
        with patch(
            "src.model.data.processed.prepare_dataset.process_vision_info",
            return_value=("img", None),
        ):
            collator._prepare_inputs(examples, add_generation_prompt=False)

        for call in processor.apply_chat_template.call_args_list:
            assert call.kwargs["add_generation_prompt"] is False

    def test_prepare_inputs_apply_chat_template_true(self, processor, examples):
        collator = _ConcreteCollator(processor)
        with patch(
            "src.model.data.processed.prepare_dataset.process_vision_info",
            return_value=("img", None),
        ):
            collator._prepare_inputs(examples, add_generation_prompt=True)

        for call in processor.apply_chat_template.call_args_list:
            assert call.kwargs["add_generation_prompt"] is True


class TestTrainQwenCollator:
    def test_labels_present(self, processor, examples):
        collator = TrainQwenCollator(processor)
        with patch(
            "src.model.data.processed.prepare_dataset.process_vision_info",
            return_value=("img", None),
        ):
            batch = collator(examples)

        assert "labels" in batch
        assert batch["labels"].shape == batch["input_ids"].shape

    def test_pad_tokens_masked(self, processor, examples):
        collator = TrainQwenCollator(processor)
        with patch(
            "src.model.data.processed.prepare_dataset.process_vision_info",
            return_value=("img", None),
        ):
            batch = collator(examples)

        pad_id = processor.tokenizer.pad_token_id
        labels = batch["labels"]
        input_ids = batch["input_ids"]
        # Все pad-токены в input_ids должны быть -100 в labels
        assert torch.all(labels[input_ids == pad_id] == -100)

    def test_image_tokens_masked(self, processor, examples):
        collator = TrainQwenCollator(processor)
        with patch(
            "src.model.data.processed.prepare_dataset.process_vision_info",
            return_value=("img", None),
        ):
            batch = collator(examples)

        image_id = processor.tokenizer.image_token_id
        labels = batch["labels"]
        input_ids = batch["input_ids"]
        # Все image-токены в input_ids должны быть -100 в labels
        assert torch.all(labels[input_ids == image_id] == -100)

    def test_tokens_before_assistant_masked(self, processor, examples):
        collator = TrainQwenCollator(processor)
        with patch(
            "src.model.data.processed.prepare_dataset.process_vision_info",
            return_value=("img", None),
        ):
            batch = collator(examples)

        labels = batch["labels"]
        # В мок-батче маркер <|im_start|>assistant\n начинается с индекса 3: [151644, 8948, 13].
        # Всё до и включая конец маркера (индексы 0-5) должно быть -100.
        assert torch.all(labels[:, :6] == -100)

    def test_tokens_after_assistant_not_masked(self, processor, examples):
        collator = TrainQwenCollator(processor)
        with patch(
            "src.model.data.processed.prepare_dataset.process_vision_info",
            return_value=("img", None),
        ):
            batch = collator(examples)

        labels = batch["labels"]
        input_ids = batch["input_ids"]
        # Токен после маркера assistant\n (индекс 6) не pad и не image - должен остаться.
        assert labels[0, 6] == input_ids[0, 6]

    def test_apply_chat_template_add_generation_prompt_false(self, processor, examples):
        collator = TrainQwenCollator(processor)
        with patch(
            "src.model.data.processed.prepare_dataset.process_vision_info",
            return_value=("img", None),
        ):
            collator(examples)

        for call in processor.apply_chat_template.call_args_list:
            assert call.kwargs["add_generation_prompt"] is False


class TestEvalQwenCollator:
    def test_no_labels(self, processor, examples):
        collator = EvalQwenCollator(processor)
        with patch(
            "src.model.data.processed.prepare_dataset.process_vision_info",
            return_value=("img", None),
        ):
            batch = collator(examples)

        assert "labels" not in batch

    def test_apply_chat_template_add_generation_prompt_true(self, processor, examples):
        collator = EvalQwenCollator(processor)
        with patch(
            "src.model.data.processed.prepare_dataset.process_vision_info",
            return_value=("img", None),
        ):
            collator(examples)

        for call in processor.apply_chat_template.call_args_list:
            assert call.kwargs["add_generation_prompt"] is True