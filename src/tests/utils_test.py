import json
import torch
import pytest
from unittest.mock import MagicMock, patch

from ..training.utils import (
    is_valid_json,
    calculate_json_valid_rate,
    JSONTrainer,
)
from transformers.trainer_utils import EvalLoopOutput


class TestIsValidJson:
    def test_valid_json_dict(self):
        assert is_valid_json('{"store_name": "Магнит"}') is True

    def test_valid_json_list(self):
        assert is_valid_json('[1, 2, 3]') is True

    def test_valid_json_empty_dict(self):
        assert is_valid_json('{}') is True

    def test_invalid_json(self):
        assert is_valid_json('{invalid json') is False

    def test_plain_text(self):
        assert is_valid_json('просто текст') is False

    def test_empty_string(self):
        assert is_valid_json('') is False

    def test_none(self):
        assert is_valid_json(None) is False

    def test_non_string_type(self):
        assert is_valid_json(12345) is False


class TestCalculateJsonValidRate:
    def test_empty_predictions(self):
        assert calculate_json_valid_rate([]) == 0.0

    def test_all_valid(self):
        preds = ['{"a": 1}', '{"b": 2}', '{"c": 3}']
        assert calculate_json_valid_rate(preds) == 1.0

    def test_none_valid(self):
        preds = ['not json', 'also not json', 'плохо']
        assert calculate_json_valid_rate(preds) == 0.0

    def test_mixed(self):
        preds = ['{"a": 1}', 'not json', '{"b": 2}']
        assert calculate_json_valid_rate(preds) == pytest.approx(2 / 3)

    def test_none_in_predictions(self):
        preds = ['{"a": 1}', None, '{"b": 2}']
        assert calculate_json_valid_rate(preds) == pytest.approx(2 / 3)


class TestJSONTrainerGetEvalDataloader:
    def _make_trainer(self):
        trainer = JSONTrainer.__new__(JSONTrainer)
        trainer.eval_data_collator = MagicMock(return_value='collated')
        trainer.eval_dataset = ['sample1', 'sample2']

        args = MagicMock()
        args.eval_batch_size = 4
        trainer.args = args
        return trainer

    def test_uses_eval_data_collator(self):
        trainer = self._make_trainer()
        loader = trainer.get_eval_dataloader()

        assert loader.collate_fn == trainer.eval_data_collator
        assert list(loader.dataset) == ['sample1', 'sample2']

    def test_uses_eval_batch_size(self):
        trainer = self._make_trainer()
        loader = trainer.get_eval_dataloader()

        assert loader.batch_size == 4

    def test_uses_passed_dataset(self):
        trainer = self._make_trainer()
        custom_dataset = ['custom1', 'custom2', 'custom3']
        loader = trainer.get_eval_dataloader(eval_dataset=custom_dataset)

        assert list(loader.dataset) == custom_dataset


class TestJSONTrainerEvaluationLoop:
    def _make_trainer(self, preds_texts, device='cpu'):
        model = MagicMock()
        model.device = torch.device(device)
        model.eval = MagicMock()

        generated_ids = [
            torch.tensor([[1, 2, 3, 10, 11, 12], [4, 5, 6, 13, 14, 15]]),
            torch.tensor([[7, 8, 9, 16, 17, 18], [19, 20, 21, 22, 23, 24]]),
        ]
        model.generate = MagicMock(side_effect=generated_ids)

        processing_class = MagicMock()
        # batch_decode возвращает список строк для каждого батча
        processing_class.batch_decode = MagicMock(
            side_effect=lambda ids, **kwargs: preds_texts
        )

        trainer = JSONTrainer.__new__(JSONTrainer)
        trainer.model = model
        trainer.processing_class = processing_class
        return trainer

    def _make_dataloader(self, num_batches=2):
        dataloader = MagicMock()
        dataloader.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6]]),
                        'attention_mask': torch.ones(2, 3, dtype=torch.long),
                    },
                    {
                        'input_ids': torch.tensor([[7, 8, 9], [10, 11, 12]]),
                        'attention_mask': torch.ones(2, 3, dtype=torch.long),
                    },
                ][:num_batches]
            )
        )
        return dataloader

    def test_returns_eval_loop_output(self):
        trainer = self._make_trainer(
            preds_texts=['{"a": 1}', 'not json']
        )
        result = trainer.evaluation_loop(
            dataloader=self._make_dataloader(num_batches=2),
            description='eval',
            metric_key_prefix='eval',
        )

        assert isinstance(result, EvalLoopOutput)

    def test_metrics_json_valid_rate(self):
        trainer = self._make_trainer(
            preds_texts=['{"a": 1}', 'not json']
        )
        result = trainer.evaluation_loop(
            dataloader=self._make_dataloader(num_batches=2),
            description='eval',
            metric_key_prefix='eval',
        )

        assert result.metrics['eval_json_valid_rate'] == pytest.approx(0.5)

    def test_custom_metric_key_prefix(self):
        trainer = self._make_trainer(
            preds_texts=['{"a": 1}', '{"b": 2}']
        )
        result = trainer.evaluation_loop(
            dataloader=self._make_dataloader(num_batches=2),
            description='eval',
            metric_key_prefix='test',
        )

        assert result.metrics['test_json_valid_rate'] == pytest.approx(1.0)

    def test_predictions_contain_decoded_texts(self):
        preds = ['{"a": 1}', 'not json']
        trainer = self._make_trainer(preds_texts=preds)
        result = trainer.evaluation_loop(
            dataloader=self._make_dataloader(num_batches=2),
            description='eval',
            metric_key_prefix='eval',
        )

        assert result.predictions == preds * 2

    def test_model_set_to_eval(self):
        trainer = self._make_trainer(
            preds_texts=['{"a": 1}', '{"b": 2}'],
            device='cpu',
        )
        trainer.evaluation_loop(
            dataloader=self._make_dataloader(num_batches=1),
            description='eval',
            metric_key_prefix='eval',
        )

        trainer.model.eval.assert_called_once()

    def test_batch_moved_to_device(self):
        trainer = self._make_trainer(
            preds_texts=['{"a": 1}', '{"b": 2}'],
            device='cpu',
        )
        # На CPU .to(device) возвращает обычные тензоры - цикл должен отработать
        result = trainer.evaluation_loop(
            dataloader=self._make_dataloader(num_batches=1),
            description='eval',
            metric_key_prefix='eval',
        )

        trainer.model.generate.assert_called_once()
        assert isinstance(result, EvalLoopOutput)

    def test_generate_called_with_batch(self):
        trainer = self._make_trainer(
            preds_texts=['{"a": 1}', 'not json']
        )
        trainer.evaluation_loop(
            dataloader=self._make_dataloader(num_batches=1),
            description='eval',
            metric_key_prefix='eval',
        )

        trainer.model.generate.assert_called_once()
        call_kwargs = trainer.model.generate.call_args.kwargs
        assert call_kwargs['max_new_tokens'] == 1024