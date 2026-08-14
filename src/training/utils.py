from transformers import Trainer
from transformers.trainer_utils import EvalLoopOutput
from torch.utils.data import DataLoader
import json
import torch

def is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def calculate_json_valid_rate(predictions):
    if not predictions:
        return 0.0

    valid = sum(
        is_valid_json(prediction)
        for prediction in predictions
    )

    return valid / len(predictions)



class JSONTrainer(Trainer):

    def __init__(self, *args, eval_data_collator=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.eval_data_collator = eval_data_collator
    

    def get_eval_dataloader(self, eval_dataset = None):

        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.eval_data_collator,
        )

    def evaluation_loop(self, dataloader, description, prediction_loss_only=False, ignore_keys=None, metric_key_prefix='eval'):

        self.model.eval()

        preds = []

        for batch in dataloader:

            
            batch = {k:v.to(self.model.device) for k, v in batch.items()}
            with torch.no_grad():
                generated_ids = self.model.generate(**batch, max_new_tokens=1024)
            generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch['input_ids'], generated_ids)
            ]
            output_text = self.processing_class.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            preds.extend(output_text)

        json_valid_rate = calculate_json_valid_rate(preds)

        metrics = {f"{metric_key_prefix}_json_valid_rate": json_valid_rate}

        return EvalLoopOutput(
            predictions=preds,
            label_ids=None,
            metrics=metrics,
            num_samples=len(preds),
        )