from datasets import load_dataset, Dataset, Image
from sklearn.model_selection import train_test_split
import json
from qwen_vl_utils import process_vision_info
import torch
from ...training.prompts import EXTRACTION_PROMPT
from abc import ABC, abstractmethod


def _target_change(t: str):
    t = json.loads(json.loads(t))
    del t['tax']
    del t['subtotal']
    del t['ignore']
    del t['tips']
    for d in t['line_items']:
        del d['item_key']
    return json.dumps(t, ensure_ascii=False)


def _make_messages(sample):
  messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample['image']
                },
                {
                    "type": "text",
                    "text": EXTRACTION_PROMPT
            }]
            },
  {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": sample['text']
                }
            ]
        },
  ]

  return {'messages': messages}


def prepare_datasets(
    dataset_name: str = "mychen76/ds_receipts_v2_test",
    test_size: float = 0.1,
    random_state: int = 42,
):
    ds = load_dataset(dataset_name)
    ds = ds['train'].to_pandas()

    ds['text'] = ds['text'].apply(_target_change)

    ds_train, ds_val = train_test_split(
        ds, test_size=test_size, random_state=random_state
    )

    ds_train = Dataset.from_pandas(ds_train).cast_column('image', Image())

    ds_val = Dataset.from_pandas(ds_val).cast_column('image', Image())

    train_ds_prepared = ds_train.remove_columns(["__index_level_0__"])
    val_ds_prepared = ds_val.remove_columns(["__index_level_0__"])

    converted_ds_train = [_make_messages(example) for example in train_ds_prepared]
    converted_ds_val = [_make_messages(example) for example in val_ds_prepared]

    return converted_ds_train, converted_ds_val


class BaseQwenCollator(ABC):

    def __init__(self, processor):
        self.processor = processor

    def _prepare_inputs(self, examples, add_generation_prompt: bool):
        texts = [
                    self.processor.apply_chat_template(
                        example["messages"],
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt
                    )
                    for example in examples
                ]
        
        image_inputs = []

        for example in examples:
            img, _ = process_vision_info(example["messages"])
            image_inputs.append(img)
    
        batch = self.processor(
                    text=texts,
                    images=image_inputs,
                    return_tensors="pt",
                    padding=True
                )

        return batch

    @abstractmethod
    def __call__(self, examples):
        pass



class TrainQwenCollator(BaseQwenCollator):

    def __call__(self, examples):
        batch = self._prepare_inputs(examples=examples, add_generation_prompt=False)
        labels = batch['input_ids'].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100  
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
        labels[labels==image_token_id] = -100
        
        assistant_tokens = self.processor.tokenizer.encode("<|im_start|>assistant\n", return_tensors='pt')
        for b in range(len(labels)):
            for el_id in range(len(labels[b])):
                if el_id + 3 <= len(labels[b]) and torch.all(labels[b][el_id:el_id+3].cpu() == assistant_tokens):
                    labels[b][:el_id+3] = -100
                    break
        batch['labels'] = labels
        
        return batch

class EvalQwenCollator(BaseQwenCollator):

    def __call__(self, examples):
        return self._prepare_inputs(examples=examples, add_generation_prompt=True)