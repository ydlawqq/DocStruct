from .data.prepare_dataset import prepare_datasets, TrainQwenCollator, EvalQwenCollator
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch
from transformers import (AutoModelForCausalLM, AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig,
                           Qwen2VLForConditionalGeneration, EarlyStoppingCallback)
from src.model.config import Config
from .utils import JSONTrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_trainer(config: Config):
    bnb_config = None

    if config.quantization.enabled:
        q = config.quantization

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=q.load_in_4bit,
            bnb_4bit_quant_type=q.quant_type,
            bnb_4bit_compute_dtype=getattr(torch, q.compute_dtype),
            bnb_4bit_use_double_quant=q.use_double_quant
        )

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.model.name, quantization_config=bnb_config, device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(config.model.name, max_pixels=768*24*24)

    base_model = prepare_model_for_kbit_training(base_model)

    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        task_type=getattr(TaskType, config.lora.task_type),
    )

    lora_model = get_peft_model(base_model, lora_config)

    lora_model.config.use_cache = False
    lora_model.gradient_checkpointing_enable()
    lora_model.enable_input_require_grads()

    converted_ds_train, converted_ds_val = prepare_datasets()

    train_collator = TrainQwenCollator(processor=processor)
    eval_collator = EvalQwenCollator(processor=processor)

    early_stopping = EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.1)

    trainer = JSONTrainer(
        model=lora_model,
        processing_class=processor.tokenizer,
        data_collator=train_collator,
        train_dataset=converted_ds_train,
        eval_dataset=converted_ds_val,
        eval_data_collator=eval_collator,
        args=SFTConfig(
            per_device_train_batch_size=config.training.per_device_train_batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            warmup_steps=config.training.warmup_steps,
            num_train_epochs=config.training.num_train_epochs,
            learning_rate=config.training.learning_rate,
            logging_steps=config.training.logging_steps,
            eval_strategy=config.training.evaluation_strategy,
            save_strategy=config.training.save_strategy,
            optim=config.training.optim,
            weight_decay=config.training.weight_decay,
            lr_scheduler_type=config.training.scheduler_type,
            seed=3407,
            output_dir=config.training.output_dir,
            metric_for_best_model=config.training.metric_for_best_model,
            greater_is_better=True,
            load_best_model_at_end=True,
            report_to='none',
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={'skip_prepare_dataset': True},
            max_length=2048,
        ),
        callbacks=[early_stopping],
    )

    return trainer