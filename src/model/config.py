import os
from dataclasses import dataclass
from typing import Optional
import yaml
import torch

DEFAULT_CONFIG_PATH = "src/configs/qwen2_2b.yaml"


@dataclass
class ModelConfig:
    name: str
    adapters_dir: str 
    torch_dtype: str = "float16"


@dataclass
class LoraConfig:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    task_type: str = ''
    bias: str = 'none'
    target_modules: list[str] = None


@dataclass
class DataConfig:
    train_path: str
    val_path: str


@dataclass
class TrainingConfig:
    output_dir: str
    num_train_epochs: int = 3
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    logging_steps: int = 10
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    warmup_steps: int = 3
    optim: str = 'adamw_torch'
    weight_decay: float = 0.001
    scheduler_type: str = 'linear'
    metric_for_best_model: str | None = None

@dataclass
class QuantizationConfig:
    load_in_4bit: bool
    use_double_quant: bool
    enabled: bool = False
    quant_type: str = 'nf4'
    compute_dtype: str = 'float16'
    


@dataclass
class Config:
    model: ModelConfig
    lora: LoraConfig
    data: DataConfig
    training: TrainingConfig
    quantization: QuantizationConfig



def load_config(path: Optional[str] = None) -> Config:
    if path is None:
        path = os.getenv("CONFIG_PATH", DEFAULT_CONFIG_PATH)

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    return Config(
        model=ModelConfig(**data['model']),
        lora=LoraConfig(**data['lora']),
        data=DataConfig(**data['data']),
        training=TrainingConfig(**data['training']),
        quantization=QuantizationConfig(**data['quantization'])
    )