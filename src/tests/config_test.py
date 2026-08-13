import os

from ..model import ModelConfig, LoraConfig, DataConfig, QuantizationConfig, load_config

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "qwen2_2b.yaml",
)


def test_config():
    config = load_config(CONFIG_PATH)
    assert config.lora.r == 16
    assert type(config.lora.target_modules) is list
    assert 'q_proj' in config.lora.target_modules


def test_model_config():
    config = load_config(CONFIG_PATH)
    assert config.model.name == "Qwen/Qwen2-VL-2B-Instruct"
    assert config.model.torch_dtype == "bfloat16"


def test_lora_config():
    config = load_config(CONFIG_PATH)
    assert config.lora.enabled is True
    assert config.lora.r == 16
    assert config.lora.alpha == 32
    assert config.lora.dropout == 0.05
    assert config.lora.bias == "none"
    assert config.lora.task_type == "CAUSAL_LM"
    assert config.lora.target_modules == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
    ]


def test_data_config():
    config = load_config(CONFIG_PATH)
    assert config.data.train_path == "data/train"
    assert config.data.val_path == "data/val"


def test_training_config():
    config = load_config(CONFIG_PATH)
    assert config.training.output_dir == "outputs/qwen2vl-2b"
    assert config.training.num_train_epochs == 3
    assert config.training.learning_rate == 2.0e-5
    assert config.training.per_device_train_batch_size == 1
    assert config.training.per_device_eval_batch_size == 1
    assert config.training.gradient_accumulation_steps == 8
    assert config.training.logging_steps == 10
    assert config.training.save_strategy == "epoch"
    assert config.training.evaluation_strategy == "epoch"
    assert config.training.warmup_steps == 5


def test_quantization_config():
    config = load_config(CONFIG_PATH)
    assert config.quntization.enabled is True
    assert config.quntization.load_in_4bit is True
    assert config.quntization.quant_type == "nf4"
    assert config.quntization.compute_dtype == "float16"
    assert config.quntization.use_double_quant is True


def test_config_types():
    config = load_config(CONFIG_PATH)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.lora, LoraConfig)
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.quntization, QuantizationConfig)