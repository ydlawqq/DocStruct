import logging
import os

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
import torch
from transformers import (AutoModelForCausalLM, AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig,
                           Qwen2VLForConditionalGeneration, EarlyStoppingCallback)
from src.model.config import Config, load_config
from PIL import Image
from ..prompts import EXTRACTION_PROMPT
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)


class VLMModel:

    def __init__(self, config: Config, quant: bool =True):
        self.config = config

     
        if quant and not torch.cuda.is_available():
            logger.warning("CUDA недоступна: квантование отключено, на CPU гружу float16")
            quant = False

        logger.info("Загрузка модели %s (quant=%s)...", self.config.model.name, quant)
        self.model = self._load_model(quant)
        self.model.eval()
        logger.info("Модель переведена в режим eval")

        logger.info("Загрузка процессора %s...", config.model.name)
        self.processor = AutoProcessor.from_pretrained(config.model.name, max_pixels=768*24*24)
        logger.info("Модель полностью загружена и готова к генерации")

    def _load_model(self, quant: bool = True):
        base_model = self._load_base_model(quant)

        # Абсолютный путь важнее относительного: в контейнере рабочая
        # директория может отличаться, а относительные пути из YAML
        # тогда ведут в никуда.
        adapters_dir = os.getenv("ADAPTERS_DIR", self.config.model.adapters_dir)
        logger.info("Загрузка адаптеров PEFT из %s...", adapters_dir)
        model = PeftModel.from_pretrained(base_model, adapters_dir)
        logger.info("Адаптеры PEFT загружены")
        return model

    def _load_base_model(self, quant: bool):
        if quant:
            q = self.config.quantization

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=q.load_in_4bit,
                bnb_4bit_quant_type=q.quant_type,
                bnb_4bit_use_double_quant=q.use_double_quant,
                bnb_4bit_compute_dtype=getattr(
                    torch,
                    q.compute_dtype,
                ),
            )
            logger.info("Загрузка базовой модели %s (с квантованием)...", self.config.model.name)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.config.model.name,
                quantization_config=quantization_config,
                device_map="auto",
            )
            logger.info("Базовая модель загружена (квантованная)")
            return model

        logger.info("Загрузка базовой модели %s (без квантования)...", self.config.model.name)

        torch_dtype = getattr(torch, self.config.model.torch_dtype)
        if not torch.cuda.is_available() and self.config.model.torch_dtype != "float16":
            logger.warning("CUDA недоступна: применяю torch_dtype=float16 для CPU-запуска")
            torch_dtype = torch.float16

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.model.name,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        logger.info("Базовая модель загружена")
        return model

    @torch.inference_mode()
    def generate(self, image: Image):
        logger.info("Начало Генерации")
        messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {"type": "text", "text": EXTRACTION_PROMPT},
        ],
    }
]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)

        inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors='pt')
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
        output_text = self.processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
        logger.info("Генерация завершена")
        return output_text[0]