import json
import logging
import os

import pytest
from PIL import Image, ImageDraw, ImageFont

from src.model.config import load_config
from src.model.inference.model import VLMModel
from datasets import load_dataset

# Настройка логирования для теста, чтобы в консоли pytest было видно прогресс загрузки модели
logger = logging.getLogger(__name__)


def setup_module(module):
    """Настраивает логирование для тестового модуля."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# Пути от корня проекта (src/tests/ -> корень)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "src", "configs", "qwen2_2b.yaml")





@pytest.mark.slow
def test_generate_returns_valid_json():
    test_image = load_dataset("mychen76/ds_receipts_v2_test", split="train[:1]")['image']
    image = test_image[0]

    """Проверяет, что генерация возвращает валидный JSON с ожидаемой структурой."""
    config = load_config(CONFIG_PATH)

    # Убеждаемся, что адаптеры существуют
    adapters_path = os.path.join(PROJECT_ROOT, config.model.adapters_dir)
    assert os.path.isdir(adapters_path), (
        f"Адаптеры не найдены по пути: {adapters_path}. "
        "Загрузите их в adapters/qwen2vl-2b"
    )

    logger.info("Создание модели...")
    model = VLMModel(config, quant=False)
    logger.info("Модель создана")

    output = model.generate(image)
    logger.info("Вывод модели:\n%s", output)

    # Результат должен быть непустой строкой
    assert isinstance(output, str)
    assert len(output.strip()) > 0, "Модель вернула пустую строку"

    # Результат должен быть валидным JSON
    try:
        data = json.loads(output)
    except json.JSONDecodeError as e:
        pytest.fail(f"Результат не является валидным JSON: {e}\nВывод модели: {output}")

    # Если модель вернула JSON, обёрнутый в кавычки — распарсить повторно
    if isinstance(data, str):
        logger.info("Модель вернула JSON-строку, распарсиваю повторно:\n%s", data)
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            pytest.fail(f"Вложенный JSON не является валидным: {e}\nВывод модели: {output}")

    # Результат должен быть JSON-объектом
    assert isinstance(data, dict), (
        f"Модель вернула не JSON-объект, а {type(data).__name__}: {output}"
    )

    # Проверяем наличие всех ожидаемых ключей
  

    # line_items должен быть списком
    assert isinstance(data["line_items"], list), "line_items должен быть списком"
