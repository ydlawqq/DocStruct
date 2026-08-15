import argparse

from src.model.config import load_config
from src.training.train import create_trainer
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    ADAPTER_PATH = PROJECT_ROOT / config.training.output_dir

    trainer = create_trainer(config)

    logging.info("Запуск обучения")
    trainer.train()

    logging.info("Обучение завершено, адаптер сохранен по пути: %s...", ADAPTER_PATH)
    trainer.save_model(str(ADAPTER_PATH))


if __name__ == "__main__":
    main()