import argparse

from src.training.config import load_config
from src.training.train import create_trainer


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

    trainer = create_trainer(config)

    trainer.train()
    trainer.save_model(config.training.output_dir)


if __name__ == "__main__":
    main()