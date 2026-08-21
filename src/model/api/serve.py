import argparse
import logging
import os

import uvicorn


def main() -> None:
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Запуск FastAPI сервиса инференса Qwen2-VL")
    parser.add_argument("--config", default=None,
                        help="Путь к YAML-конфигу модели. Если не задан: CONFIG_PATH или дефолт")
    parser.add_argument("--host", default="0.0.0.0", help="Хост (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Порт (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Автоперезагрузка uvicorn для разработки")
    args = parser.parse_args()

    if args.config:
        os.environ["CONFIG_PATH"] = args.config

    uvicorn.run(
        "src.model.api.endpoint:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()