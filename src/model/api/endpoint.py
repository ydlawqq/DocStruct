from contextlib import asynccontextmanager
from io import BytesIO

import json

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

from ..config import load_config
from ..inference.model import VLMModel
from .queue import InferenceQueue

import logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()

    logger.info('Модель загружается')
    model = VLMModel(config=config, quant=config.quantization.enabled)

    inference_queue = InferenceQueue(model=model)

    await inference_queue.start()

    app.state.inference_queue = inference_queue
    app.state.is_model_ready = True
    logger.info('Статус модели: %s', getattr(app.state, "is_model_ready", "Не загружена"))

    yield

    await inference_queue.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    if not getattr(app.state, "is_model_ready", False):
        raise HTTPException(status_code=503, detail='loading model')
    return {"status": "ok"}


@app.post("/extract")
async def extract(
    file: UploadFile = File(...)
):
    logger.debug('Пришел запрос')
    logger.info('ПЗ')
    image_bytes = await file.read()

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    result = await app.state.inference_queue.submit(image)

    return {
        "result": json.loads(result),
    }
