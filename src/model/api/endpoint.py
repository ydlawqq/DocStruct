from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, UploadFile, File
from PIL import Image

from ..config import load_config
from ..inference.model import VLMModel
from .queue import InferenceQueue

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()

    model = VLMModel(config=config, quant=config.quantization.enabled)

    inference_queue = InferenceQueue(model=model)

    await inference_queue.start()

    app.state.inference_queue = inference_queue

    yield

    await inference_queue.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/extract")
async def extract(
    file: UploadFile = File(...)
):
    image_bytes = await file.read()

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    result = await app.state.inference_queue.submit(image)

    return {
        "result": result,
    }
