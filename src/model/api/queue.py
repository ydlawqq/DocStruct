import asyncio
from ..inference.model import VLMModel
from PIL import Image
from dataclasses import dataclass


@dataclass
class InferenceJob:
    image: Image.Image
    future: asyncio.Future

class InferenceQueue:
    def __init__(self, model: VLMModel):
        self.model = model
        self.queue = asyncio.Queue()
        self.worker_task: asyncio.Task | None = None

    async def start(self):
        self.worker_task = asyncio.create_task(self._worker())

    async def stop(self):
        if self.worker_task:
            self.worker_task.cancel()

            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
    async def submit(self, image: Image.Image):
        loop = asyncio.get_running_loop()

        future = loop.create_future()

        job = InferenceJob(image=image, future=future)
        await self.queue.put(job)

        return await future

    async def _worker(self):
        while True:
            job = await self.queue.get()

            try:
                result = await asyncio.to_thread(self.model.generate , job.image)
                job.future.set_result(result)

            except Exception as exc:
                job.future.set_exception(exc)

            finally:
                self.queue.task_done()


