import os

import httpx

class InferenceClient:


    def __init__(self, url: str):
        self.url = url

    async def generate(self, image: str):

        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:

            with open(image, 'rb') as image_file:
                files = {"file": (os.path.basename(image), image_file, "image/jpeg")}

                response = await client.post(
                    f"{self.url}/extract", files=files
                    )

        response.raise_for_status()

        return response.json()