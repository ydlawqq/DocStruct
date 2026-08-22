import json
import logging
import os

import chainlit as cl
from dotenv import load_dotenv

from client import InferenceClient
from db.connector import engine, session
from db.models.models import run_models
from db.repos.receipt import ReceiptRepo
from db.repos.user import UserRepo

import json

load_dotenv()

logger = logging.getLogger(__name__)

model_url = os.getenv('MODEL_URL')
client = InferenceClient(model_url)


def _parse_total(value) -> float:
    if not value:
        return 0.0
    try:
        return float(str(value).replace(" ", "").replace(",", "."))
    except (ValueError, TypeError):
        return 0.0


def _get_external_id() -> str:
    user = cl.user_session.get("user")
    return getattr(user, "identifier", "anonymous")


@cl.on_chat_start
async def start():
    try:
        await run_models(engine)
    except Exception:
        logger.exception("Не удалось создать таблицы в БД")

    external_id = _get_external_id()
    try:
        async with session() as db_session:
            repo = UserRepo(db_session)
            await repo.add_user_at_first_call(external_id=external_id, country="")
    except Exception:
        logger.exception("Не удалось сохранить пользователя в БД")

    await cl.Message(content='Привет, пока я умею только обрабатывать чеки, в jpg формате, но я только учусь!!!').send()


@cl.on_message
async def on_message(message: cl.Message):
    image = next(
        (element for element in message.elements if isinstance(element, cl.Image)), None
    )

    if not image:
        await cl.Message(content='Пришли фотографию!').send()
        return

    result = await client.generate(image=image.path)

    data = json.loads(result.get("result", result))

    external_id = _get_external_id()
    try:
        async with session() as db_session:
            user_repo = UserRepo(db_session)
            db_user = await user_repo.get_user_by_external_id(external_id)

            if db_user is None:
                await user_repo.add_user_at_first_call(external_id=external_id, country="")
                db_user = await user_repo.get_user_by_external_id(external_id)

            if db_user is not None:
                receipt_repo = ReceiptRepo(db_session)
                await receipt_repo.add_receipt(
                    user_id=db_user.id,
                    shop_name=data.get("store_name", ""),
                    total=_parse_total(data.get("total")),
                    products=data,
                )
    except Exception:
        logger.exception("Не удалось сохранить чек в БД")

    await cl.Message(
        content=f"Ваша покупка была сохранена, как: {json.dumps(data, ensure_ascii=False)}"
    ).send()