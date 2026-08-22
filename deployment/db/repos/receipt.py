from sqlalchemy.ext.asyncio import AsyncSession
from ..models.models import User, Receipt
from sqlalchemy import update, select
from sqlalchemy.dialects.postgresql import insert
import datetime



class ReceiptRepo:
    def __init__(self, session: AsyncSession):
        self.session = session


    async def add_receipt(self, **kwargs):
        stmt = insert(Receipt).values(**kwargs)
        await self.session.execute(stmt)
        await self.session.commit()


    async def get_receipts_by_user_id(self, user_id):
        stmt = select(Receipt).where(Receipt.user_id == user_id)
        result = await self.session.execute(stmt)
        receipts = result.scalars().all()
        return receipts

    