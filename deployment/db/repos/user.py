from sqlalchemy.ext.asyncio import AsyncSession
from ..models.models import User
from sqlalchemy import update, select
from sqlalchemy.dialects.postgresql import insert
import datetime



class UserRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_user_at_first_call(self, **kwargs):
        stmt = insert(User).values(**kwargs).on_conflict_do_nothing(index_elements=[User.external_id])
        await self.session.execute(stmt)
        await self.session.commit()

    async def get_user_by_external_id(self, external_id: str):
        stmt = select(User).where(User.external_id == external_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

