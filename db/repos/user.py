from sqlalchemy.ext.asyncio import AsyncSession
from ..models.models import User
from sqlalchemy import update, select
from sqlalchemy.dialects.postgresql import insert
import datetime



class UserRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_user_at_first_call(self, **kwargs):
        id = kwargs.get('id')
        stmt = insert(User).values(**kwargs).on_conflict_do_nothing(index_elements=[User.id])
        await self.session.execute(stmt)

