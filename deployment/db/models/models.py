from sqlalchemy.orm import Mapped, mapped_column, relationship, DeclarativeBase
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey, text, Text, Time, TIMESTAMP, Enum, CheckConstraint, BigInteger
import datetime
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.ext.asyncio import AsyncEngine

class Base(DeclarativeBase):
    pass




class User(Base):
    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(unique=True, index=True)
    country: Mapped[str] = mapped_column(default='')
    firs_call: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(timezone=True), default=lambda : datetime.datetime.now(datetime.timezone.utc))

    receipts: Mapped[list["Receipt"]] = relationship(back_populates="user")


class Receipt(Base):
    __tablename__ = 'receipts'

    id: Mapped[int] = mapped_column(primary_key=True)
    time: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(timezone=True), default=lambda : datetime.datetime.now(datetime.timezone.utc))
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id',ondelete='CASCADE'))
    shop_name: Mapped[str] = mapped_column(default='')
    total: Mapped[float] = mapped_column(default=0.0)
    products: Mapped[dict] = mapped_column(JSON, default=dict)

    user: Mapped["User"] = relationship(back_populates="receipts")


async def run_models(engine: AsyncEngine):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


