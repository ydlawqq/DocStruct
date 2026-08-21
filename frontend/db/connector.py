from .database import settings

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

sync_engine = create_engine(url=settings.db_url)

engine = create_async_engine(url=settings.db_aurl, pool_size=5, max_overflow=10, echo=False)


session = async_sessionmaker(engine)



