from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_project_root = Path(__file__).resolve().parent.parent
_env_file = _project_root/".env"



class Settings(BaseSettings):

    DB_HOST: str
    DB_USER: str
    DB_PASS: str
    DB_PORT: str
    DB_NAME: str 


    model_config = SettingsConfigDict(

        env_file=_env_file,
        env_file_encoding='utf-8',
        case_sensitive='True',
        extra='ignore'

    )


    @property
    def db_aurl(self)->str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASS}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}" )

    @property
    def db_url(self)->str:
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASS}"
                f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}" )


settings = Settings()


