"""Configuration management for the trading bot."""
from __future__ import annotations

import json
from json import JSONDecodeError
from functools import lru_cache
from typing import Any, List

from pydantic import Field, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from pydantic_settings.sources import DotEnvSettingsSource, ForceDecode, NoDecode


def _json_loads(value: str) -> Any:
    """Parse JSON values, returning the raw string if parsing fails."""

    try:
        return json.loads(value)
    except JSONDecodeError:
        return value


class _SafeDotEnvSettingsSource(DotEnvSettingsSource):
    """Dotenv settings source that tolerates non-JSON values."""

    def decode_complex_value(self, field_name: str, field: FieldInfo, value: Any) -> Any:
        if field and (
            NoDecode in field.metadata
            or (self.config.get("enable_decoding") is False and ForceDecode not in field.metadata)
        ):
            return value

        return _json_loads(value)


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    mode: str = Field("backtest", pattern="^(backtest|paper|live)$")
    binance_api_key: str | None = Field(default=None)
    binance_api_secret: str | None = Field(default=None)
    binance_base_url: str = Field("https://fapi.binance.com")
    default_data_source: str = Field("binance", pattern="^(binance|synthetic|csv)$")
    data_source_csv_path: str | None = Field(default=None)
    leverage: float = Field(3.0, ge=1.0, le=20.0)
    risk_pct: float = Field(0.01, ge=0.001, le=0.05)
    max_dd_daily: float = Field(0.03, ge=0.0, le=0.5)
    max_dd_weekly: float = Field(0.08, ge=0.0, le=0.5)
    top_n_signals: int = Field(3, ge=1, le=10)
    universe: List[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "BNBUSDT"])
    timeframes: List[str] = Field(default_factory=lambda: ["1m", "5m", "15m"])
    redis_url: str = Field("redis://redis:6379/0")
    db_url: str = Field("postgresql+psycopg2://bot:bot@postgres:5432/bot")
    telegram_token: str | None = Field(default=None)
    telegram_chat_id: str | None = Field(default=None)
    prometheus_port: int = Field(9000, ge=1024, le=65535)
    selector_threshold: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description="Probability threshold applied to ML selector scores.",
    )
    selector_horizon: int = Field(
        5,
        ge=1,
        le=50,
        description="Forward return horizon (in bars) used to label training samples.",
    )
    selector_window: int = Field(
        100,
        ge=10,
        le=2000,
        description="Rolling window size (in samples) for walk-forward selector training.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        safe_dotenv = _SafeDotEnvSettingsSource(
            settings_cls=settings_cls,
            env_file=getattr(dotenv_settings, "env_file", None),
            env_file_encoding=getattr(dotenv_settings, "env_file_encoding", None),
            env_prefix=getattr(dotenv_settings, "env_prefix", None),
            env_nested_delimiter=getattr(dotenv_settings, "env_nested_delimiter", None),
            env_nested_max_split=getattr(dotenv_settings, "env_nested_max_split", None),
            env_ignore_empty=getattr(dotenv_settings, "env_ignore_empty", None),
            env_parse_none_str=getattr(dotenv_settings, "env_parse_none_str", None),
            env_parse_enums=getattr(dotenv_settings, "env_parse_enums", None),
        )
        return init_settings, env_settings, safe_dotenv, file_secret_settings

    @field_validator("universe", "timeframes", mode="before")
    def split_csv(cls, value: str | List[str]) -> List[str]:  # type: ignore[override]
        if isinstance(value, str):
            return [token.strip() for token in value.split(",") if token.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance to avoid repeated parsing."""

    return Settings()
