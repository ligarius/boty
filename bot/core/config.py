"""Configuration management for the trading bot."""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    mode: str = Field("backtest", pattern="^(backtest|paper|live)$")
    binance_api_key: str | None = Field(default=None, env="BINANCE_API_KEY")
    binance_api_secret: str | None = Field(default=None, env="BINANCE_API_SECRET")
    binance_base_url: str = Field("https://fapi.binance.com", env="BINANCE_BASE_URL")
    leverage: float = Field(3.0, ge=1.0, le=20.0)
    risk_pct: float = Field(0.01, ge=0.001, le=0.05)
    max_dd_daily: float = Field(0.03, ge=0.0, le=0.5)
    max_dd_weekly: float = Field(0.08, ge=0.0, le=0.5)
    top_n_signals: int = Field(3, ge=1, le=10)
    universe: List[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "BNBUSDT"])
    timeframes: List[str] = Field(default_factory=lambda: ["1m", "5m", "15m"])
    redis_url: str = Field("redis://redis:6379/0", env="REDIS_URL")
    db_url: str = Field("postgresql+psycopg2://bot:bot@postgres:5432/bot", env="DB_URL")
    telegram_token: str | None = Field(default=None, env="TELEGRAM_TOKEN")
    telegram_chat_id: str | None = Field(default=None, env="TELEGRAM_CHAT_ID")
    prometheus_port: int = Field(9000, ge=1024, le=65535)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @field_validator("universe", "timeframes", mode="before")
    def split_csv(cls, value: str | List[str]) -> List[str]:  # type: ignore[override]
        if isinstance(value, str):
            return [token.strip() for token in value.split(",") if token.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance to avoid repeated parsing."""

    return Settings()
