"""Tests for application settings configuration."""

from __future__ import annotations

from bot.core.config import Settings


def test_settings_reads_env_file(tmp_path, monkeypatch):
    """Settings should load values overridden in a local .env file."""

    env_file = tmp_path / ".env"
    env_file.write_text("MODE=paper\nLEVERAGE=5\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.mode == "paper"
    assert settings.leverage == 5


def test_settings_accepts_csv_lists(tmp_path, monkeypatch):
    """Raw CSV strings in the .env should become tokenized lists."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "UNIVERSE=BTCUSDT,ETHUSDT\nTIMEFRAMES=1m,15m\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.universe == ["BTCUSDT", "ETHUSDT"]
    assert settings.timeframes == ["1m", "15m"]
