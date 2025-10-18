"""Tests for dashboard-related API endpoints."""

import pytest
from fastapi.testclient import TestClient

from bot.api.main import app, settings


@pytest.fixture(autouse=True)
def restore_mode():
    """Ensure application mode is restored after each test."""

    original_mode = settings.mode
    try:
        yield
    finally:
        settings.mode = original_mode  # type: ignore[misc]


def test_post_mode_accepts_valid_modes():
    client = TestClient(app)

    settings.mode = 'backtest'  # type: ignore[misc]
    for mode in ('backtest', 'paper'):
        response = client.post('/mode', json={'mode': mode})
        assert response.status_code == 200
        assert response.json() == {'mode': mode}
        assert settings.mode == mode

    settings.mode = 'live'  # type: ignore[misc]
    response = client.post('/mode', json={'mode': 'live'})
    assert response.status_code == 200
    assert response.json() == {'mode': 'live'}
    assert settings.mode == 'live'


def test_post_mode_rejects_invalid_mode():
    client = TestClient(app)

    settings.mode = 'backtest'  # type: ignore[misc]
    response = client.post('/mode', json={'mode': 'scalping'})
    assert response.status_code == 400
    assert response.json() == {'detail': 'invalid mode'}
    assert settings.mode == 'backtest'


def test_post_mode_rejects_live_when_locked():
    client = TestClient(app)

    settings.mode = 'paper'  # type: ignore[misc]
    response = client.post('/mode', json={'mode': 'live'})
    assert response.status_code == 403
    assert response.json() == {'detail': 'Live mode locked until validation'}
    assert settings.mode == 'paper'
