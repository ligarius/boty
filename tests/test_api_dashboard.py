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


def test_post_backtest_returns_metrics():
    client = TestClient(app)

    payload = {
        'symbol': 'BTCUSDT',
        'timeframe': '1m',
        'start': '2024-01-01T00:00:00',
        'end': '2024-01-02T00:00:00',
    }
    response = client.post('/backtest', json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data['symbol'] == payload['symbol']
    assert data['timeframe'] == payload['timeframe']
    assert data['start'] == payload['start']
    assert data['end'] == payload['end']
    assert isinstance(data['go_live_ready'], bool)
    metrics = data['metrics']
    for key in ('roi', 'sharpe', 'profit_factor', 'win_rate', 'max_drawdown'):
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_post_backtest_validates_input():
    client = TestClient(app)

    response = client.post('/backtest', json={'symbol': 'BTC', 'timeframe': '1m'})
    assert response.status_code == 400
    assert response.json()['detail'] == 'symbol, timeframe, start and end are required'

    response = client.post(
        '/backtest',
        json={
            'symbol': 'BTCUSDT',
            'timeframe': '1m',
            'start': '2024-01-02T00:00:00',
            'end': '2024-01-01T00:00:00',
        },
    )
    assert response.status_code == 400
    assert response.json()['detail'] == 'end must be after start'


def test_post_live_guard_requires_live_mode():
    client = TestClient(app)

    settings.mode = 'paper'  # type: ignore[misc]
    response = client.post('/live/guard')
    assert response.status_code == 400
    assert response.json()['detail'] == 'Live mode not enabled'


def test_post_live_guard_returns_metrics_when_live():
    client = TestClient(app)

    settings.mode = 'live'  # type: ignore[misc]
    response = client.post('/live/guard')
    assert response.status_code == 200
    data = response.json()
    assert data['mode'] == 'live'
    assert data['symbol'] == 'BTCUSDT'
    assert isinstance(data['go_live_ready'], bool)
    metrics = data['metrics']
    for key in ('roi', 'sharpe', 'profit_factor', 'win_rate', 'max_drawdown'):
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_get_report_daily_returns_serialized_metrics():
    client = TestClient(app)

    response = client.get('/report/daily')
    assert response.status_code == 200
    data = response.json()
    assert data['symbol'] == 'BTCUSDT'
    assert data['timeframe'] == '1m'
    metrics = data['metrics']
    assert set(metrics.keys()) == {'roi', 'sharpe', 'profit_factor', 'win_rate', 'max_drawdown'}
    assert isinstance(data['chart_csv'], str)
    assert data['chart_csv'].startswith('timestamp,close')
