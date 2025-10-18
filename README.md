# Prompt Maestro

Prompt Maestro es un bot de trading de futuros USDⓈ-M para Binance con arquitectura productiva mínima viable. Incluye estrategias momentum y reversión, meta-selección ML, backtesting con vectorbt, ejecución distribuida con Celery y observabilidad vía Prometheus.

## Requisitos
- Docker y docker compose
- Python 3.11 (si se ejecuta sin contenedores)

## Configuración
Copiar `.env.example` a `.env` y completar variables:

```
BINANCE_API_KEY=
BINANCE_API_SECRET=
BINANCE_BASE_URL=https://fapi.binance.com
MODE=backtest
LEVERAGE=3
RISK_PCT=0.01
MAX_DD_DAILY=0.03
MAX_DD_WEEKLY=0.08
TOP_N_SIGNALS=3
UNIVERSE=BTCUSDT,ETHUSDT,BNBUSDT
TIMEFRAMES=1m,5m,15m
DB_URL=postgresql+psycopg2://bot:bot@postgres:5432/bot
REDIS_URL=redis://redis:6379/0
TELEGRAM_TOKEN=
TELEGRAM_CHAT_ID=
```

## Ejecución con Docker

```bash
docker compose up -d
```

Endpoints disponibles:
- `GET /docs`
- `GET /status`
- `GET /report/daily`

## Scripts CLI

```bash
# Backtest
bot/scripts/run_backtest.sh BTCUSDT 1m 2023-01-01 2023-02-01
# Paper trading (worker Celery)
bot/scripts/run_paper.sh
# Validación antes de live
bot/scripts/run_live.sh
# Reporte diario
bot/scripts/report_daily.sh
```

> Nota: Los scripts CLI aceptan `PYTHON_BIN=/ruta/a/python` para forzar un intérprete específico, con fallback automático a `python3` o `python` si no se define.

## Migraciones y base de datos
El repositorio utiliza SQLAlchemy para gestionar tablas `signals` y `trades`. Al levantar el contenedor o instanciar `Repository`, las tablas se crean automáticamente.

## Backtesting y validación
- Motor vectorbt (`bot/backtest/engine.py`).
- Métricas de éxito: Sharpe ≥ 1.2, Profit Factor ≥ 1.3, MaxDD ≤ 8%, Win Rate ≥ 45%.
- `bot/scripts/run_backtest_cli.py` imprime métricas y determina si se puede habilitar modo live.

## Observabilidad
- Logs JSON (`bot/obs/logging.py`).
- Métricas Prometheus (`/metrics`).
- Configuración Prometheus en `bot/infra/prometheus.yml`.

## Tests

```bash
pytest
```

## Checklist antes de Live
- [ ] Configurar claves API en `.env` y habilitar IPs en Binance.
- [ ] Ejecutar `bot/scripts/run_backtest.sh` y verificar métricas superiores a los mínimos.
- [ ] Correr al menos 72 horas en modo paper (`bot/scripts/run_paper.sh`) sin drawdown > 8%.
- [ ] Revisar alertas y métricas Prometheus.
- [ ] Confirmar stops y reduce-only habilitados en Binance.
- [ ] Actualizar `MODE=live` y ejecutar `bot/scripts/run_live.sh`.
