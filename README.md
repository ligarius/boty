# PulseForge

PulseForge es un bot de trading de futuros USDⓈ-M para Binance con arquitectura productiva mínima viable. Incluye estrategias momentum y reversión, meta-selección ML, backtesting con vectorbt, ejecución distribuida con Celery y observabilidad vía Prometheus.

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
SELECTOR_THRESHOLD=0.05
SELECTOR_HORIZON=5
SELECTOR_WINDOW=100
UNIVERSE=BTCUSDT,ETHUSDT,BNBUSDT
TIMEFRAMES=1m,5m,15m
DB_URL=postgresql+psycopg2://bot:bot@postgres:5432/bot
REDIS_URL=redis://redis:6379/0
TELEGRAM_TOKEN=
TELEGRAM_CHAT_ID=
```

## Ejecución con Docker

Levantar toda la plataforma (API, worker Celery, Redis, Postgres y Prometheus):

```bash
docker compose up -d
```

Comandos útiles para operar los contenedores:

```bash
# Ver estado general
docker compose ps

# Seguir logs del API FastAPI
docker compose logs -f api

# Seguir logs del worker de señales
docker compose logs -f worker

# Reiniciar componentes individuales
docker compose restart api
docker compose restart worker
```

Una vez levantados, la API queda disponible en `http://localhost:8000` y Prometheus en `http://localhost:9090`.

### Ejecución local (sin Docker)

Si prefieres correrlo en tu máquina sin contenedores:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Variables de entorno desde .env
export $(grep -v '^#' .env | xargs)

# API FastAPI
uvicorn bot.api.main:app --host 0.0.0.0 --port 8000

# Worker Celery en otra terminal
celery -A bot.exec.celery_app.celery_app worker --loglevel=info
```

> Nota: Los scripts CLI aceptan `PYTHON_BIN=/ruta/a/python` para forzar un intérprete específico. Si las dependencias no están instaladas localmente, intentarán ejecutarse automáticamente dentro de los contenedores (`docker compose exec api/worker`). Tanto `run_python_module` como `run_celery_worker` capturan `stdout`/`stderr` para detectar `ImportError`/`ModuleNotFoundError` (o exit codes 127) y rehúsan automáticamente hacia los contenedores correspondientes.

## Monitoreo y operación

- **Documentación interactiva:** abrir `http://localhost:8000/docs`
- **Salud básica:** `curl http://localhost:8000/health`
- **Estado de equity y drawdown:** `curl http://localhost:8000/status`
- **Reporte diario en CSV:** `curl http://localhost:8000/report/daily`
- **Cambio de modo:** `curl -X POST http://localhost:8000/mode -H 'Content-Type: application/json' -d '{"mode": "paper"}'`
- **Pausar/Reanudar señales:** `curl -X POST http://localhost:8000/pause` / `curl -X POST http://localhost:8000/resume`
- **Actualizar universo:** `curl -X POST http://localhost:8000/universe -H 'Content-Type: application/json' -d '{"symbols": "BTCUSDT,ETHUSDT"}'`
- **Ajustar riesgo:** `curl -X POST http://localhost:8000/risk -H 'Content-Type: application/json' -d '{"risk_pct": 0.02}'`
- **Métricas Prometheus:** `curl http://localhost:8000/metrics`
- **Dashboards Prometheus:** abrir `http://localhost:9090` y crear queries sobre los exportadores disponibles.

Para ejecutar chequeos manuales o tareas recurrentes:

```bash
# Backtest con rango de fechas
bot/scripts/run_backtest.sh BTCUSDT 1m 2023-01-01 2023-02-01

# Worker en modo paper trading (Celery)
bot/scripts/run_paper.sh

# Validación previa a live
bot/scripts/run_live.sh

# Generar reporte diario
bot/scripts/report_daily.sh
```

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
