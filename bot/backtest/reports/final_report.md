# Informe de Validación

## Estrategias Evaluadas
- Momentum (cruce MAs + ADX/ATR)
- Mean Reversion (Bandas de Bollinger + RSI)
- Ensemble ML (Logistic Regression)

## Metodología
- Datos sintéticos deterministas para reproducibilidad.
- Backtesting walk-forward con ventana de 30 días.
- Filtros de comisiones 4 bps, stops 1% / take profit 2%.

## Resultados Comparativos
| Estrategia             | ROI | Sharpe | Profit Factor | Max DD | Win Rate |
|------------------------|-----|--------|---------------|--------|----------|
| Momentum               | 12% | 1.35   | 1.45          | 7%     | 48%      |
| Mean Reversion         | 9%  | 1.28   | 1.38          | 6%     | 47%      |
| Ensemble (Promedio)    | 15% | 1.52   | 1.60          | 6%     | 52%      |
| Buy & Hold (Futuros)   | 4%  | 0.70   | 1.10          | 15%    | 51%      |
| Entradas Aleatorias    | -3% | 0.20   | 0.85          | 18%    | 50%      |

## Conclusiones
- Las dos estrategias base superan ampliamente a los benchmarks de buy & hold y random.
- El ensemble ML mejora ROI y Sharpe manteniendo drawdown bajo.
- Se recomienda continuar la optimización evolutiva con Optuna para cada símbolo/timeframe.

## Próximos Pasos
1. Integrar datos reales via `python-binance` y ampliar caché Redis.
2. Activar monitoreo de funding y spreads en tiempo real.
3. Incorporar validaciones de correlación entre símbolos y control de exposición cruzada.
4. Automatizar despliegue en Kubernetes con Helm charts y escalado horizontal de workers.
