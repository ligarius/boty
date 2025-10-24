# Plan integral de actualización del bot de trading

## 1. Objetivo general
Alinear el bot con metas de rentabilidad sostenida y control del riesgo, automatizando el ciclo de aprendizaje para que evolucione sin intervención manual constante.

## 2. Diversificación de estrategias
1. **Momentum refinado**
   - Ajustar `momentum_fast`, `momentum_slow` y `momentum_adx_period` con límites que privilegien configuraciones con ROI y profit factor positivos.
   - Incorporar filtros ATR para modular el tamaño de posición según volatilidad.
2. **Mean reversion Bollinger + RSI**
   - Parametrizar ventanas y umbrales para entradas en rangos laterales.
   - Implementar backtests cruzados con el dataset actual para evaluar sinergias con momentum.
3. **Estrategias adicionales**
   - Desarrollar al menos una estrategia basada en ruptura de volatilidad y otra en market-making simple con spreads dinámicos.
   - Documentar cada estrategia con su espacio de parámetros y condiciones de activación.

## 3. Meta-estrategia por ensemble
1. Consolidar señales en `EnsembleSelector`, ajustando el modelo de regresión logística con características de régimen (volatilidad, tendencia, liquidez).
2. Implementar ponderaciones dinámicas basadas en desempeño reciente (ROI, Sharpe) y estabilidad (drawdown).
3. Establecer reglas de fallback para desactivar estrategias con degradación continua y redistribuir peso automáticamente.

## 4. Automatización del ciclo de aprendizaje
1. **Pipeline diario/nocturno**
   - Orquestar con un scheduler (por ejemplo, Airflow o cron) la secuencia: ingesta de datos → tuning → validación fuera de muestra → despliegue.
   - Registrar cada corrida en PostgreSQL con parámetros, métricas y decisiones de despliegue.
2. **Tuning continuo**
   - Configurar `EvolutionaryTuner` para que promueva trials solo si superan umbrales `ROI > 0`, `profit_factor > 1`, `max_drawdown < 0.35`.
   - Habilitar pruning temprano usando el historial persistido para evitar gastar recursos en configuraciones deficientes.
3. **Validación fuera de muestra**
   - Reservar ventanas recientes de datos (al menos 20% del periodo) para pruebas no utilizadas en el tuning.
   - Comparar los resultados de validación contra la media móvil del desempeño histórico para garantizar mejoras reales.

## 5. Gestión de datos y métricas
1. Establecer pipelines de limpieza y normalización para cada fuente (Binance, CSV) con validaciones de calidad.
2. Construir un dashboard en Prometheus/Grafana con métricas clave: ROI rolling, drawdown, profit factor, latencia de ejecución y tasa de errores.
3. Integrar alertas automáticas (Slack/Email) cuando métricas críticas crucen umbrales definidos.

## 6. Gobernanza y control de riesgo
1. Definir políticas de capital máximo en riesgo por estrategia y por día.
2. Implementar stops programáticos y circuit breakers que pausen la operativa tras pérdidas consecutivas.
3. Auditar decisiones del bot almacenando logs detallados de señales, tamaño de posición y métricas en tiempo real.

## 7. Roadmap de implementación
1. **Semana 1-2**: Documentar estrategias actuales, definir nuevos KPIs, preparar infraestructura de datos.
2. **Semana 3-4**: Implementar y probar estrategias adicionales; ajustar ensemble con datasets históricos.
3. **Semana 5-6**: Desarrollar pipeline automatizado (scheduler, validación, despliegue controlado).
4. **Semana 7-8**: Integrar dashboards, alertas y políticas de riesgo; realizar prueba piloto en entorno simulado.
5. **Semana 9**: Revisión integral, ajustes finales y despliegue progresivo en producción con monitoreo reforzado.

## 8. Métricas de éxito
- ROI mensual positivo mantenido por al menos tres ciclos consecutivos.
- Profit factor ≥ 1.2 y drawdown máximo ≤ 0.3 en validaciones recientes.
- Tasa de decisiones automatizadas sin intervención > 80% con logs auditables.

Este plan traduce las recomendaciones previas en acciones concretas que modernizan el bot, fortalecen su robustez y reducen la dependencia de ajustes manuales.
