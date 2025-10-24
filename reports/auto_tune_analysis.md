# Auto Tune Analysis for BTCUSDT Momentum Strategy

## Resumen Ejecutivo
- El mejor resultado obtenido tras 50 ensayos corresponde al ensayo 28 con una puntuación compuesta de **-19.73**, mejorando el valor base de **-103.31**.
- A pesar de la mejora respecto a la línea base, los indicadores siguen siendo negativos: ROI -29.55 %, ratio de Sharpe -5.82 y profit factor 0.75.
- La estrategia **no está lista para ir a producción** ("Go-live ready: False"), debido al desempeño general negativo y métricas de riesgo desfavorables.

## Interpretación de Métricas Clave
- **ROI (-29.55 %)**: Incluso el mejor conjunto de parámetros genera pérdidas acumuladas.
- **Sharpe (-5.82)**: La relación riesgo/retorno es altamente desfavorable, indicando que la volatilidad de los rendimientos no se compensa con beneficios.
- **Max Drawdown (34.58 %)**: Existieron caídas significativas en el capital, lo que implica riesgos de pérdida elevados.
- **Profit Factor (0.75)**: Los beneficios totales son inferiores a las pérdidas totales; cualquier operación seguiría destruyendo valor.
- **Win Rate (43.49 %)**: Menos de la mitad de las operaciones resultan ganadoras, lo que unido al profit factor < 1 refuerza la señal de que la estrategia no es rentable.

## Comparación con la Línea Base
- La optimización reduce el ROI negativo de -77.11 % a -29.55 % y mejora el profit factor de 0.40 a 0.75.
- El max drawdown también disminuye (de 77.24 % a 34.58 %), mostrando cierta mejora en la gestión del riesgo.
- Sin embargo, estas mejoras no alcanzan niveles de rentabilidad aceptables; la estrategia aún destruye capital.

## Conclusiones
1. **El bot no cumple el objetivo de generar ganancias**. A pesar de mejoras significativas respecto a la línea base, el desempeño sigue siendo claramente negativo.
2. **No se recomienda el despliegue en vivo** hasta lograr métricas positivas, especialmente ROI, Sharpe y profit factor superiores a 1.
3. **Próximos pasos sugeridos**:
   - Extender la optimización con más ensayos o diferentes rangos de parámetros, incluyendo ajustes a la ventana de selector y umbral de momentum.
   - Evaluar otras estrategias o incorporar filtros adicionales (por ejemplo, filtros de volatilidad o gestión dinámica de posición).
   - Analizar periodos de entrenamiento alternativos y validar en datos fuera de muestra para asegurar robustez.

## Parámetros Óptimos del Ensayo 28
- momentum_fast: 25
- momentum_slow: 84
- momentum_adx_period: 15
- mean_window: 71
- mean_z_threshold: 2.5310
- selector_threshold: 0.0551
- selector_window: 489
- selector_horizon: 10

## Métricas de Entrenamiento
- Accuracy: 0.5071
- F1: 0.5604

Estos valores sugieren que el modelo de clasificación subyacente tiene un desempeño apenas superior al azar, por lo que se recomienda revisar el pipeline de entrenamiento y la calidad de los datos etiquetados.
