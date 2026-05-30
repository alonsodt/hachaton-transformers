# 📊 RESULTS — registro de experimentos

> Una fila por experimento. **Commitea SIEMPRE el RMSE local** antes de gastar un
> intento de subida. Ordenar por RMSE relativo medio ascendente.

## Marca de referencia — baselines (walk-forward, bloque de 252 días, 5 orígenes)

RMSE **relativo medio** = media entre los 6 índices de (RMSE / nivel del índice).
Se usa el relativo porque los índices van de ~10⁴ a ~10⁶ y el RMSE absoluto lo
dominarían Index_A/D.

| Baseline | RMSE rel. medio | Veredicto |
|----------|-----------------|-----------|
| **naive_flat** (último valor) | **≈ 8.6 %** | ✅ **base elegida** — suelo RMSE |
| ewma_30 (último valor suavizado) | ≈ 8.6 % | empate técnico con naive |
| drift_log / damped / blend | ≈ 900 %+ | ❌ explotan: la extrapolación log-lineal de tendencia se dispara en series con tendencia inestable |

**Conclusión:** a horizonte de 252 días, con índices volátiles y tendencias que se
revierten, el *forecast* plano (último valor) es el óptimo en RMSE. Cualquier modelo
de Fase 2 (features exógenas + GBM/red) debe **batir el ~8.6 % de naive_flat de forma
consistente** para merecer una subida.

## Envíos / experimentos

| Fecha | Autor | Modelo / config | RMSE local (rel.) | ¿Subido? | RMSE plataforma | Notas |
|-------|-------|-----------------|-------------------|----------|-----------------|-------|
| 2026-05-30 | (auto) | `naive_flat` | ≈ 8.6 % | ⬜ pendiente | — | **Primera base** → `submissions/submission_20260530_naive_flat.xlsx` |

## Ideas para batir la base (Fase 2)
1. **Drift acotado**: el drift actual explota; limitar la pendiente (cap percentil) o
   modelar **retornos log** en vez del nivel. Probar damped con `phi` bajo (0.90).
2. **Modelo con exógenas** (macro/network/news ya disponibles en el test): predecir
   **retorno acumulado** con `HistGradientBoostingRegressor` (sklearn, sin instalar
   nada) usando lags de exógenas + calendario. Los árboles NO extrapolan niveles →
   predecir retornos, no niveles.
3. **News**: sentimiento diario (lexicón) como feature de riesgo/momentum.
4. **Blend** del mejor modelo con naive_flat para regularizar a horizonte largo.
