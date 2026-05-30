# 📊 RESULTS — registro de experimentos

> Una fila por experimento. **Commitea SIEMPRE el RMSE local** antes de gastar un
> intento de subida. Ordenar por **MACRO absoluto** ascendente.

## ⚠️ LA MÉTRICA (lección de la 1ª subida — leer antes de tocar nada)

El leaderboard puntúa **`rmse_macro` = media del RMSE ABSOLUTO de los 6 índices**.
Verificado contra los números de la 1ª subida:
`mean(rmse_index_a..f) = (176514+41342+8749+199854+15779+3259)/6 ≈ 74250 ≈ rmse_macro (74197)`.

**Index_A e Index_D concentran ~85% de la puntuación** (sus niveles son ~2·10⁶):

| Índice | RMSE plataforma (naive) | % del score |
|--------|------------------------|-------------|
| Index_D | 199.854 | 45% |
| Index_A | 176.514 | 40% |
| Index_B | 41.342 | 9% |
| resto (C,E,F) | <16k c/u | 6% |

👉 Hay que optimizar el **RMSE ABSOLUTO** y vigilar **A y D**. El error inicial fue
ordenar por RMSE *relativo* (normalizado por nivel) y subir `naive_flat`, que es de
los **peores** modelos para la métrica real.

Además: la validación local es **conservadora** (naive local A≈268k vs plataforma
176k) → **no hay overfitting**, el backtest es de fiar.

## Marca de referencia — backtest FIEL (252 días NATURALES, 24 orígenes, RMSE absoluto)

`python -m src.run_baselines`

| Modelo | Index_A | Index_D | A+D | MACRO | MACRO recientes | Veredicto |
|--------|--------:|--------:|----:|------:|----------------:|-----------|
| **drift_full** | 66.952 | 73.096 | 140.048 | **27.648** | **94.861** | ✅ **MEJOR** — drift log historia completa |
| blend_full_0.7 | 69.590 | 75.727 | 145.318 | 28.489 | 98.251 | cobertura anti-crash |
| damped_252_098 | 73.503 | 80.057 | 153.560 | 30.349 | 101.191 | |
| naive_flat | 79.638 | 85.393 | 165.032 | 32.330 | 108.146 | suelo (1ª subida) |
| drift_log_252 | 82.767 | 92.638 | 175.405 | 35.122 | 129.594 | ❌ peor: ventana corta se dispara |

**Conclusión:** `drift_full` baja el MACRO un **−14,5%** (y −12% en los orígenes
recientes), ganando a naive en A en **17/24** orígenes y en D en **16/24**. El drift
de historia completa (~+12,8%/año en A y D) es estable; el de ventana corta (252d)
es el que "explota" — por eso parecía que *todo* drift era malo. Para una serie con
drift positivo, el óptimo de RMSE es `E[Pₜ₊ₕ]=Pₜ·exp(μh) > último valor`, así que
`naive_flat` queda **sesgado a la baja** justo en A y D.

## Envíos / experimentos

| Fecha | Autor | Modelo / config | RMSE local (MACRO) | ¿Subido? | RMSE plataforma | Notas |
|-------|-------|-----------------|--------------------|----------|-----------------|-------|
| 2026-05-30 | (auto) | `naive_flat` | 32.330 | ✅ sí | **74.197** | 1ª base. Métrica equivocada (relativo) |
| 2026-05-30 | nosotros | `drift_full` | **27.648** | ⬜ pendiente | — | `submissions/submission_20260530_drift_full.xlsx`. Esperado ≈ 63–66k |

## Ideas para seguir batiendo (Fase 2)
1. **Exógenas para A y D** (donde se gana/pierde): `HistGradientBoostingRegressor`
   prediciendo **retorno log acumulado** a 252d con lags de macro/network (árboles
   NO extrapolan niveles → predecir retornos, no niveles). Solo merece subir si
   bate a `drift_full` en MACRO **y** en A+D de forma consistente.
2. **Calibrar el drift**: probar drift con media-vida (damped) sobre historia
   completa, o mezclar drift_full con la tasa de los últimos ~5 años.
3. **Blend** drift_full × exógenas para regularizar a horizonte largo.
4. Cuidado: cualquier mejora se mide con `src.run_baselines` (MACRO absoluto), nunca
   a ojo ni por relativo.
