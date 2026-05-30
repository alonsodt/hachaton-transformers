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
| **drift_blend_5y_0.7** | 66.668 | 72.722 | 139.390 | **27.554** | **94.787** | ✅ **NUEVO MEJOR** — drift estructural + 30% régimen 5y |
| drift_full | 66.952 | 73.096 | 140.048 | 27.648 | 94.861 | drift log historia completa (envío 2º, ya validado) |
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
| 2026-05-30 | nosotros | `drift_blend_5y_0.7` | **27.554** | ⬜ pendiente | — | Calibración A/D: drift estructural + 30% régimen 5y. Mejora limpia (A, D, MACRO y recent ↓) pero **marginal (−0,3%)**. `submissions/submission_20260530_drift_blend_5y_0.7.xlsx` |

### Calibración de drift A/D (Fase 2) — conclusión

Mezclar la tasa estructural (43 años) con la del régimen reciente, ancladas al
último precio: `slope = w·full + (1−w)·recent`. Barrido en `src.run_baselines`:

- **Ganador: `drift_blend_5y_0.7`** (ventana 5y, 70% estructural). Bate a drift_full
  en A, D, MACRO **y** MACRO_recent → cumple la regla anti-overfit.
- Patrón **monótono e interpretable** (no es suerte): 5y_0.7 > 5y_0.5 > full; las
  variantes de 3 años o recent-puro **empeoran** (extrapolan el tramo caliente).
- Mezcla óptima por-índice = 27.472 (solo −0,3% extra) metiendo ewma/blend en
  B/C/F → **descartada por overfit** en índices que pesan 6%.
- **Veredicto operativo:** mejora real pero pequeña. NO merece un intento dedicado;
  llevarla como 2º envío en lugar de drift_full (es gratis y estrictamente mejor).

## Fase 3 — Redes neuronales y Transformer (HECHO; ver `deep_20260530.md`)

`python -m src.deep_models` — DLinear + Transformer (con positional encoding),
forecast DIRECTO del drift, walk-forward fiel, comparados vs drift_full.

| Modelo | MACRO (6 oríg.) | vs drift_full | gana a drift_full |
|--------|----------------:|--------------:|-------------------|
| ens_drift+tr | 78.384 | −0,0% | 4/6 |
| **drift_full** | 78.397 | — | — |
| transformer | 78.726 | +0,4% | 4/6 |
| dlinear | 80.542 | +2,7% | 1/6 |
| naive_flat | 91.439 | +16,6% | 0/6 |

**Veredicto: las redes NO baten a `drift_full` de forma fiable** (empate técnico;
pierden en el origen volátil). Confirma el aviso DLinear. **Se mantiene `drift_full`
como envío.** No subir la red.

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
