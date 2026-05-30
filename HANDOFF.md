# 🔁 HANDOFF — Protocolo de relevo (equipo de 3)

> Rellenar **siempre** al agotar el límite diario de subidas, antes de soltar el
> teclado. Pegar el bloque nuevo ARRIBA del todo.

## Estado actual (2026-05-30) — relevo 2
- 🧠 **Métrica entendida:** leaderboard = `rmse_macro` = media del RMSE **ABSOLUTO**
  de los 6 índices. **Index_A + Index_D = ~85% del score.** (Antes se ordenaba por
  relativo → se subió `naive_flat`, casi el peor modelo. Corregido.) Ver RESULTS.md.
- ✅ **1ª subida (naive_flat): RMSE plataforma = 74.197.** Local era 32.330 (MACRO),
  o sea la validación es **conservadora → NO hay overfitting**, el backtest es fiable.
- 🆕 **Modelo nuevo `drift_full`** (drift log de historia completa): MACRO local
  **27.648 (−14,5% vs naive)**, gana en A 17/24 orígenes y en D 16/24.
- 📦 **Envío listo PENDIENTE de subir:** `submissions/submission_20260530_drift_full.xlsx`
  (252 filas, 0 nulos, todos >0). Esperado en plataforma ≈ **63–66k**.
- 🔧 Arreglado: la columna `Date` del envío se preserva como TEXTO `YYYY-MM-DD`
  (sin hora), igual que el template. También: template ahora dentro del repo
  (`template/`), backtest fiel a 252 días naturales, validación ordena por MACRO.
- ▶️ **Siguiente:** subir `drift_full`. Luego Fase 2 = exógenas (GBM de retornos) para
  A y D. Reproducir: `python -m src.run_baselines` y `python -m src.predict`.

---

## Plantilla de relevo (copiar y rellenar)

```
## HANDOFF — AAAA-MM-DD HH:MM — de @___ → @___
Intentos: usados hoy __/__  | resetea a las __:__
Mejor envío del día: archivo ____  | RMSE plataforma ____

Mejor modelo AHORA:
  - rama/config: ____
  - RMSE LOCAL (walk-forward 252d, rel. medio): ____
  - RMSE plataforma: ____
  - ¿gap local vs plataforma?  ☐ pequeño (confío en local) ☐ grande (OVERFITTING → parar y revisar validación)

Probado hoy:
  | modelo | RMSE local | RMSE plat | ¿subido? | notas |
  | ------ | ---------- | --------- | -------- | ----- |
  |        |            |           |          |       |

Qué FALLÓ / callejones sin salida:
  - ...

Siguientes pasos (en orden):
  1. ...
  2. ...

Para reproducir el mejor:
  python -m src.run_baselines           # tabla RMSE local
  python -m src.predict --model ____    # genera el envío
Bloqueos abiertos:
  - ...
```
<!-- pega el siguiente handoff encima de esta línea -->
