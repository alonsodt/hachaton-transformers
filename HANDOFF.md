# 🔁 HANDOFF — Protocolo de relevo (equipo de 3)

> Rellenar **siempre** al agotar el límite diario de subidas, antes de soltar el
> teclado. Pegar el bloque nuevo ARRIBA del todo.

## Estado actual (2026-05-30)
- ✅ Pipeline montado y verificado. Primera base generada.
- 📦 Envío base: `submissions/submission_20260530_naive_flat.xlsx` (252 filas, 0 nulos).
- 📈 RMSE local de la base: **≈ 8.6 %** relativo (walk-forward 252d). **PENDIENTE de subir.**
- ▶️ Siguiente: subir la base para anclar el leaderboard y luego atacar Fase 2 (ver RESULTS.md).

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
