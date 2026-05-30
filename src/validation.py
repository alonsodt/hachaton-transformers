"""Validacion walk-forward de origen movil — replica el test real (bloque de 252d).

Por cada indice y cada origen: entrenar hasta T, predecir el bloque de H fechas
SIGUIENTES de una vez (forecast directo, no a 1 dia), y medir RMSE contra el real.
Reportamos por indice y el promedio entre indices (que es lo que aproxima el
leaderboard si la metrica promedia columnas).

Regla anti-overfitting: un modelo solo es 'mejor' si baja el RMSE medio SIN
disparar la desviacion entre origenes (consistencia).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import models as M


def rmse(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def walk_forward_series(series: pd.Series, fn, horizon: int, n_origins: int) -> list[float]:
    """RMSEs de un modelo sobre una serie, en ventanas no solapadas hacia atras."""
    n = len(series)
    out = []
    for i in range(n_origins):
        val_end = n - i * horizon
        val_start = val_end - horizon
        if val_start < max(horizon, 252):  # historia minima para entrenar
            break
        hist = series.iloc[:val_start]
        actual = series.iloc[val_start:val_end]
        pred = fn(hist, actual.index)
        out.append(rmse(actual.to_numpy(), pred))
    return out


def evaluate_all(horizon: int = C.BACKTEST_HORIZON,
                 n_origins: int = C.BACKTEST_N_ORIGINS) -> pd.DataFrame:
    """Tabla: modelo x indice (RMSE medio) + columna promedio y std media."""
    idx = D.load_indices()
    rows = []
    for name, fn in M.REGISTRY.items():
        per_index_mean = {}
        per_index_std = {}
        for t in C.TARGETS:
            rmses = walk_forward_series(idx[t].dropna(), fn, horizon, n_origins)
            per_index_mean[t] = np.mean(rmses) if rmses else np.nan
            per_index_std[t] = np.std(rmses) if rmses else np.nan
        row = {"modelo": name}
        row.update({t: per_index_mean[t] for t in C.TARGETS})
        row["RMSE_medio"] = np.mean(list(per_index_mean.values()))
        # RMSE relativo medio (normalizado por nivel) -> comparacion justa entre indices
        rel = [per_index_mean[t] / idx[t].iloc[-1] for t in C.TARGETS]
        row["RMSE_rel_medio"] = float(np.mean(rel))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("RMSE_rel_medio").reset_index(drop=True)
    return df
