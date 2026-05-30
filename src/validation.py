"""Validacion FIEL al test real — replica la estructura exacta del envio.

El test abarca 252 dias NATURALES (2028-12-13 -> 2029-08-21). Por eso validamos
sobre ventanas de 252 dias naturales hacia atras: por cada origen T0 entrenamos
con todo hasta T0 y predecimos los dias habiles que caen en (T0, T0+252d] de una
sola vez (forecast directo, no recursivo). Asi el backtest mide LO MISMO que se
puntua, evitando el sesgo de validar sobre 252 dias habiles (~1 ano, mas drift).

METRICA = la del leaderboard: RMSE ABSOLUTO medio entre los 6 indices (MACRO).
Ojo: NO usamos RMSE relativo para rankear. El leaderboard es absoluto y lo dominan
Index_A e Index_D (~85%); ordenar por relativo llevo a subir el peor modelo (naive).

Regla anti-overfitting: un modelo solo es 'mejor' si baja el MACRO absoluto SIN
empeorar A/D ni disparar la dispersion entre origenes (consistencia).
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


def _origins(last_date: pd.Timestamp, horizon_days: int, n: int) -> list[pd.Timestamp]:
    """Fechas de corte T0, separadas `horizon_days` dias naturales hacia atras."""
    return [last_date - pd.Timedelta(days=horizon_days * (i + 1)) for i in range(n)]


def walk_forward_series(series: pd.Series, fn, horizon_days: int,
                        origins: list[pd.Timestamp]) -> list[float]:
    """RMSEs de un modelo sobre una serie, ventana de `horizon_days` dias naturales."""
    out = []
    for t0 in origins:
        hist = series.loc[:t0]
        fut = series.loc[t0 + pd.Timedelta(days=1): t0 + pd.Timedelta(days=horizon_days)]
        if len(hist) < 504 or len(fut) < 50:  # historia y ventana minimas
            continue
        out.append(rmse(fut.to_numpy(), fn(hist, fut.index)))
    return out


def evaluate_all(horizon_days: int = C.BACKTEST_HORIZON_DAYS,
                 n_origins: int = C.BACKTEST_N_ORIGINS) -> pd.DataFrame:
    """Tabla: modelo x indice (RMSE absoluto medio) + MACRO + MACRO de recientes.

    MACRO        = media del RMSE absoluto de los 6 indices en TODOS los origenes.
    MACRO_recent = lo mismo en los 4 origenes mas recientes (regimen mas parecido
                   al test 2028-2029); util para detectar modelos que solo ganan en
                   el pasado lejano.
    """
    idx = D.load_indices()
    origins = _origins(idx.index.max(), horizon_days, n_origins)
    n_recent = 4
    rows = []
    for name, fn in M.REGISTRY.items():
        mean_all, mean_rec, std_all = {}, {}, {}
        for t in C.TARGETS:
            rmses = walk_forward_series(idx[t].dropna(), fn, horizon_days, origins)
            mean_all[t] = np.mean(rmses) if rmses else np.nan
            mean_rec[t] = np.mean(rmses[:n_recent]) if rmses else np.nan
            std_all[t] = np.std(rmses) if rmses else np.nan
        row = {"modelo": name}
        row.update({t: mean_all[t] for t in C.TARGETS})
        row["A+D"] = mean_all["Index_A"] + mean_all["Index_D"]
        row["MACRO"] = float(np.mean(list(mean_all.values())))
        row["MACRO_recent"] = float(np.mean(list(mean_rec.values())))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("MACRO").reset_index(drop=True)
