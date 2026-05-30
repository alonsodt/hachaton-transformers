"""Ingesta y limpieza.

Para el PRIMER envio solo necesitamos los indices (las 6 series objetivo). Los
loaders de exogenas (macro/network/news) quedan listos para la Fase 2 cuando
metamos modelos con features; se cargan con merge_asof(backward) para no traer
informacion de T+1 (anti-leakage).

Decisiones:
- Las fechas vienen ordenadas; parseamos a datetime y dejamos un indice de fecha.
- Indices: NO reindexamos a calendario natural para los baselines univariantes;
  trabajamos sobre la rejilla nativa de cada serie y proyectamos por dias de
  calendario reales (robusto a huecos de fin de semana / festivos).
- Exogenas: para modelos con features, se reindexan a la rejilla de los indices
  y se hace ffill (ultimo dato conocido; nunca media -> evita inventar e introducir
  leakage del futuro).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def load_indices() -> pd.DataFrame:
    """DataFrame indexado por fecha con columnas Index_A..Index_F."""
    df = pd.read_csv(C.F_TRAIN_INDICES)
    df[C.DATE_COL] = pd.to_datetime(df[C.DATE_COL])
    df = df.sort_values(C.DATE_COL).drop_duplicates(C.DATE_COL).set_index(C.DATE_COL)
    return df[C.TARGETS].astype(float)


def get_series(name: str) -> pd.Series:
    """Serie de un indice concreto, limpia (sin NaN), indexada por fecha."""
    s = load_indices()[name].dropna()
    return s


# ----------------- Exogenas (para Fase 2; no se usan en baselines) -----------------

def load_exog_numeric() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (train_exog, test_exog) numericas: macro + network unidas por fecha."""
    def _read(path):
        d = pd.read_csv(path)
        d[C.DATE_COL] = pd.to_datetime(d[C.DATE_COL])
        return d.sort_values(C.DATE_COL).drop_duplicates(C.DATE_COL).set_index(C.DATE_COL)

    tr = _read(C.F_TRAIN_MACRO).join(_read(C.F_TRAIN_NET), how="outer").sort_index()
    te = _read(C.F_TEST_MACRO).join(_read(C.F_TEST_NET), how="outer").sort_index()
    return tr, te


def load_news_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrega noticias a nivel diario: nº de titulares y nº de fuentes distintas.

    (Placeholder para Fase 2; el sentimiento se anadira con un lexicon/VADER.)
    """
    def _agg(path):
        d = pd.read_csv(path)
        d[C.DATE_COL] = pd.to_datetime(d[C.DATE_COL])
        g = d.groupby(C.DATE_COL).agg(
            news_count=("Headline", "size"),
            n_sources=("Source", "nunique"),
        )
        return g

    return _agg(C.F_TRAIN_NEWS), _agg(C.F_TEST_NEWS)


if __name__ == "__main__":
    idx = load_indices()
    print("indices:", idx.shape, idx.index.min().date(), "->", idx.index.max().date())
    print(idx.tail(3).round(1).to_string())
