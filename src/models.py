"""Modelos de prediccion a horizonte largo (univariantes, por indice).

Interfaz comun:  forecast(hist, future_dates) -> np.ndarray
  hist          : pd.Series indexada por fecha (un indice).
  future_dates  : pd.DatetimeIndex de las fechas a predecir.

POR QUE EN ESPACIO LOG
----------------------
Los indices son multiplicativos (crecen/decrecen en %), con vol diaria ~1-3% y
niveles de 10^4 a 10^6. Modelar log(precio) hace el drift y la tendencia lineales
y evita predicciones negativas. Devolvemos siempre el nivel (exp).

POR QUE OJO CON EL DRIFT
------------------------
Las tendencias son INESTABLES (ej. Index_F: +53% a 1 ano pero -20% a 2). Extrapolar
la pendiente reciente a 252 dias puede dispararse. Por eso incluimos variantes
AMORTIGUADAS (damped) que frenan la tendencia con el horizonte, y dejamos que el
walk-forward elija empiricamente. Para un random-walk puro, el optimo en RMSE es el
ultimo valor (naive_flat): es nuestro suelo de referencia.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _days_ahead(hist: pd.Series, future_dates: pd.DatetimeIndex) -> np.ndarray:
    last = hist.index[-1]
    return np.array([(d - last).days for d in future_dates], dtype=float)


def naive_flat(hist: pd.Series, future_dates: pd.DatetimeIndex) -> np.ndarray:
    return np.full(len(future_dates), float(hist.iloc[-1]))


def _log_slope_per_day(hist: pd.Series, window: int) -> tuple[float, float]:
    """OLS de log(precio) ~ dias, sobre los ultimos `window` puntos.
    Devuelve (intercept_en_last, pendiente_por_dia)."""
    h = hist.iloc[-window:] if len(hist) > window else hist
    days = np.array([(d - h.index[-1]).days for d in h.index], dtype=float)
    y = np.log(h.to_numpy())
    # pendiente por minimos cuadrados
    dm = days - days.mean()
    slope = float((dm * (y - y.mean())).sum() / (dm**2).sum())
    last_log = float(y[-1])
    return last_log, slope


def drift_log(hist, future_dates, window=252) -> np.ndarray:
    last_log, slope = _log_slope_per_day(hist, window)
    return np.exp(last_log + slope * _days_ahead(hist, future_dates))


def damped_drift_log(hist, future_dates, window=252, phi=0.98) -> np.ndarray:
    """Tendencia amortiguada (Gardner): la pendiente se atenua con el horizonte.
    eff(h) = slope * (phi*(1-phi^h)/(1-phi)) sobre el ranking diario h=1..H."""
    last_log, slope = _log_slope_per_day(hist, window)
    h = np.arange(1, len(future_dates) + 1)
    eff = slope * (phi * (1 - phi**h) / (1 - phi))
    return np.exp(last_log + eff)


def ewma_flat(hist, future_dates, span=30) -> np.ndarray:
    val = float(np.exp(np.log(hist).ewm(span=span).mean().iloc[-1]))
    return np.full(len(future_dates), val)


def blend_flat_drift(hist, future_dates, window=252, w=0.5) -> np.ndarray:
    """Media geometrica de flat y drift: tendencia atenuada de forma simple."""
    f = naive_flat(hist, future_dates)
    d = drift_log(hist, future_dates, window)
    return np.exp(w * np.log(d) + (1 - w) * np.log(f))


# Registro a evaluar en el walk-forward
REGISTRY = {
    "naive_flat": naive_flat,
    "drift_log_252": lambda h, f: drift_log(h, f, window=252),
    "drift_log_756": lambda h, f: drift_log(h, f, window=756),
    "damped_252_098": lambda h, f: damped_drift_log(h, f, window=252, phi=0.98),
    "damped_252_095": lambda h, f: damped_drift_log(h, f, window=252, phi=0.95),
    "ewma_30": ewma_flat,
    "blend_flat_drift252": lambda h, f: blend_flat_drift(h, f, window=252, w=0.5),
}
