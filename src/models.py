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


def drift_full(hist, future_dates) -> np.ndarray:
    """Drift log-lineal con la HISTORIA COMPLETA (43 anos).

    GANADOR del backtest fiel (252 dias naturales, 24 origenes): MACRO 27.6k vs
    32.3k de naive_flat (-14,5%), y -12% en los origenes recientes. Bate a naive en
    A en 17/24 origenes y en D en 16/24.

    POR QUE FUNCIONA donde drift_252 fallaba: la pendiente con ventana corta es
    ruidosa y se dispara (extrapola un tramo caliente a 252d). Con TODA la historia
    la pendiente es la tasa estructural (~+12,8%/ano en A y D) -> estable y, por
    construccion, el estimador de minimo RMSE para una serie con drift positivo
    (E[P_t+h] = P_t * exp(mu*h) > ultimo valor). naive_flat ignora ese drift y
    queda sesgado a la baja justo en A y D, que pesan el 85% del leaderboard.
    """
    return drift_log(hist, future_dates, window=len(hist))


def _log_slope_recent_days(hist: pd.Series, days: int) -> float:
    """OLS de log(precio) ~ dias sobre la ventana de los ultimos `days` NATURALES.

    A diferencia de `_log_slope_per_day` (ventana en nº de puntos), aqui la ventana
    es por CALENDARIO real, asi el 'reciente' es comparable entre origenes del
    backtest (cada origen tiene distinta densidad de puntos). Mide la tasa del
    regimen reciente, que en A y D (~50-60%/ano) va MUY por encima de la
    estructural de 43 anos (~17%/ano).
    """
    cutoff = hist.index[-1] - pd.Timedelta(days=days)
    h = hist.loc[cutoff:]
    if len(h) < 30:  # ventana demasiado corta -> cae a la pendiente plena
        return _log_slope_per_day(hist, len(hist))[1]
    d = np.array([(t - h.index[-1]).days for t in h.index], dtype=float)
    y = np.log(h.to_numpy())
    dm = d - d.mean()
    return float((dm * (y - y.mean())).sum() / (dm**2).sum())


def drift_blend(hist, future_dates, recent_days: int = 5 * 365,
                w_full: float = 0.5) -> np.ndarray:
    """Drift log-lineal con pendiente = mezcla de la ESTRUCTURAL (43 anos) y la del
    REGIMEN RECIENTE (ventana `recent_days` naturales).

    POR QUE. `drift_full` usa solo la tasa de 43 anos (17%/ano en A y D) y se queda
    corto si el momento reciente (50-60%/ano) continua; un drift de ventana corta
    pura se dispara y explota el RMSE (ya visto con drift_log_252). La mezcla busca
    el punto medio: capturar parte de la aceleracion reciente sin extrapolar un
    tramo caliente entero. Ancla en el ultimo precio observado (no en el ajuste).
    """
    _, slope_full = _log_slope_per_day(hist, len(hist))
    slope_recent = _log_slope_recent_days(hist, recent_days)
    slope = w_full * slope_full + (1.0 - w_full) * slope_recent
    last_log = float(np.log(hist.iloc[-1]))
    return np.exp(last_log + slope * _days_ahead(hist, future_dates))


def blend_full(hist, future_dates, w=0.7) -> np.ndarray:
    """Cobertura: media geometrica de drift_full (w) y flat (1-w).

    Algo peor que drift_full en el backtest pero mas robusto si 2029 corrige (un
    crash penaliza a quien extrapola la subida). w=0.7 conserva la mayor parte de
    la mejora cediendo un poco a cambio de menor cola de riesgo.
    """
    d = np.log(drift_full(hist, future_dates))
    f = np.log(naive_flat(hist, future_dates))
    return np.exp(w * d + (1 - w) * f)


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


# Registro a evaluar en el walk-forward.
# Orden = de mejor a peor segun el backtest fiel por RMSE ABSOLUTO (la metrica real).
REGISTRY = {
    "drift_full": drift_full,                                    # <- MEJOR (envio actual)
    # --- drift calibrado: estructural x regimen reciente (Fase 2, foco A y D) ---
    "drift_blend_5y_0.7": lambda h, f: drift_blend(h, f, recent_days=5*365, w_full=0.7),
    "drift_blend_5y_0.5": lambda h, f: drift_blend(h, f, recent_days=5*365, w_full=0.5),
    "drift_blend_3y_0.7": lambda h, f: drift_blend(h, f, recent_days=3*365, w_full=0.7),
    "drift_blend_3y_0.5": lambda h, f: drift_blend(h, f, recent_days=3*365, w_full=0.5),
    "drift_recent_5y": lambda h, f: drift_blend(h, f, recent_days=5*365, w_full=0.0),
    "drift_recent_3y": lambda h, f: drift_blend(h, f, recent_days=3*365, w_full=0.0),
    "blend_full_0.7": lambda h, f: blend_full(h, f, w=0.7),      # cobertura anti-crash
    "blend_full_0.5": lambda h, f: blend_full(h, f, w=0.5),
    "naive_flat": naive_flat,                                    # suelo de referencia (1a subida)
    "drift_log_756": lambda h, f: drift_log(h, f, window=756),
    "damped_252_098": lambda h, f: damped_drift_log(h, f, window=252, phi=0.98),
    "ewma_30": ewma_flat,
    "drift_log_252": lambda h, f: drift_log(h, f, window=252),   # peor: ventana corta se dispara
}
