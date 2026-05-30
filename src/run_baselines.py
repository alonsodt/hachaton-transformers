"""Corre el walk-forward de todos los baselines y guarda la tabla de RMSE local.

    python -m src.run_baselines

Ordena por MACRO = RMSE ABSOLUTO medio entre los 6 indices, que es EXACTAMENTE lo
que puntua el leaderboard (verificado contra la 1a subida: rmse_macro = media de
los 6 RMSE absolutos). Index_A e Index_D pesan ~85%, por eso mostramos tambien la
columna A+D y el MACRO de los origenes recientes (regimen mas parecido al test).
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from . import config as C
from .validation import evaluate_all


def main() -> None:
    table = evaluate_all()
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:,.0f}")

    show = table[["modelo", "Index_A", "Index_D", "A+D", "MACRO", "MACRO_recent"]]
    print("\n=== Ranking baselines (backtest fiel 252 dias naturales, RMSE ABSOLUTO) ===")
    print(show.to_string(index=False))

    best = table.iloc[0]["modelo"]
    print(f"\nMEJOR modelo (MACRO absoluto = metrica del leaderboard): {best}")

    C.EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = C.EXPERIMENTS / f"baselines_{stamp}.csv"
    table.to_csv(out, index=False)
    print(f"Tabla completa por indice guardada en: {out}")


if __name__ == "__main__":
    main()
