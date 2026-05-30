"""Corre el walk-forward de todos los baselines y guarda la tabla de RMSE local.

    python -m src.run_baselines

La tabla se ordena por RMSE relativo medio (normalizado por el nivel de cada
indice) porque los 6 indices estan a escalas muy distintas (10^4 a 10^6) y el
RMSE absoluto lo dominarian Index_A/D. Tambien mostramos el RMSE absoluto medio.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from . import config as C
from .validation import evaluate_all


def main() -> None:
    table = evaluate_all()
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

    show = table[["modelo", "RMSE_medio", "RMSE_rel_medio"]].copy()
    show["RMSE_rel_medio"] = (show["RMSE_rel_medio"] * 100).round(3).astype(str) + "%"
    print("\n=== Ranking baselines (walk-forward 252d) ===")
    print(show.to_string(index=False))

    best = table.iloc[0]["modelo"]
    print(f"\nMEJOR baseline (RMSE relativo): {best}")

    C.EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = C.EXPERIMENTS / f"baselines_{stamp}.csv"
    table.to_csv(out, index=False)
    print(f"Tabla completa por indice guardada en: {out}")


if __name__ == "__main__":
    main()
