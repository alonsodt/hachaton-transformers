"""Genera el envio: reentrena con TODO el historico y rellena la plantilla .xlsx.

    python -m src.predict --model naive_flat
    python -m src.predict --model damped_252_098

Rellena la hoja 'submission' (Date, pred_index_a..f) preservando sus filas
exactas, mapeando por fecha. Escribe en submissions/.
"""
from __future__ import annotations

import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import models as M


def build_submission(model: str) -> pd.DataFrame:
    idx = D.load_indices()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tpl = pd.read_excel(C.TEMPLATE_XLSX, sheet_name=C.TEMPLATE_SHEET)
    tpl[C.DATE_COL] = pd.to_datetime(tpl[C.DATE_COL])
    future_dates = pd.DatetimeIndex(tpl[C.DATE_COL])

    fn = M.REGISTRY[model]
    for t in C.TARGETS:
        series = idx[t].dropna()
        preds = fn(series, future_dates)
        tpl[C.SUB_COL[t]] = np.asarray(preds, dtype=float)

    return tpl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="naive_flat", choices=list(M.REGISTRY))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sub = build_submission(args.model)

    # sanity checks antes de escribir
    assert len(sub) == 252, f"esperaba 252 filas, hay {len(sub)}"
    for t in C.TARGETS:
        col = C.SUB_COL[t]
        assert sub[col].notna().all(), f"NaN en {col}"
        assert (sub[col] > 0).all(), f"valores no positivos en {col}"

    C.SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    out = args.out or (C.SUBMISSIONS / f"submission_{stamp}_{args.model}.xlsx")
    sub.to_excel(out, sheet_name=C.TEMPLATE_SHEET, index=False)
    print(f"[predict] modelo={args.model} | filas={len(sub)}")
    print(f"[predict] envio escrito en: {out}")
    print(sub.head(3).to_string(index=False))
    print("...")
    print(sub.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
