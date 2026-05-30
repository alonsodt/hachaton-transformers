"""Fase 3 — Redes neuronales y Transformer para forecast directo a largo horizonte.

    python -m src.deep_models            # walk-forward: compara DL vs drift_full/naive
    python -m src.deep_models --quick    # menos epocas/origenes (prueba rapida)

POR QUE ESTE PLANTEAMIENTO (y no un seq2seq de 252 pasos)
---------------------------------------------------------
El test es un forecast DIRECTO a 252 dias naturales de 6 indices cuasi-paseo-
aleatorio con drift. Predecir autoregresivamente acumularia error (bola de nieve);
y un seq2seq que escupe 252 niveles sobre-ajusta (pocos origenes efectivos). En su
lugar cada red predice UN escalar por (indice, origen): el **drift diario log medio**
del horizonte, y reconstruimos el camino P_t0 * exp(slope * dias). Esto:
  - es DIRECTO (no recursivo) -> no acumula error,
  - es exactamente comparable a drift_full (que usa el slope historico fijo): aqui
    el slope lo CONDICIONA la red a la secuencia reciente (momentum/vol),
  - regulariza el horizonte largo (1 grado de libertad, no 252).
La pregunta que responde el backtest: ¿una red estima mejor el drift futuro que la
media de 43 anos? (El paper DLinear avisa: a menudo NO. Por eso incluimos DLinear y
lo validamos con honestidad antes de gastar una subida.)

ANTI-LEAKAGE
------------
En cada origen T0 reentrenamos SOLO con muestras cuyo futuro termina <= T0. La
entrada se estandariza por ventana (sin estadisticos globales). P_t0 es el ultimo
nivel conocido. Nada del futuro entra en el entrenamiento ni en el escalado.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import models as M
from .validation import rmse, _origins

# torch se importa aqui (perezoso) para no obligar a tenerlo en el resto del pipeline
import torch
import torch.nn as nn

torch.set_num_threads(6)   # hilos fijos: evita oversubscription/throttling en CPU

SEED = 0
L = 32           # ventana de entrada (dias habiles de retornos log); corta = mas rapido
MAX_SAMPLES = 2000   # submuestreo del set de entrenamiento (CPU); suficiente p/ converger
HORIZON_DAYS = C.BACKTEST_HORIZON_DAYS  # 252 dias naturales
MIN_FUT = 50     # dias habiles minimos en la ventana futura


# --------------------------------------------------------------------------- #
#  Arquitecturas (todas: entrada (B, L) de retornos log estandarizados -> escalar)
# --------------------------------------------------------------------------- #
class DLinear(nn.Module):
    """Modelo lineal sobre la ventana (recomendacion del paper DLinear)."""
    def __init__(self, seq_len=L):
        super().__init__()
        self.fc = nn.Linear(seq_len, 1)

    def forward(self, x):  # x: (B, L)
        return self.fc(x).squeeze(-1)


class LSTMReg(nn.Module):
    """Red neuronal profunda recurrente (requisito Fase 3)."""
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):  # x: (B, L)
        out, _ = self.lstm(x.unsqueeze(-1))   # (B, L, H)
        return self.head(out[:, -1, :]).squeeze(-1)


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=L):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, d)

    def forward(self, x):  # (B, L, d)
        return x + self.pe[:, : x.size(1)]


class TimeSeriesTransformer(nn.Module):
    """Encoder Transformer con positional encoding y atencion sobre el contexto
    temporal; pooling medio -> cabeza lineal al drift escalar (Fase 3, requisito)."""
    def __init__(self, d_model=24, nhead=4, layers=1, ff=48):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.pos = _PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):  # x: (B, L)
        h = self.proj(x.unsqueeze(-1))      # (B, L, d)
        h = self.pos(h)
        h = self.encoder(h)                 # (B, L, d)
        return self.head(h.mean(dim=1)).squeeze(-1)


ARCHITECTURES = {"dlinear": DLinear, "lstm": LSTMReg, "transformer": TimeSeriesTransformer}


# --------------------------------------------------------------------------- #
#  Construccion de muestras (X = ventana de retornos log; y = drift diario realizado)
# --------------------------------------------------------------------------- #
def _build_samples(idx: pd.DataFrame, cutoff: pd.Timestamp):
    """Muestras (X, y) de TODOS los indices cuyo futuro termina <= cutoff (anti-leakage).

    Vectorizado con searchsorted sobre dias-ordinales (evita .loc por fila).
    X = ultimos L retornos log hasta T0;  y = drift diario log realizado hacia el
    ultimo dia habil dentro de (T0, T0+252d].
    """
    cutoff_ord = int(np.datetime64(cutoff, "D").astype("int64"))
    X, y = [], []
    for t in C.TARGETS:
        s = idx[t].dropna()
        logp = np.log(s.to_numpy())
        rets = np.diff(logp)                                  # retornos log (len n-1)
        dord = s.index.to_numpy().astype("datetime64[D]").astype("int64")
        n = len(s)
        # primer indice futuro = i+1 ; ultimo = primer date > T0+252d, menos 1
        hi = np.searchsorted(dord, dord + HORIZON_DAYS, side="right") - 1  # end index per i
        for i in range(L, n - MIN_FUT):
            end = hi[i]
            if end - (i + 1) + 1 < MIN_FUT:                   # nº de dias habiles futuros
                continue
            if dord[end] > cutoff_ord:                        # el futuro no pasa del corte
                continue
            ndays = dord[end] - dord[i]
            if ndays <= 0:
                continue
            X.append(rets[i - L: i])
            y.append((logp[end] - logp[i]) / ndays)           # drift diario realizado
    return np.asarray(X, np.float32), np.asarray(y, np.float32)


def _standardize_rows(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / sd


def _train_one(arch_name, X, y, epochs, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if len(X) > MAX_SAMPLES:                      # submuestreo reproducible (coste CPU)
        sel = np.random.RandomState(SEED).choice(len(X), MAX_SAMPLES, replace=False)
        X, y = X[sel], y[sel]
    Xs = _standardize_rows(X)
    y_mu, y_sd = float(y.mean()), float(y.std() + 1e-12)
    ys = (y - y_mu) / y_sd
    Xt = torch.tensor(Xs, device=device)
    yt = torch.tensor(ys, device=device)

    model = ARCHITECTURES[arch_name]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    n = len(Xt)
    bs = min(256, n)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for j in range(0, n, bs):
            b = perm[j: j + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
    model.eval()
    return model, (y_mu, y_sd)


def _predict_slope(model, y_stats, window_rets, device):
    y_mu, y_sd = y_stats
    x = _standardize_rows(window_rets[None, :].astype(np.float32))
    with torch.no_grad():
        out = model(torch.tensor(x, device=device)).cpu().numpy()[0]
    return out * y_sd + y_mu


# --------------------------------------------------------------------------- #
#  Walk-forward: reentrena por origen y compara DL vs drift_full / naive
# --------------------------------------------------------------------------- #
def evaluate(n_origins=6, epochs=60, archs=("dlinear", "lstm", "transformer")):
    device = "cpu"
    idx = D.load_indices()
    origins = _origins(idx.index.max(), HORIZON_DAYS, n_origins)

    # baselines de referencia sobre LOS MISMOS origenes
    base_fns = {"naive_flat": M.naive_flat, "drift_full": M.drift_full}
    names = list(base_fns) + list(archs) + ["ens_drift+tr"]
    sq_err = {k: [] for k in names}
    # MACRO por origen (media de RMSE de los 6 indices en cada origen) -> consistencia
    per_origin = {k: [] for k in names}

    for oi, t0 in enumerate(origins):
        # entrenar cada red con datos cuyo futuro termina <= t0
        import time
        X, y = _build_samples(idx, cutoff=t0)
        print(f"  origen {oi+1}/{len(origins)} ({t0.date()}): {len(X)} muestras", flush=True)
        trained = {}
        for a in archs:
            t_ini = time.time()
            trained[a] = _train_one(a, X, y, epochs, device)
            print(f"      {a:12s} entrenado en {time.time()-t_ini:5.1f}s", flush=True)

        tmp = {k: [] for k in names}   # RMSE por indice en ESTE origen
        for t in C.TARGETS:
            s = idx[t].dropna()
            hist = s.loc[:t0]
            fut = s.loc[t0 + pd.Timedelta(days=1): t0 + pd.Timedelta(days=HORIZON_DAYS)]
            if len(hist) < L + 1 or len(fut) < MIN_FUT:
                continue
            actual = fut.to_numpy()
            cal = np.array([(d - t0).days for d in fut.index], float)
            P0 = float(hist.iloc[-1])
            window = np.diff(np.log(hist.to_numpy()))[-L:]

            def _record(name, pred):
                e = (pred - actual) ** 2
                sq_err[name].append(e)
                tmp[name].append(float(np.sqrt(np.mean(e))))

            for name, fn in base_fns.items():
                _record(name, fn(hist, fut.index))

            slopes = {}
            for a in archs:
                model, ystats = trained[a]
                slopes[a] = _predict_slope(model, ystats, window, device)
                _record(a, P0 * np.exp(slopes[a] * cal))

            # ensemble: media del drift historico y el del transformer
            if "transformer" in archs:
                _, sl_full = M._log_slope_per_day(hist, len(hist))
                _record("ens_drift+tr", P0 * np.exp((0.5 * sl_full + 0.5 * slopes["transformer"]) * cal))

        for name in names:
            if tmp[name]:
                per_origin[name].append(float(np.mean(tmp[name])))  # MACRO de este origen

    # resumen global + tabla MACRO por origen (consistencia)
    rows = []
    for name, errs in sq_err.items():
        if not errs:
            continue
        rmses = [float(np.sqrt(np.mean(e))) for e in errs]
        rows.append({"modelo": name, "MACRO": float(np.mean(rmses))})
    out = pd.DataFrame(rows).sort_values("MACRO").reset_index(drop=True)
    po = pd.DataFrame(per_origin, index=[f"orig{ i+1}" for i in range(len(origins))]).T
    return out, po


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--archs", default=None, help="csv: dlinear,lstm,transformer")
    args = ap.parse_args()
    n_origins, epochs = (3, 25) if args.quick else (6, 30)
    # LSTM excluido del run por defecto: es ~10x mas lento en CPU y el PEOR en el
    # quick. Para incluirlo: --archs dlinear,lstm,transformer
    archs = tuple(args.archs.split(",")) if args.archs else ("dlinear", "transformer")
    print(f"[deep] walk-forward: {n_origins} origenes, {epochs} epocas, archs={archs}, device=cpu")
    table, po = evaluate(n_origins=n_origins, epochs=epochs, archs=archs)
    pd.set_option("display.float_format", lambda x: f"{x:,.0f}")
    pd.set_option("display.width", 200)
    print("\n=== DL vs baselines (RMSE absoluto medio, mismos origenes) ===")
    print(table.to_string(index=False))
    print("\n=== MACRO por origen (consistencia: orig1 = mas reciente) ===")
    print(po.to_string())

    best = table.iloc[0]["modelo"]
    drift_row = po.loc["drift_full"] if "drift_full" in po.index else None
    print(f"\nMEJOR (global): {best}")
    # REGLA ROBUSTA: solo merece subir un DL si bate a drift_full en >=2/3 de los
    # origenes Y mejora el MACRO global > 1%. Si no, drift_full sigue siendo la apuesta.
    if drift_row is not None:
        for cand in [m for m in table["modelo"] if m not in ("naive_flat", "drift_full")]:
            wins = int((po.loc[cand] < drift_row).sum())
            delta = (table.loc[table.modelo == cand, "MACRO"].iloc[0] /
                     table.loc[table.modelo == "drift_full", "MACRO"].iloc[0] - 1) * 100
            verdict = ("SUBIBLE" if wins >= int(np.ceil(len(po.columns) * 2 / 3)) and delta < -1
                       else "ruido / NO bate drift_full de forma fiable")
            print(f"  {cand:14s} gana a drift_full en {wins}/{len(po.columns)} origenes "
                  f"| MACRO {delta:+.1f}% -> {verdict}")


if __name__ == "__main__":
    main()
