"""Constantes del reto. Unico sitio donde tocar fechas, rutas y mapeo de columnas.

Verificado contra el bundle real (inspeccion 2026-05-30):
- 6 series objetivo: Index_A..Index_F (niveles tipo indice, base 1000 en origen).
- Train (indices):  1985-06-24 -> 2028-12-12  (~11.956 filas, dias habiles aprox).
- Ventana a predecir: 2028-12-13 -> 2029-08-21  (252 filas, contiguo al train).
- Exogenas (macro/network/news) DISPONIBLES tambien en la ventana de test.
- Plantilla de envio: hoja 'submission', columnas Date + pred_index_a..pred_index_f.
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data_raw"
SUBMISSIONS = ROOT / "submissions"
EXPERIMENTS = ROOT / "experiments"

# Ficheros
F_TRAIN_INDICES = DATA_RAW / "train_indices.csv"
F_TRAIN_MACRO = DATA_RAW / "train_macro_factors.csv"
F_TRAIN_NET = DATA_RAW / "train_network_metrics.csv"
F_TRAIN_NEWS = DATA_RAW / "train_news.csv"
F_TEST_MACRO = DATA_RAW / "test_macro_factors.csv"
F_TEST_NET = DATA_RAW / "test_network_metrics.csv"
F_TEST_NEWS = DATA_RAW / "test_news.csv"

# Plantilla de envio (.xlsx, hoja 'submission')
TEMPLATE_XLSX = Path(
    r"C:\Users\alons\Downloads"
    r"\hackaton_miax_prediccion_prod_rmse_20260530_v1_submission_template.xlsx"
)
TEMPLATE_SHEET = "submission"

DATE_COL = "Date"
TARGETS = ["Index_A", "Index_B", "Index_C", "Index_D", "Index_E", "Index_F"]
# Index_A -> pred_index_a
SUB_COL = {t: f"pred_index_{t.split('_')[1].lower()}" for t in TARGETS}

# Horizonte
HORIZON_START = "2028-12-13"
HORIZON_END = "2029-08-21"

# Backtest: el bloque de validacion = mismo tamano que el test real (~252 dias)
BACKTEST_HORIZON = 252
BACKTEST_N_ORIGINS = 5
