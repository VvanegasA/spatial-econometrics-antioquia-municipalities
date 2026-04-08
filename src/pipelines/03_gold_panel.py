# Gold Layer: Final Panel for Spatial Econometrics
# Antioquia municipalities, 2015-2024
# Run from repo root: python src/pipelines/03_gold_panel.py
#
# Variables currently included:
#   - va_total, ln_va_total     (DANE — municipal value added)
#   - homicidios, tasa_homicidios  (INMLCF / crime records)
#   - cobertura_secundaria      (SIMAT / TerriData — human capital proxy)
#
# Upcoming — uncomment the merge block when the silver is ready:
#   - idf                       (DNP — Fiscal Performance Index, 2000-2023)
#   - desplazados               (UARIV — forced displacement)
#   - mortalidad_infantil       (DANE — Vital Statistics)

import os
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parents[2])
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=f"logs/gold_panel_{datetime.today().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

SILVER_PATH = os.path.join("data", "silver")
GOLD_PATH   = os.path.join("data", "gold")
os.makedirs(GOLD_PATH, exist_ok=True)

# -------------------------------------------------------------------------
# Step 1 — Load all Silver datasets
# -------------------------------------------------------------------------

dane = pd.read_parquet(os.path.join(SILVER_PATH, "dane_silver.parquet"))
seg  = pd.read_parquet(os.path.join(SILVER_PATH, "seguridad_silver.parquet"))
sim  = pd.read_parquet(os.path.join(SILVER_PATH, "simat_silver.parquet"))
pobl = pd.read_parquet(os.path.join(SILVER_PATH, "poblacion_silver.parquet"))

print(f"[INFO] Silver DANE       : {dane.shape}")
print(f"[INFO] Silver Security   : {seg.shape}")
print(f"[INFO] Silver SIMAT      : {sim.shape}")
print(f"[INFO] Silver Population : {pobl.shape}")

# -------------------------------------------------------------------------
# Step 2 — DANE is the anchor; all others join onto it
# -------------------------------------------------------------------------

panel = dane[["cod_mpio", "municipio", "subregion", "year",
              "va_total", "ln_va_total"]].copy()

# -------------------------------------------------------------------------
# Step 3 — Merge population and compute per-capita metrics
# va_total is in billions of COP; multiply by 1,000 to get millions per person.
# -------------------------------------------------------------------------

panel = panel.merge(pobl[["cod_mpio", "year", "poblacion"]],
                    on=["cod_mpio", "year"], how="left")

panel["va_per_capita"]    = (panel["va_total"] * 1_000) / panel["poblacion"]
panel["ln_va_per_capita"] = np.log(panel["va_per_capita"].clip(lower=0.001))
panel["ln_poblacion"]     = np.log(panel["poblacion"].clip(lower=1))

print(f"\n[INFO] After population merge: {panel.shape}")

# -------------------------------------------------------------------------
# Step 3b — Merge homicide counts
# Missing municipality-years get 0 homicides (no record = no event).
# -------------------------------------------------------------------------

panel = panel.merge(seg[["cod_mpio", "year", "homicidios"]],
                    on=["cod_mpio", "year"], how="left")
panel["homicidios"] = panel["homicidios"].fillna(0).astype(int)
print(f"[INFO] After security merge: {panel.shape}")

# Normalize by population so large and small municipalities are comparable.
panel["tasa_homicidios"] = np.where(
    panel["poblacion"] > 0,
    panel["homicidios"] / panel["poblacion"] * 100_000,
    np.nan
)

# -------------------------------------------------------------------------
# Step 4 — Merge secondary coverage
# SIMAT data runs through 2023; 2024 will be NaN (expected statistical lag).
# -------------------------------------------------------------------------

panel = panel.merge(sim[["cod_mpio", "year", "cobertura_secundaria"]],
                    on=["cod_mpio", "year"], how="left")
n_nulos_cob = panel["cobertura_secundaria"].isna().sum()
print(f"[INFO] After SIMAT merge: {panel.shape}")
print(f"  Null coverage (expected ~125 for 2024): {n_nulos_cob}")

# -------------------------------------------------------------------------
# Step 5 — Placeholder: upcoming data sources
# Uncomment and add the corresponding silver script when data is available.
# -------------------------------------------------------------------------

# --- DNP: Fiscal Performance Index (IDF) ---
# dnp = pd.read_parquet(os.path.join(SILVER_PATH, "dnp_silver.parquet"))
# panel = panel.merge(dnp[["cod_mpio", "year", "idf"]], on=["cod_mpio", "year"], how="left")

# --- UARIV: Forced displacement ---
# uariv = pd.read_parquet(os.path.join(SILVER_PATH, "uariv_silver.parquet"))
# panel = panel.merge(uariv[["cod_mpio", "year", "desplazados"]], on=["cod_mpio", "year"], how="left")

# --- DANE Vital Statistics: Infant mortality ---
# vitales = pd.read_parquet(os.path.join(SILVER_PATH, "vitales_silver.parquet"))
# panel = panel.merge(vitales[["cod_mpio", "year", "mortalidad_infantil"]], on=["cod_mpio", "year"], how="left")

# -------------------------------------------------------------------------
# Step 6 — Sort and verify panel balance
# -------------------------------------------------------------------------

panel = panel.sort_values(["cod_mpio", "year"]).reset_index(drop=True)

n_mpios    = panel["cod_mpio"].nunique()
n_years    = panel["year"].nunique()
n_obs      = len(panel)
n_expected = n_mpios * n_years
balanced   = n_obs == n_expected

print(f"\n[INFO] Panel balance by year:")
print(panel.groupby("year")["cod_mpio"].count().to_string())
print(f"\n  Municipalities  : {n_mpios:,}")
print(f"  Years           : {n_years}")
print(f"  Observations    : {n_obs:,}")
print(f"  Expected (N*T)  : {n_expected:,}")
print(f"  Status          : {'BALANCED' if balanced else 'UNBALANCED'}")

# -------------------------------------------------------------------------
# Step 7 — Descriptive statistics
# -------------------------------------------------------------------------

cols_desc = ["va_total", "ln_va_total", "poblacion", "ln_poblacion", "va_per_capita",
             "ln_va_per_capita", "tasa_homicidios", "cobertura_secundaria"]
print("\n" + "=" * 55)
print("  DESCRIPTIVE STATISTICS — GOLD PANEL")
print("=" * 55)
print(panel[cols_desc].describe().round(2).to_string())
print("=" * 55)

# -------------------------------------------------------------------------
# Step 8 — Save Gold (three formats for flexibility)
# -------------------------------------------------------------------------

panel.to_parquet(os.path.join(GOLD_PATH, "panel_gold.parquet"), index=False)
panel.to_csv(os.path.join(GOLD_PATH, "panel_gold.csv"), index=False, encoding="utf-8-sig")
panel.to_excel(os.path.join(GOLD_PATH, "panel_gold.xlsx"), index=False)

print(f"\n[SUCCESS] Gold parquet : data/gold/panel_gold.parquet")
print(f"[SUCCESS] Gold CSV     : data/gold/panel_gold.csv")
print(f"[SUCCESS] Gold Excel   : data/gold/panel_gold.xlsx")

logging.info(f"Gold panel saved: {panel.shape} — cols: {panel.columns.tolist()}")
print("\n[SUCCESS] Gold panel completed — ready for spatial econometrics.")