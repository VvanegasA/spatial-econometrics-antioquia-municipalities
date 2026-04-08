# Silver Layer: Municipal Population — DANE (2015-2024)
# Reshapes wide-format projections to a long panel (one row per municipality-year).
# Run from repo root: python src/pipelines/02_silver_poblacion.py

import os
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parents[2])
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=f"logs/02_silver_poblacion_{datetime.today().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

BRONZE_PATH = os.path.join("data", "bronze", "poblacion_bronze.parquet")
SILVER_PATH = os.path.join("data", "silver")
os.makedirs(SILVER_PATH, exist_ok=True)

if not os.path.exists(BRONZE_PATH):
    raise FileNotFoundError(
        "[ERROR] Run src/pipelines/01_bronze_dane.py first."
    )

df = pd.read_parquet(BRONZE_PATH)
print(f"[INFO] Bronze loaded: {df.shape}")

# -------------------------------------------------------------------------
# Step 1 — Melt from wide (one col per year) to long (one row per mpio-year)
# Only keep the years that overlap with our analysis window.
# -------------------------------------------------------------------------

years = [str(y) for y in range(2015, 2036)]
years_present = [y for y in years if y in df.columns]

cols_to_keep = ["DPMP", "MPIO"] + years_present
df = df[cols_to_keep].copy()

df_long = pd.melt(
    df,
    id_vars=["DPMP", "MPIO"],
    value_vars=years_present,
    var_name="year",
    value_name="poblacion"
)

# -------------------------------------------------------------------------
# Step 2 — Standardize DIVIPOLA codes
# -------------------------------------------------------------------------

df_long.rename(columns={"DPMP": "cod_mpio", "MPIO": "municipio_poblacion"}, inplace=True)

df_long = df_long[df_long["cod_mpio"].notna()]
df_long = df_long[~df_long["cod_mpio"].astype(str).str.lower().str.contains("total|dane", na=False)]

df_long["cod_mpio"] = (
    df_long["cod_mpio"]
    .astype(str)
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
    .str.zfill(5)
)

# -------------------------------------------------------------------------
# Step 3 — Cast types and filter to analysis window
# -------------------------------------------------------------------------

df_long["year"]     = pd.to_numeric(df_long["year"], errors="coerce").astype("Int64")
df_long["poblacion"] = pd.to_numeric(df_long["poblacion"], errors="coerce")

df_long = df_long.dropna(subset=["cod_mpio", "year", "poblacion"])
df_long = df_long[df_long["year"].between(2015, 2024)]
df_long = df_long.sort_values(["cod_mpio", "year"]).reset_index(drop=True)

# -------------------------------------------------------------------------
# Step 4 — Save Silver
# -------------------------------------------------------------------------

df_long.to_parquet(os.path.join(SILVER_PATH, "poblacion_silver.parquet"), index=False)
df_long.to_csv(os.path.join(SILVER_PATH, "poblacion_silver.csv"), index=False, encoding="utf-8-sig")

print("\n" + "=" * 55)
print("  QUALITY REPORT — SILVER POPULATION")
print("=" * 55)
print(f"  Observations       : {len(df_long):,}")
print(f"  Unique municipalities : {df_long['cod_mpio'].nunique():,}")
print(f"  Years              : {sorted(df_long['year'].dropna().unique().tolist())}")
print(f"  Null poblacion     : {df_long['poblacion'].isna().sum():,}")
print(f"  Population min     : {df_long['poblacion'].min():,.0f}")
print(f"  Population max     : {df_long['poblacion'].max():,.0f}")
print("=" * 55)

logging.info(f"Silver Population saved: {df_long.shape}")
print("\n[SUCCESS] Silver Population completed.")
