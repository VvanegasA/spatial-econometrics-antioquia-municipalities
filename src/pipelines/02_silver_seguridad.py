# Silver Layer: Security — Homicides by Municipality-Year
# Antioquia, 2015-2024
# Run from repo root: python src/pipelines/02_silver_seguridad.py
#
# Input : data/bronze/seguridad_bronze.parquet
# Output: data/silver/seguridad_silver.parquet
#         One row per municipality-year; key column: homicidios (count)

import os
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

os.chdir(Path(__file__).resolve().parents[2])
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=f"logs/02_silver_seguridad_{datetime.today().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

BRONZE_PATH = os.path.join("data", "bronze", "seguridad_bronze.parquet")
SILVER_PATH = os.path.join("data", "silver")
os.makedirs(SILVER_PATH, exist_ok=True)

if not os.path.exists(BRONZE_PATH):
    raise FileNotFoundError(
        "[ERROR] Run src/pipelines/01_bronze_seguridad.py first (from repo root)."
    )

df = pd.read_parquet(BRONZE_PATH)
print(f"[INFO] Bronze loaded: {df.shape}")

# -------------------------------------------------------------------------
# Step 1 — Keep only Antioquia records
# -------------------------------------------------------------------------

n_before = len(df)
df = df[df["DEPARTAMENTO"].str.upper().str.strip() == "ANTIOQUIA"]
print(f"[INFO] Filtered to Antioquia: {len(df):,} rows (from {n_before:,})")

# -------------------------------------------------------------------------
# Step 2 — Filter to intentional homicides only
# Print all crime types first so the filter value can be verified against
# the actual source data if the file is ever updated.
# -------------------------------------------------------------------------

print("\n[INFO] Crime types in Antioquia:")
print(df["Tipo_Delito"].value_counts().to_string())

df = df[df["Tipo_Delito"].str.strip() == "Homicidio-intencional"]
print(f"\n[INFO] After homicide filter: {len(df):,} rows")

if len(df) == 0:
    print("\n[WARNING] No records matched 'Homicidio-intencional'.")
    print("  Check the crime types printed above and adjust the filter value.")
    raise ValueError("Homicide filter returned empty — adjust Tipo_Delito value.")

# -------------------------------------------------------------------------
# Step 3 — Extract year from event date
# -------------------------------------------------------------------------

df["FECHA HECHO"] = pd.to_datetime(df["FECHA HECHO"], errors="coerce")
df["year"] = df["FECHA HECHO"].dt.year
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)

print(f"[INFO] Years available: {sorted(df['year'].unique().tolist())}")

# -------------------------------------------------------------------------
# Step 4 — Standardize DIVIPOLA to 5-digit string
# -------------------------------------------------------------------------

df["cod_mpio"] = (
    df["CODIGO DANE"]
    .astype(str)
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
    .str.zfill(5)
)
print(f"[INFO] DIVIPOLA sample: {df['cod_mpio'].head(3).tolist()}")

# -------------------------------------------------------------------------
# Step 5 — Restrict to the analysis window
# -------------------------------------------------------------------------

df = df[df["year"].between(2015, 2024)]
print(f"[INFO] After year filter (2015-2024): {len(df):,} rows")

# -------------------------------------------------------------------------
# Step 6 — Aggregate: total homicide count per municipality-year
# -------------------------------------------------------------------------

df_agg = (
    df.groupby(["cod_mpio", "year"])["CANTIDAD"]
    .sum()
    .reset_index()
    .rename(columns={"CANTIDAD": "homicidios"})
)

df_agg["homicidios"] = df_agg["homicidios"].astype(int)
df_agg = df_agg.sort_values(["cod_mpio", "year"]).reset_index(drop=True)

print(f"\n[INFO] Homicide panel: {df_agg.shape}")
print(f"  Unique municipalities: {df_agg['cod_mpio'].nunique()}")
print(f"  Unique years         : {sorted(df_agg['year'].unique().tolist())}")
print("\n[INFO] Sample:")
print(df_agg.head(10).to_string())

# -------------------------------------------------------------------------
# Step 7 — Save Silver
# -------------------------------------------------------------------------

df_agg.to_parquet(os.path.join(SILVER_PATH, "seguridad_silver.parquet"), index=False)
df_agg.to_csv(os.path.join(SILVER_PATH, "seguridad_silver.csv"), index=False, encoding="utf-8-sig")

# -------------------------------------------------------------------------
# Step 8 — Quality report
# -------------------------------------------------------------------------

print("\n" + "=" * 55)
print("  QUALITY REPORT — SILVER SECURITY")
print("=" * 55)
print(f"  Observations       : {len(df_agg):,}")
print(f"  Unique municipalities : {df_agg['cod_mpio'].nunique():,}")
print(f"  Years              : {sorted(df_agg['year'].unique().tolist())}")
print(f"  Min homicides      : {df_agg['homicidios'].min()}")
print(f"  Max homicides      : {df_agg['homicidios'].max()}")
print(f"  Total homicides    : {df_agg['homicidios'].sum():,}")
print("=" * 55)

logging.info(f"Silver Security saved: {df_agg.shape}")
print("\n[SUCCESS] Silver Security completed.")