# Silver Layer: SIMAT — Net Secondary School Coverage
# Antioquia, 2015-2023 (SIMAT has a one-year reporting lag; 2024 not yet available)
# Run from repo root: python src/pipelines/02_silver_simat.py
#
# Input : data/bronze/simat_bronze.parquet
# Output: data/silver/simat_silver.parquet
#         One row per municipality-year; key column: cobertura_secundaria (%)

import os
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

os.chdir(Path(__file__).resolve().parents[2])
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=f"logs/02_silver_simat_{datetime.today().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

BRONZE_PATH = os.path.join("data", "bronze", "simat_bronze.parquet")
SILVER_PATH = os.path.join("data", "silver")
os.makedirs(SILVER_PATH, exist_ok=True)

if not os.path.exists(BRONZE_PATH):
    raise FileNotFoundError(
        "[ERROR] Run src/pipelines/01_bronze_simat.py first (from repo root)."
    )

df = pd.read_parquet(BRONZE_PATH)
print(f"[INFO] Bronze loaded: {df.shape}")

# -------------------------------------------------------------------------
# Step 1 — Rename columns to our standard schema
# -------------------------------------------------------------------------

df = df.rename(columns={
    "Departamento"   : "departamento",
    "Código Entidad" : "cod_mpio",
    "Entidad"        : "municipio",
    "Indicador"      : "indicador",
    "Dato Numérico"  : "cobertura_secundaria",
    "Año"            : "year"
})
print(f"[INFO] Renamed columns: {df.columns.tolist()}")

# -------------------------------------------------------------------------
# Step 2 — Keep only Antioquia
# -------------------------------------------------------------------------

print("\n[INFO] Department values (top 10):")
print(df["departamento"].value_counts().head(10).to_string())

n_before = len(df)
df = df[df["departamento"].str.upper().str.strip() == "ANTIOQUIA"]
print(f"\n[INFO] Filtered to Antioquia: {len(df):,} rows (from {n_before:,})")

# -------------------------------------------------------------------------
# Step 3 — Filter years and drop the departmental aggregate row (cod 05000)
# -------------------------------------------------------------------------

df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
df = df[df["year"].between(2015, 2023)]
df = df[df["cod_mpio"] != "05000"]  # departmental total, not a municipality

print(f"[INFO] After year filter (2015-2023): {len(df):,} rows")
print(f"[INFO] Years available: {sorted(df['year'].dropna().unique().tolist())}")

# -------------------------------------------------------------------------
# Step 4 — Standardize DIVIPOLA to 5-digit string
# -------------------------------------------------------------------------

df["cod_mpio"] = (
    df["cod_mpio"]
    .astype(str)
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
    .str.zfill(5)
)
print(f"[INFO] DIVIPOLA sample: {df['cod_mpio'].head(3).tolist()}")

# -------------------------------------------------------------------------
# Step 5 — Clean and cast coverage variable
# -------------------------------------------------------------------------

# Drop indicator column — it's redundant once we know the file contains
# only the net secondary coverage indicator.
if "indicador" in df.columns:
    df = df.drop(columns=["indicador"])

# Source sometimes uses comma as decimal separator
df["cobertura_secundaria"] = (
    df["cobertura_secundaria"]
    .astype(str)
    .str.replace(",", ".", regex=False)
)
df["cobertura_secundaria"] = pd.to_numeric(df["cobertura_secundaria"], errors="coerce")

df = df.dropna(subset=["cod_mpio", "year", "cobertura_secundaria"])
df = df.sort_values(["cod_mpio", "year"]).reset_index(drop=True)

# -------------------------------------------------------------------------
# Step 6 — Save Silver
# -------------------------------------------------------------------------

df.to_parquet(os.path.join(SILVER_PATH, "simat_silver.parquet"), index=False)
df.to_csv(os.path.join(SILVER_PATH, "simat_silver.csv"), index=False, encoding="utf-8-sig")

# -------------------------------------------------------------------------
# Step 7 — Quality report
# -------------------------------------------------------------------------

print("\n" + "=" * 55)
print("  QUALITY REPORT — SILVER SIMAT")
print("=" * 55)
print(f"  Observations       : {len(df):,}")
print(f"  Unique municipalities : {df['cod_mpio'].nunique():,}")
print(f"  Years              : {sorted(df['year'].dropna().unique().tolist())}")
print(f"  Null coverage      : {df['cobertura_secundaria'].isna().sum():,}")
print(f"  Coverage min       : {df['cobertura_secundaria'].min():.1f}%")
print(f"  Coverage max       : {df['cobertura_secundaria'].max():.1f}%")
print(f"  Coverage mean      : {df['cobertura_secundaria'].mean():.1f}%")
print("=" * 55)
print("\n[INFO] First rows:")
print(df.head(6).to_string())

logging.info(f"Silver SIMAT saved: {df.shape}")
print("\n[SUCCESS] Silver SIMAT completed.")