# Silver Layer: Municipal Value Added — DANE (2015-2024)
# Cleans, standardizes DIVIPOLA codes, and pivots to long panel format.
# Run from repo root: python src/pipelines/02_silver_dane.py
#
# Excel layout (confirmed):
#   Col A  -> Year
#   Col E  -> Municipality code (DIVIPOLA 5-digit)
#   Col F  -> Sub-region
#   Col G  -> Municipality name
#   Col H-V -> Economic branches (VA by sector)
#   Col W  -> Total VA

import os
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parents[2])
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=f"logs/02_silver_{datetime.today().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

BRONZE_PATH = os.path.join("data", "bronze", "dane_bronze.parquet")
SILVER_PATH = os.path.join("data", "silver")
os.makedirs(SILVER_PATH, exist_ok=True)

if not os.path.exists(BRONZE_PATH):
    raise FileNotFoundError(
        "[ERROR] Run src/pipelines/01_bronze_dane.py first (from repo root)."
    )

df = pd.read_parquet(BRONZE_PATH)
print(f"[INFO] Bronze loaded: {df.shape}")
print("[INFO] Columns:", df.columns.tolist())

# -------------------------------------------------------------------------
# Step 1 — Map column positions to clean names
# Column indices are fixed by the Excel structure; adjust only if the source
# file layout changes.
# -------------------------------------------------------------------------

col_year     = df.columns[0]   # Col A
col_codmpio  = df.columns[4]   # Col E
col_subr     = df.columns[5]   # Col F
col_mpio     = df.columns[6]   # Col G
col_va_total = df.columns[22]  # Col W

# Economic branches: cols H to V (indices 7-21)
cols_ramas = df.columns[7:22].tolist()

print(f"\n[INFO] year col      : {col_year}")
print(f"[INFO] cod_mpio col  : {col_codmpio}")
print(f"[INFO] subregion col : {col_subr}")
print(f"[INFO] municipio col : {col_mpio}")
print(f"[INFO] va_total col  : {col_va_total}")
print(f"[INFO] branch cols   : {cols_ramas}")

# -------------------------------------------------------------------------
# Step 2 — Select and rename columns
# -------------------------------------------------------------------------

cols_usar = [col_year, col_codmpio, col_subr, col_mpio, col_va_total] + cols_ramas
df = df[cols_usar].copy()

rename = {
    col_year    : "year",
    col_codmpio : "cod_mpio",
    col_subr    : "subregion",
    col_mpio    : "municipio",
    col_va_total: "va_total"
}
df = df.rename(columns=rename)

# Standardize branch column names to rama_01, rama_02, ...
ramas_rename = {col: f"rama_{str(i+1).zfill(2)}" for i, col in enumerate(cols_ramas)}
df = df.rename(columns=ramas_rename)
cols_ramas_limpias = list(ramas_rename.values())

print(f"\n[INFO] Final columns: {df.columns.tolist()}")

# -------------------------------------------------------------------------
# Step 3 — Drop non-municipality rows (subtotals, blanks, header rows)
# -------------------------------------------------------------------------

n_before = len(df)
df = df[df["cod_mpio"].notna()]
df = df[~df["cod_mpio"].astype(str).str.lower().str.contains(
    "total|nacional|region|subtotal|código|nan", na=False)]
print(f"\n[INFO] Rows dropped (totals/nulls): {n_before - len(df)}")

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
# Step 5 — Cast data types
# -------------------------------------------------------------------------

df["year"]     = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
df["va_total"] = pd.to_numeric(df["va_total"], errors="coerce")

for col in cols_ramas_limpias:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["cod_mpio", "year", "va_total"])
df = df[df["year"].between(2015, 2024)]

# -------------------------------------------------------------------------
# Step 6 — Dependent variable: log of total VA
# Small clip prevents log(0) on municipalities with near-zero activity.
# -------------------------------------------------------------------------

df["ln_va_total"] = np.log(df["va_total"].clip(lower=0.001))

# -------------------------------------------------------------------------
# Step 7 — Save Silver (one row = one municipality-year)
# -------------------------------------------------------------------------

df = df.sort_values(["cod_mpio", "year"]).reset_index(drop=True)

df.to_parquet(os.path.join(SILVER_PATH, "dane_silver.parquet"), index=False)
df.to_csv(os.path.join(SILVER_PATH, "dane_silver.csv"), index=False, encoding="utf-8-sig")

# -------------------------------------------------------------------------
# Step 8 — Quality report
# -------------------------------------------------------------------------

print("\n" + "=" * 55)
print("  QUALITY REPORT — SILVER DANE")
print("=" * 55)
print(f"  Observations       : {len(df):,}")
print(f"  Unique municipalities : {df['cod_mpio'].nunique():,}")
print(f"  Years              : {sorted(df['year'].dropna().unique().tolist())}")
print(f"  Null va_total      : {df['va_total'].isna().sum():,}")
print(f"  VA min             : {df['va_total'].min():,.1f} million COP")
print(f"  VA max             : {df['va_total'].max():,.1f} million COP")
print("=" * 55)

logging.info(f"Silver DANE saved: {df.shape}")
print("\n[SUCCESS] Silver DANE completed.")
