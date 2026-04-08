# Bronze Layer: Security — Crimes Dataset (national)
# Loads the raw parquet as-is; no transformations here.
# Run from repo root: python src/pipelines/01_bronze_seguridad.py

import os
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

os.chdir(Path(__file__).resolve().parents[2])
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=f"logs/01_bronze_seguridad_{datetime.today().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

RAW_PATH    = os.path.join("data", "raw", "delitos_colombia.parquet")
BRONZE_PATH = os.path.join("data", "bronze")
os.makedirs(BRONZE_PATH, exist_ok=True)

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"[ERROR] File not found: {RAW_PATH}\n  Place it under data/raw/")

print(f"[INFO] Loading: {RAW_PATH}")

df = pd.read_parquet(RAW_PATH)

print(f"[INFO] Shape: {df.shape}")
print("\n[INFO] Columns:")
for i, col in enumerate(df.columns):
    print(f"   [{i}] {col}")
print("\n[INFO] First 3 rows:")
print(df.head(3).to_string())
print("\n[INFO] Department counts (top 10):")
print(df["DEPARTAMENTO"].value_counts().head(10).to_string())
print("\n[INFO] Crime types:")
print(df["Tipo_Delito"].value_counts().to_string())

df.to_parquet(os.path.join(BRONZE_PATH, "seguridad_bronze.parquet"), index=False)
df.to_csv(os.path.join(BRONZE_PATH, "seguridad_bronze.csv"), index=False, encoding="utf-8-sig")

logging.info(f"Bronze security saved: {df.shape}")
print("\n[SUCCESS] Bronze Security completed.")