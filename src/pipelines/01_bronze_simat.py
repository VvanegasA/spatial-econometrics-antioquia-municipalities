# Bronze Layer: SIMAT — Net Secondary School Coverage
# Source: TerriData DNP — TerriData_Dim4.xlsx
# Loads the raw Excel as-is; no transformations here.
# Run from repo root: python src/pipelines/01_bronze_simat.py

import os
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

os.chdir(Path(__file__).resolve().parents[2])
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=f"logs/01_bronze_simat_{datetime.today().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

RAW_PATH    = os.path.join("data", "raw", "TerriData_Dim4.xlsx")
BRONZE_PATH = os.path.join("data", "bronze")
os.makedirs(BRONZE_PATH, exist_ok=True)

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"[ERROR] File not found: {RAW_PATH}")

print(f"[INFO] Loading: {RAW_PATH}")

xl = pd.ExcelFile(RAW_PATH, engine="openpyxl")
print(f"[INFO] Available sheets: {xl.sheet_names}")

df = pd.read_excel(RAW_PATH, sheet_name=0, engine="openpyxl")

print(f"[INFO] Shape: {df.shape}")
print("\n[INFO] Columns:")
for i, col in enumerate(df.columns):
    print(f"   [{i}] {col}")
print("\n[INFO] First 3 rows:")
print(df.head(3).to_string())

if "Indicador" in df.columns:
    print("\n[INFO] Indicator values (top 10):")
    print(df["Indicador"].value_counts().head(10).to_string())

if "Año" in df.columns:
    print("\n[INFO] Available years:")
    print(sorted(df["Año"].dropna().unique().tolist()))

df.to_parquet(os.path.join(BRONZE_PATH, "simat_bronze.parquet"), index=False)
df.to_csv(os.path.join(BRONZE_PATH, "simat_bronze.csv"), index=False, encoding="utf-8-sig")

logging.info(f"Bronze SIMAT saved: {df.shape}")
print("\n[SUCCESS] Bronze SIMAT completed.")