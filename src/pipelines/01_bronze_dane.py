# Bronze Layer: Municipal Value Added — DANE (2015-2024)
# Loads the raw Excel file as-is; no transformations here.
# Run from repo root: python src/pipelines/01_bronze_dane.py

import os
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

os.chdir(Path(__file__).resolve().parents[2])
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=f"logs/01_bronze_{datetime.today().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

RAW_PATH    = os.path.join("data", "raw", "PIB-VA_Mpal_2015-2024pr_Publ_02-09-2025_v2.xlsm")
BRONZE_PATH = os.path.join("data", "bronze")
os.makedirs(BRONZE_PATH, exist_ok=True)

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"[ERROR] File not found: {RAW_PATH}\n  Place it under data/raw/")

print(f"[INFO] Loading: {RAW_PATH}")

# Row 4 in Excel is the header (index 3 in Python)
df = pd.read_excel(
    RAW_PATH,
    sheet_name="PIB Mpal 2015-2024 Cons",
    header=3,
    engine="openpyxl"
)
print(f"[INFO] Value Added sheet — shape: {df.shape}")
print("\n[INFO] Detected columns:")
for i, col in enumerate(df.columns):
    print(f"   [{i}] {col}")
print("\n[INFO] First 3 rows:")
print(df.head(3).to_string())

df.to_parquet(os.path.join(BRONZE_PATH, "dane_bronze.parquet"), index=False)
df.to_csv(os.path.join(BRONZE_PATH, "dane_bronze.csv"), index=False, encoding="utf-8-sig")

# Population tab — header is on row 3 (index 2)
df_pob = pd.read_excel(
    RAW_PATH,
    sheet_name="POBLACION",
    header=2,
    engine="openpyxl"
)
print(f"\n[INFO] Population sheet — shape: {df_pob.shape}")

df_pob.to_parquet(os.path.join(BRONZE_PATH, "poblacion_bronze.parquet"), index=False)
df_pob.to_csv(os.path.join(BRONZE_PATH, "poblacion_bronze.csv"), index=False, encoding="utf-8-sig")

logging.info(f"Bronze VA saved: {df.shape}")
logging.info(f"Bronze Population saved: {df_pob.shape}")
print("\n[SUCCESS] Bronze DANE completed.")