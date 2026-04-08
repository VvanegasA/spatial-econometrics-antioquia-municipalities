#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Queen-Contiguity Spatial Weight Matrix (W) for Antioquia Municipalities
==============================================================================

Constructs, validates, and persists the spatial weight matrix W used by
the panel econometric models. W is built from the official municipality
shapefile using Queen contiguity (shared border or corner = neighbors).

The matrix is row-standardized (each row sums to 1) so that the spatial lag
Wy represents a proper weighted average across neighbors — a requirement for
the SAR and SEM estimators in spreg.

Outputs saved to data/gold/:
  - W_queen.pkl          Python object (used directly by the model scripts)
  - W_queen.gal          GeoDa-compatible format (for external tools / R / Stata)
  - W_queen_info.json    Metadata and neighbor statistics
  - W_neighbors_summary.csv  Readable summary per municipality

Run from repo root: python src/models/build_W_queen.py
"""

import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[2])

import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen, w_subset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)

RAW_PATH  = Path("data/raw")
GOLD_PATH = Path("data/gold")
GOLD_PATH.mkdir(exist_ok=True)

SHP_FILE = RAW_PATH / "shp" / "Municipios.shp"

print("=" * 70)
print("  Building W — Queen Contiguity, Antioquia Municipalities")
print("=" * 70)

# -------------------------------------------------------------------------
# Step 1 — Load shapefile
# The shapefile already covers only the 125 Antioquia municipalities.
# -------------------------------------------------------------------------

print(f"\n[INFO] Shapefile: {SHP_FILE}")
print(f"[INFO] Exists   : {SHP_FILE.exists()}")

if not SHP_FILE.exists():
    print(f"[ERROR] Shapefile not found: {SHP_FILE}")
    sys.exit(1)

gdf = gpd.read_file(SHP_FILE)

print(f"\n[INFO] Shapefile loaded: {len(gdf)} features")
print(f"[INFO] Columns : {list(gdf.columns)}")
print(f"[INFO] CRS     : {gdf.crs}")
print("\n[INFO] Sample rows:")
print(gdf[["COD_MPIO", "MPIO_NOMBR"]].head(3).to_string(index=False))

# -------------------------------------------------------------------------
# Step 2 — Verify municipality count
# -------------------------------------------------------------------------

print(f"\n[INFO] Municipalities found: {len(gdf)} (expected: 125)")

if len(gdf) != 125:
    print(f"[WARNING] Expected 125 municipalities, got {len(gdf)}. Check the shapefile.")

# -------------------------------------------------------------------------
# Step 3 — Prepare DIVIPOLA identifier
# W uses DIVIPOLA codes as node IDs so they can be joined directly to the
# panel dataset without any additional key mapping.
# -------------------------------------------------------------------------

gdf["cod_mpio"] = gdf["COD_MPIO"].astype(str).str.strip()

print("\n[INFO] DIVIPOLA samples:")
for _, row in gdf[["cod_mpio", "MPIO_NOMBR"]].head(5).iterrows():
    print(f"  {row['cod_mpio']} -> {row['MPIO_NOMBR']}")

# -------------------------------------------------------------------------
# Step 4 — Build Queen contiguity matrix
# Two municipalities are neighbors if they share a border or a vertex.
# This is the most common choice in regional econometrics; it captures
# more local interactions than Rook (border only).
# -------------------------------------------------------------------------

print("\n[INFO] Building W (Queen contiguity)...")
w = Queen.from_dataframe(gdf, idVariable="cod_mpio")
print("[INFO] W built.")

# -------------------------------------------------------------------------
# Step 5 — Row-standardize
# Dividing each row by its sum makes Wy an unweighted neighborhood average,
# which is required by spreg Panel_FE_Lag and Panel_FE_Error.
# -------------------------------------------------------------------------

w.transform = "r"

sample_id      = list(w.neighbors.keys())[0]
sample_weights = w.weights[sample_id]
print(f"\n[INFO] Row-standardized. Sample check for {sample_id}:")
print(f"  Neighbors : {len(w.neighbors[sample_id])}")
print(f"  Weight sum: {sum(sample_weights):.4f}  (should be 1.0)")

# -------------------------------------------------------------------------
# Step 6 — Connectivity statistics
# -------------------------------------------------------------------------

cardinalities = list(w.cardinalities.values())
n = w.n

print(f"""
[INFO] W summary
  Municipalities  : {n}
  Total connections : {w.s0:.0f}
  Density           : {w.s0 / (n * (n - 1)):.4f}
  Neighbors — mean  : {np.mean(cardinalities):.1f}
  Neighbors — median: {np.median(cardinalities):.1f}
  Neighbors — min   : {np.min(cardinalities)}
  Neighbors — max   : {np.max(cardinalities)}
  Neighbors — std   : {np.std(cardinalities):.2f}
""")

# Most and least connected
max_n = max(cardinalities)
min_n = min(cardinalities)

print(f"[INFO] Municipality with most neighbors ({max_n}):")
for mpio, cn in w.cardinalities.items():
    if cn == max_n:
        name = gdf[gdf["cod_mpio"] == mpio]["MPIO_NOMBR"].values
        if len(name):
            print(f"  {mpio} - {name[0]}")

print(f"\n[INFO] Municipality with fewest neighbors ({min_n}):")
for mpio, cn in w.cardinalities.items():
    if cn == min_n:
        name = gdf[gdf["cod_mpio"] == mpio]["MPIO_NOMBR"].values
        if len(name):
            print(f"  {mpio} - {name[0]}")

isolated = [m for m, c in w.cardinalities.items() if c == 0]
if isolated:
    print(f"\n[WARNING] {len(isolated)} island(s) detected (no neighbors): {isolated}")
else:
    print("\n[INFO] No isolated municipalities — all have at least one neighbor.")

# -------------------------------------------------------------------------
# Step 7 — Spot-check known neighborhoods
# -------------------------------------------------------------------------

for cod, label in [("05001", "Medellin"), ("05002", "Abejorral")]:
    if cod in w.neighbors:
        vecinos = w.neighbors[cod]
        names = [
            gdf[gdf["cod_mpio"] == v]["MPIO_NOMBR"].values[0]
            for v in vecinos[:6]
            if len(gdf[gdf["cod_mpio"] == v]) > 0
        ]
        print(f"\n[INFO] {label} ({cod}) — {len(vecinos)} neighbors: {', '.join(names)}")

# -------------------------------------------------------------------------
# Step 8 — Validate alignment with Gold panel
# The W must cover exactly the municipalities present in the panel; any gap
# would cause the spatial estimator to crash or produce wrong results.
# -------------------------------------------------------------------------

print("\n[INFO] Validating against gold panel...")
panel = pd.read_parquet(GOLD_PATH / "panel_gold.parquet")

panel_mpios = set(str(x).zfill(5) for x in panel["cod_mpio"].unique())
w_mpios     = set(w.neighbors.keys())

missing_in_w   = panel_mpios - w_mpios
extra_in_w     = w_mpios - panel_mpios

print(f"  Panel municipalities : {len(panel_mpios)}")
print(f"  W municipalities     : {len(w_mpios)}")

if missing_in_w:
    print(f"  [WARNING] In panel but missing from W: {list(missing_in_w)[:5]}")
else:
    print("  [OK] All panel municipalities are covered by W.")

if extra_in_w:
    print(f"  [INFO] In W but not in panel (will be subsetted at model time): {len(extra_in_w)}")

# -------------------------------------------------------------------------
# Step 9 — Persist W in multiple formats
# -------------------------------------------------------------------------

# Pickle — primary format for the model scripts
pkl_path = GOLD_PATH / "W_queen.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(w, f)
print(f"\n[SUCCESS] Saved pickle : {pkl_path}")

# GAL — interoperable with GeoDa, R (spdep), Stata
gal_path = GOLD_PATH / "W_queen.gal"
try:
    w.to_file(str(gal_path))
    print(f"[SUCCESS] Saved GAL   : {gal_path}")
except Exception as e:
    # Fall back to manual GAL write if libpysal's exporter fails
    with open(gal_path, "w") as f:
        f.write(f"0 {w.n} Queen contiguity\n")
        for mpio in w.neighbors:
            vecinos = w.neighbors[mpio]
            f.write(f"{mpio} {len(vecinos)}\n")
            if vecinos:
                f.write(" ".join(vecinos) + "\n")
    print(f"[SUCCESS] Saved GAL (manual fallback): {gal_path}")

# JSON metadata — useful for documentation and reproducibility checks
json_path = GOLD_PATH / "W_queen_info.json"
metadata = {
    "created"       : datetime.now().isoformat(),
    "method"        : "Queen contiguity",
    "transformation": "row-standardized (r)",
    "n_municipios"  : n,
    "n_connections" : int(w.s0),
    "density"       : w.s0 / (n * (n - 1)),
    "neighbors_stats": {
        "mean"  : float(np.mean(cardinalities)),
        "median": float(np.median(cardinalities)),
        "min"   : int(np.min(cardinalities)),
        "max"   : int(np.max(cardinalities)),
        "std"   : float(np.std(cardinalities))
    },
    "id_column"      : "cod_mpio",
    "shapefile"      : str(SHP_FILE),
    "validation"     : {
        "panel_municipios": len(panel_mpios),
        "w_municipios"    : len(w_mpios),
        "match"           : len(panel_mpios & w_mpios)
    },
    "municipios_list": sorted(list(w.neighbors.keys()))
}

with open(json_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"[SUCCESS] Saved JSON  : {json_path}")

# -------------------------------------------------------------------------
# Step 10 — Save readable CSV summary (one row per municipality)
# -------------------------------------------------------------------------

summary_data = []
for mpio in w.neighbors:
    n_neighbors = len(w.neighbors[mpio])
    name = gdf[gdf["cod_mpio"] == mpio]["MPIO_NOMBR"].values
    name = name[0] if len(name) > 0 else "NA"
    sub  = panel[panel["cod_mpio"] == mpio]["subregion"].unique()
    sub  = sub[0] if len(sub) > 0 else "NA"
    nb_sample = ";".join(w.neighbors[mpio][:5])
    if len(w.neighbors[mpio]) > 5:
        nb_sample += "..."
    summary_data.append({
        "cod_mpio"      : mpio,
        "municipio"     : name,
        "subregion"     : sub,
        "n_neighbors"   : n_neighbors,
        "neighbor_codes": nb_sample
    })

summary_df = (pd.DataFrame(summary_data)
              .sort_values("cod_mpio")
              .reset_index(drop=True))
out_csv = GOLD_PATH / "W_neighbors_summary.csv"
summary_df.to_csv(out_csv, index=False)
print(f"[SUCCESS] Saved summary: {out_csv}")

# -------------------------------------------------------------------------
# Final summary
# -------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  W CONSTRUCTION COMPLETE")
print("=" * 70)
print(f"""
  Municipalities : {n}
  Connections    : {w.s0:.0f}
  Avg neighbors  : {np.mean(cardinalities):.1f}
  Method         : Queen contiguity, row-standardized
  Saved to       : data/gold/
    - W_queen.pkl
    - W_queen.gal
    - W_queen_info.json
    - W_neighbors_summary.csv

  Next step: run src/models/spatial_model_sdm.py
""")

logger.info(f"W built successfully: {n} nodes, {w.s0:.0f} connections.")
