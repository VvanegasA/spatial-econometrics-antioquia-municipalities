# Source layout

Run all scripts from the **repository root** so `data/`, `logs/`, and `results/` resolve correctly (each script also `chdir`s to the root).

## Pipelines (medallion)

1. **Bronze:** `src/pipelines/01_bronze_dane.py`, `01_bronze_seguridad.py`, `01_bronze_simat.py`
2. **Silver:** `src/pipelines/02_silver_dane.py`, `02_silver_seguridad.py`, `02_silver_simat.py`
3. **Gold:** `src/pipelines/03_gold_panel.py`

## Analysis and models

- **ESDA:** `src/analysis/eda_spatial.py`
- **Spatial weights (Queen):** `src/models/build_W_queen.py` — typically before SDM
- **SDM:** `src/models/spatial_model_sdm.py`

Example:

```bash
python src/pipelines/01_bronze_dane.py
python src/analysis/eda_spatial.py
python src/models/build_W_queen.py
python src/models/spatial_model_sdm.py
```
