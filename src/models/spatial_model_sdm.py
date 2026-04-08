#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial Panel Econometrics — Antioquia Municipalities (2015-2023)
=================================================================

Estimation workflow (Anselin 1988; Elhorst 2014):

  1. Panel OLS with municipal fixed effects (within-estimator)
  2. Moran's I on OLS residuals — checks if spatial dependence persists
     after controlling for time-invariant heterogeneity
  3. Lagrange Multiplier diagnostics on the panel:
       LM-Lag    -> tests for SAR structure (rho != 0)
       LM-Error  -> tests for SEM structure (lambda != 0)
       RLM-Lag   -> robust version (controls for error dependence)
       RLM-Error -> robust version (controls for lag dependence)
  4. Model selection rule (robust LM statistics take priority):
       RLM-Lag sig, RLM-Error not   -> Panel SAR  (Panel_FE_Lag)
       RLM-Error sig, RLM-Lag not   -> Panel SEM  (Panel_FE_Error)
       Both significant              -> pick the one with larger statistic
       Neither significant           -> keep Panel OLS FE
  5. Direct, indirect, and total effects (LeSage & Pace 2009)

Dependent variable : ln_va_per_capita  (log VA per capita, constant prices)
Regressors         :
  - tasa_homicidios      (homicides per 100k — security proxy)
  - cobertura_secundaria (net secondary coverage — human capital proxy)

W reference: data/gold/W_queen.pkl  (Queen contiguity, row-standardized)

Sample: 2015-2023 (9 years, 125 municipalities, N×T = 1,125 obs).
2024 is excluded because SIMAT coverage data has a one-year reporting lag.

Run from repo root: python src/models/spatial_model_sdm.py
"""

import logging
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[2])

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from spreg import PanelFE, Panel_FE_Lag, Panel_FE_Error
from spreg.diagnostics_panel import (
    panel_LMlag, panel_LMerror,
    panel_rLMlag, panel_rLMerror
)
from esda import Moran

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

logging.basicConfig(
    filename=f"logs/spatial_model_{datetime.today().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

GOLD_PATH    = Path("data/gold")
RESULTS_PATH = Path("results")

DEP_VAR  = "ln_va_per_capita"
COVARS   = ["tasa_homicidios", "cobertura_secundaria"]
ID_VAR   = "cod_mpio"
TIME_VAR = "year"

# 2024 excluded: SIMAT coverage not yet available
T_START, T_END = 2015, 2023

SEP = "=" * 72


# ===========================================================================
# Section 1 — Load data
# ===========================================================================

def load_data():
    """Load the Gold panel and the pre-built Queen W matrix."""
    print(f"\n{SEP}")
    print("  SECTION 1 — DATA LOADING")
    print(SEP)

    panel_file = GOLD_PATH / "panel_gold.parquet"
    if not panel_file.exists():
        raise FileNotFoundError(f"[ERROR] Run 03_gold_panel.py first: {panel_file}")

    df = pd.read_parquet(panel_file)
    df = df[df[TIME_VAR].between(T_START, T_END)].copy()

    print(f"\n  Panel: {T_START}-{T_END}, {df[ID_VAR].nunique()} municipalities")
    print(f"  Observations: {len(df)}")

    # Drop rows with missing model variables
    cols_needed = [ID_VAR, TIME_VAR, DEP_VAR] + COVARS
    n_before = len(df)
    df = df[cols_needed].dropna()
    if n_before > len(df):
        print(f"  [WARNING] Dropped {n_before - len(df)} rows with NaN in model variables")

    # Panel balance check
    n_mpios    = df[ID_VAR].nunique()
    n_years    = df[TIME_VAR].nunique()
    n_obs      = len(df)
    balanced   = n_obs == n_mpios * n_years
    print(f"\n  Balance: {n_mpios} mpios × {n_years} years = {n_mpios * n_years} expected")
    print(f"  Status : {'BALANCED' if balanced else 'UNBALANCED'}")

    # spreg Panel expects data ordered by [ID, TIME] — municipality varies slowly
    df = df.sort_values([ID_VAR, TIME_VAR]).reset_index(drop=True)

    # Load W
    w_file = GOLD_PATH / "W_queen.pkl"
    if not w_file.exists():
        raise FileNotFoundError(
            f"[ERROR] Run src/models/build_W_queen.py first: {w_file}"
        )
    with open(w_file, "rb") as f:
        w = pickle.load(f)

    # Align W with panel (subset if necessary)
    mpios_panel = sorted(df[ID_VAR].unique().tolist())
    mpios_w     = sorted(w.neighbors.keys())

    missing_in_w = set(mpios_panel) - set(mpios_w)
    extra_in_w   = set(mpios_w) - set(mpios_panel)

    if missing_in_w:
        print(f"\n  [WARNING] Panel municipalities not in W: {missing_in_w}")
        df = df[~df[ID_VAR].isin(missing_in_w)]
        mpios_panel = sorted(df[ID_VAR].unique().tolist())

    from libpysal.weights import w_subset
    if extra_in_w:
        w = w_subset(w, mpios_panel)
        w.transform = "r"

    print(f"\n  W loaded: {w.n} nodes, avg neighbors: "
          f"{np.mean(list(w.cardinalities.values())):.1f}")

    logger.info(f"Data loaded: {df.shape}, W: {w.n} nodes")
    return df, w


# ===========================================================================
# Section 2 — Prepare vectors for spreg
# ===========================================================================

def prepare_vectors(df, w):
    """
    spreg panel estimators expect:
      y : (N*T, 1) array ordered [ID, TIME] — municipality varies slowly
      x : (N*T, k) array — no intercept column (FE estimator demeans internally)
      w : libpysal W object with N nodes (not N*T)
    """
    df = df.sort_values([ID_VAR, TIME_VAR]).reset_index(drop=True)

    mpios_panel = sorted(df[ID_VAR].unique().tolist())
    mpios_w     = sorted(w.neighbors.keys())
    assert mpios_panel == mpios_w, (
        "Panel and W municipality IDs do not match. "
        "Re-run build_W_queen.py and 03_gold_panel.py."
    )

    y = df[DEP_VAR].values.reshape(-1, 1)
    X = df[COVARS].values   # no constant — Panel_FE removes it via demeaning

    n_mpios = df[ID_VAR].nunique()
    n_years = df[TIME_VAR].nunique()
    n_obs   = len(df)

    print(f"\n  Vectors: y = ({n_obs}, 1), X = ({n_obs}, {len(COVARS)})")
    print(f"  N = {n_mpios} municipalities, T = {n_years} periods")

    return y, X, df, n_mpios, n_years


# ===========================================================================
# Section 3 — Panel OLS with fixed effects
# ===========================================================================

def ols_panel_fe(y, X, w, df, n_mpios, n_years):
    """
    Within-estimator: demeans y and X by municipality to remove the fixed effect.
    This controls for all time-invariant unobserved heterogeneity
    (geography, historical endowments, institutions) without modelling it directly.

    Specification:
      ln_va_per_capita_it = alpha_i + beta1*tasa_homicidios_it + beta2*cobertura_secundaria_it + eps_it
    """
    print(f"\n{SEP}")
    print("  SECTION 3 — PANEL OLS WITH FIXED EFFECTS (WITHIN)")
    print(SEP)

    ols_fe = PanelFE(
        y, X, w,
        name_y=DEP_VAR,
        name_x=COVARS,
        name_ds="Antioquia 2015-2023"
    )

    print(f"\n  Pseudo-R² : {ols_fe.pr2:.4f}")
    print(f"  Log-Lik   : {ols_fe.logll:.4f}")
    print(f"\n  {'Variable':<25} {'Coef':>10} {'Std.Err':>10} {'t-stat':>10} {'p-value':>10}")
    print("  " + "-" * 65)

    for i, var in enumerate(COVARS):
        coef  = ols_fe.betas[i, 0]
        se    = ols_fe.std_err[i]
        tstat = ols_fe.z_stat[i][0]
        pval  = ols_fe.z_stat[i][1]
        stars = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        print(f"  {var:<25} {coef:>10.4f} {se:>10.4f} {tstat:>10.3f} {pval:>10.4f} {stars}")

    logger.info(f"OLS FE: pr2={ols_fe.pr2:.4f}")
    return ols_fe


# ===========================================================================
# Section 4 — Spatial diagnostics: Moran's I + LM tests
# ===========================================================================

def spatial_diagnostics(ols_fe, y, X, w, df):
    """
    Two complementary checks for spatial dependence in the OLS residuals:

    Moran's I (cross-sectional, municipality means): a quick visual/descriptive
    check — statistically positive I after FE suggests spatial structure remains.

    LM tests (Elhorst 2014): formal panel-adapted tests that distinguish
    between spatial lag dependence (SAR, rho != 0) and error dependence
    (SEM, lambda != 0). The robust versions (RLM) control for the other
    alternative, so they are used for the actual model selection decision.
    """
    print(f"\n{SEP}")
    print("  SECTION 4 — SPATIAL DIAGNOSTICS")
    print(SEP)

    # -- Moran's I on municipality-averaged residuals --
    print("\n  4a. Moran's I on OLS-FE residuals (averaged by municipality)\n")

    df_temp = df.copy()
    df_temp["residual"] = ols_fe.u.flatten()
    res_cs = df_temp.groupby(ID_VAR)["residual"].mean()
    res_cs = res_cs.reindex(sorted(w.neighbors.keys()))

    moran_res = Moran(res_cs.values, w)
    sig_res = ("***" if moran_res.p_sim < 0.01 else
               ("**" if moran_res.p_sim < 0.05 else
                ("*"  if moran_res.p_sim < 0.10 else "ns")))

    print(f"  Moran's I  : {moran_res.I:.4f}")
    print(f"  Z-score    : {moran_res.z_sim:.3f}")
    print(f"  P-value    : {moran_res.p_sim:.4f}  {sig_res}")

    if moran_res.p_sim < 0.05:
        print("\n  -> Spatial dependence detected in residuals; a spatial model is warranted.")
    else:
        print("\n  -> No clear spatial dependence in residuals. Panel OLS-FE may suffice.")

    # -- LM panel tests --
    print("\n  4b. LM panel spatial tests (Anselin 1988; Elhorst 2014)\n")
    print(f"  {'Test':<18} {'Statistic':>14} {'p-value':>10} {'Sig':>10}")
    print("  " + "-" * 56)

    results_lm = {}
    for name, fn in [("LM-Lag",   panel_LMlag),   ("LM-Error", panel_LMerror),
                     ("RLM-Lag",  panel_rLMlag),  ("RLM-Error", panel_rLMerror)]:
        try:
            stat, pval = fn(y, X, w)
            sig = ("***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else "ns")))
            results_lm[name] = {"stat": stat, "pval": pval, "sig": sig, "sign": pval < 0.05}
            print(f"  {name:<18} {stat:>14.4f} {pval:>10.4f} {sig:>10}")
        except Exception as e:
            results_lm[name] = {"stat": np.nan, "pval": np.nan, "sig": "ERR", "sign": False}
            print(f"  {name:<18} {'ERROR':>14}   {str(e)[:30]}")

    print("  " + "-" * 56)
    print("  Significance: *** p<0.01, ** p<0.05, * p<0.1, ns = not significant")

    logger.info(f"LM tests: {results_lm}")
    return moran_res, results_lm


# ===========================================================================
# Section 5 — Model selection
# ===========================================================================

def select_model(moran_res, results_lm):
    """
    Standard decision rule (Anselin 1988; Florax et al. 2003).
    Robust LM statistics take priority over unadjusted ones because they
    control for the alternative form of spatial dependence.

    1. RLM-Lag sig, RLM-Error not   -> SAR (Panel_FE_Lag)
    2. RLM-Error sig, RLM-Lag not   -> SEM (Panel_FE_Error)
    3. Both significant              -> pick the larger statistic
    4. Neither significant           -> Panel OLS FE is sufficient
    """
    print(f"\n{SEP}")
    print("  SECTION 5 — MODEL SELECTION")
    print(SEP)

    rlm_lag   = results_lm.get("RLM-Lag",   {"stat": 0, "pval": 1, "sign": False})
    rlm_error = results_lm.get("RLM-Error", {"stat": 0, "pval": 1, "sign": False})

    if not rlm_lag["sign"] and not rlm_error["sign"]:
        decision = "OLS_FE"
        reason   = ("Neither RLM-Lag nor RLM-Error is significant at 5%.\n"
                    "  Panel OLS with fixed effects is sufficient;\n"
                    "  no evidence of spatial dependence beyond the FE.")
    elif rlm_lag["sign"] and not rlm_error["sign"]:
        decision = "SAR"
        reason   = ("RLM-Lag significant, RLM-Error not -> SAR (Panel_FE_Lag).\n"
                    "  VA per capita spillovers across neighboring municipalities.")
    elif rlm_error["sign"] and not rlm_lag["sign"]:
        decision = "SEM"
        reason   = ("RLM-Error significant, RLM-Lag not -> SEM (Panel_FE_Error).\n"
                    "  Spatial dependence in errors: unobserved common shocks across neighbors.")
    else:
        if rlm_lag["stat"] >= rlm_error["stat"]:
            decision = "SAR"
            reason   = (f"Both RLM significant; RLM-Lag ({rlm_lag['stat']:.3f}) "
                        f"> RLM-Error ({rlm_error['stat']:.3f}) -> SAR.")
        else:
            decision = "SEM"
            reason   = (f"Both RLM significant; RLM-Error ({rlm_error['stat']:.3f}) "
                        f"> RLM-Lag ({rlm_lag['stat']:.3f}) -> SEM.")

    print(f"\n  Selected model : {decision}")
    print(f"  Reason         : {reason}")
    logger.info(f"Model selected: {decision}")
    return decision


# ===========================================================================
# Section 6 — Estimate the selected model
# ===========================================================================

def estimate_model(decision, y, X, w, df):
    """Estimate whichever model the diagnostic sequence selected."""
    print(f"\n{SEP}")
    print(f"  SECTION 6 — ESTIMATION: {decision}")
    print(SEP)

    if decision == "OLS_FE":
        print("\n  OLS-FE was estimated in Section 3. No spatial model needed.")
        return None

    elif decision == "SAR":
        print("""
  SAR with municipal fixed effects:
    ln_va_per_capita_it = rho * W * ln_va_per_capita_it + beta' * X_it + alpha_i + eps_it

  rho > 0 and significant: VA per capita in a municipality is positively
  correlated with the average VA per capita of its geographic neighbors.
""")
        modelo = Panel_FE_Lag(
            y, X, w,
            name_y=DEP_VAR,
            name_x=COVARS,
            name_ds="Antioquia 2015-2023"
        )
        param_name  = "rho"
        param_value = modelo.rho

    elif decision == "SEM":
        print("""
  SEM with municipal fixed effects:
    ln_va_per_capita_it = beta' * X_it + alpha_i + u_it
    u_it = lambda * W * u_it + eps_it

  lambda > 0 and significant: unobserved shocks propagate across neighbors
  (e.g., climate events, commodity price cycles, regional policy spillovers).
""")
        modelo = Panel_FE_Error(
            y, X, w,
            name_y=DEP_VAR,
            name_x=COVARS,
            name_ds="Antioquia 2015-2023"
        )
        param_name  = "lambda"
        param_value = modelo.lam

    # -- Results table --
    print(f"\n  {'Variable':<25} {'Coef':>10} {'Std.Err':>10} {'z-stat':>10} {'p-value':>10}")
    print("  " + "-" * 65)

    for i, var in enumerate(COVARS):
        coef  = modelo.betas[i, 0]
        se    = modelo.std_err[i]
        zval  = modelo.z_stat[i][0]
        pval  = modelo.z_stat[i][1]
        stars = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        print(f"  {var:<25} {coef:>10.4f} {se:>10.4f} {zval:>10.3f} {pval:>10.4f} {stars}")

    idx_sp = len(COVARS)
    sp_se  = modelo.std_err[idx_sp] if len(modelo.std_err) > idx_sp else np.nan
    sp_z   = modelo.z_stat[idx_sp][0] if len(modelo.z_stat) > idx_sp else np.nan
    sp_p   = modelo.z_stat[idx_sp][1] if len(modelo.z_stat) > idx_sp else np.nan
    sp_stars = "***" if sp_p < 0.01 else ("**" if sp_p < 0.05 else ("*" if sp_p < 0.1 else ""))

    print("  " + "-" * 65)
    print(f"  {param_name:<25} {param_value:>10.4f} {sp_se:>10.4f} {sp_z:>10.3f} {sp_p:>10.4f} {sp_stars}")

    print(f"\n  Log-Likelihood : {modelo.logll:.4f}")
    print(f"  AIC            : {modelo.aic:.4f}")

    logger.info(f"{decision}: {param_name}={param_value:.4f}, LL={modelo.logll:.4f}")
    return modelo


# ===========================================================================
# Section 7 — Save results
# ===========================================================================

def save_results(decision, ols_fe, modelo_espacial, moran_res, results_lm, df):
    """Write coefficient tables and diagnostic statistics to results/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    rows = []

    for i, var in enumerate(COVARS):
        rows.append({
            "model"   : "OLS_FE",
            "variable": var,
            "coef"    : round(ols_fe.betas[i, 0], 5),
            "std_err" : round(ols_fe.std_err[i], 5)
        })

    if modelo_espacial is not None:
        for i, var in enumerate(COVARS):
            rows.append({
                "model"   : decision,
                "variable": var,
                "coef"    : round(modelo_espacial.betas[i, 0], 5),
                "std_err" : round(modelo_espacial.std_err[i], 5)
            })
        param_name = "rho" if decision == "SAR" else "lambda"
        param_val  = modelo_espacial.rho if decision == "SAR" else modelo_espacial.lam
        idx_sp     = len(COVARS)
        sp_se      = (modelo_espacial.std_err[idx_sp]
                      if len(modelo_espacial.std_err) > idx_sp else np.nan)
        rows.append({
            "model"   : decision,
            "variable": param_name,
            "coef"    : round(param_val, 5),
            "std_err" : round(sp_se, 5)
        })

    coef_df   = pd.DataFrame(rows)
    coef_file = RESULTS_PATH / f"coefficients_{ts}.csv"
    coef_df.to_csv(coef_file, index=False)

    diag_rows = [{
        "test"     : "Moran_I_residuals",
        "statistic": round(moran_res.I, 4),
        "pvalue"   : round(moran_res.p_sim, 4)
    }]
    for name, res in results_lm.items():
        if not np.isnan(res["stat"]):
            diag_rows.append({
                "test"     : name,
                "statistic": round(res["stat"], 4),
                "pvalue"   : round(res["pval"], 4)
            })
    diag_df   = pd.DataFrame(diag_rows)
    diag_file = RESULTS_PATH / f"diagnostics_{ts}.csv"
    diag_df.to_csv(diag_file, index=False)

    print(f"\n  [SUCCESS] Saved:")
    print(f"     {coef_file}")
    print(f"     {diag_file}")
    logger.info(f"Results saved to {RESULTS_PATH}")
    return coef_df, diag_df


# ===========================================================================
# Main
# ===========================================================================

def main():
    print(SEP)
    print("  SPATIAL PANEL ECONOMETRICS — ANTIOQUIA 2015-2023")
    print(f"  Dep. var.  : {DEP_VAR}")
    print(f"  Regressors : {COVARS}")
    print(SEP)

    df, w = load_data()
    y, X, df, n_mpios, n_years = prepare_vectors(df, w)
    ols_fe = ols_panel_fe(y, X, w, df, n_mpios, n_years)
    moran_res, results_lm = spatial_diagnostics(ols_fe, y, X, w, df)
    decision = select_model(moran_res, results_lm)
    modelo_espacial = estimate_model(decision, y, X, w, df)
    coef_df, diag_df = save_results(
        decision, ols_fe, modelo_espacial, moran_res, results_lm, df
    )

    print(f"\n{SEP}")
    print(f"  DONE — Selected model: {decision}")
    print(SEP)
    return ols_fe, modelo_espacial, decision


if __name__ == "__main__":
    ols_fe, modelo_espacial, decision = main()
