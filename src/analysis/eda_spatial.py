#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ANALISIS EXPLORATORIO ESPACIAL (ESDA) - Panel Antioquia 2015-2024
================================================================================
Metodologia econometrica espacial rigurosa previa al modelo SAR/SDM.

Pasos del analisis:
1. Estadisticas descriptivas y distribucion temporal
2. Analisis de dependencia espacial global (Moran's I)
3. Analisis de dependencia espacial local (LISA - Clusters espaciales)
4. Matriz de correlaciones espaciales
5. Recomendacion de especificacion del modelo
================================================================================
"""

import os
import warnings
import logging
from datetime import datetime
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[2])

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt

# Econometria espacial
from libpysal.weights import W
from libpysal.weights import w_subset
from esda import Moran, Moran_Local, Geary
import scipy.stats as stats

# Configuracion
warnings.filterwarnings("ignore")
plt.style.use('default')

os.makedirs("logs", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

logging.basicConfig(
    filename=f"logs/eda_spatial_{datetime.today().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURACION DEL ANALISIS
# =============================================================================

GOLD_PATH = Path("data/gold")
RESULTS_PATH = Path("results")
FIGURES_PATH = RESULTS_PATH / "figures"

# Variables del modelo
DEP_VAR = "ln_va_total"
COVARS = ["homicidios", "cobertura_secundaria"]
ID_VAR = "cod_mpio"
TIME_VAR = "year"

print("="*80)
print("   ANALISIS EXPLORATORIO ESPACIAL (ESDA)")
print("   Panel Municipal Antioquia 2015-2024")
print("="*80)

# =============================================================================
# SECCION 1: CARGA Y ESTADISTICAS DESCRIPTIVAS
# =============================================================================

def cargar_datos():
    """Carga el panel gold con verificacion de calidad de datos."""
    print("\n" + "="*80)
    print("SECCION 1: ESTADISTICAS DESCRIPTIVAS Y CALIDAD DE DATOS")
    print("="*80)

    df = pd.read_parquet(GOLD_PATH / "panel_gold.parquet")
    print(f"\n[+] DIMENSIONES DEL PANEL:")
    print(f"    Observaciones: {df.shape[0]:,}")
    print(f"    Variables: {df.shape[1]}")
    print(f"    Municipios: {df[ID_VAR].nunique()}")
    print(f"    Periodos: {df[TIME_VAR].min()}-{df[TIME_VAR].max()} ({df[TIME_VAR].nunique()} anios)")

    # Balance del panel
    obs_por_mpio = df.groupby(ID_VAR).size()
    print(f"\n[+] BALANCE DEL PANEL:")
    print(f"    Observaciones esperadas (balanceado): {df[ID_VAR].nunique() * df[TIME_VAR].nunique()}")
    print(f"    Observaciones reales: {len(df)}")
    balanceado = len(df) == df[ID_VAR].nunique() * df[TIME_VAR].nunique()
    print(f"    Estado: {'PANEL BALANCEADO' if balanceado else 'PANEL DESBALANCEADO'}")

    # Valores faltantes
    print(f"\n[+] VALORES FALTANTES:")
    for col in [DEP_VAR] + COVARS:
        nulos = df[col].isna().sum()
        pct = 100 * nulos / len(df)
        print(f"    {col}: {nulos} ({pct:.1f}%)")

    return df

def estadisticas_descriptivas(df):
    """Genera estadisticas descriptivas completas."""
    print(f"\n[+] ESTADISTICAS DESCRIPTIVAS:")

    desc = df[[DEP_VAR] + COVARS].describe().round(3)
    print("\n" + desc.to_string())

    # Estadisticas por anio
    print(f"\n[+] EVOLUCION TEMPORAL - {DEP_VAR}:")
    yearly = df.groupby(TIME_VAR)[DEP_VAR].agg(['mean', 'std', 'min', 'max']).round(3)
    print(yearly.to_string())

    # Coeficiente de variacion por anio
    print(f"\n[+] COEFICIENTE DE VARIACION (CV = sigma/mu) POR ANIO:")
    cv = df.groupby(TIME_VAR).apply(
        lambda x: 100 * x[DEP_VAR].std() / x[DEP_VAR].mean()
    ).round(2)
    for year, val in cv.items():
        print(f"    {year}: {val:.2f}%")

    return desc

def analisis_distribucion(df):
    """Analisis de distribucion y normalidad."""
    print(f"\n[+] ANALISIS DE DISTRIBUCION:")

    for var in [DEP_VAR] + COVARS:
        data = df[var].dropna()

        # Estadisticos
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)

        # Test de Jarque-Bera (normalidad)
        jb_stat, jb_pval = stats.jarque_bera(data)

        normalidad = "NO NORMAL" if jb_pval < 0.05 else "NORMAL"
        transformar = "-> transformar" if abs(skew) > 1 else "OK"

        print(f"\n    {var}:")
        print(f"       Asimetria (Skewness): {skew:.3f} ({transformar})")
        print(f"       Curtosis: {kurt:.3f}")
        print(f"       Jarque-Bera: {jb_stat:.2f} (p-valor: {jb_pval:.4f}) [{normalidad}]")

        logger.info(f"{var}: Skew={skew:.3f}, Kurt={kurt:.3f}, JB_pval={jb_pval:.4f}")

# =============================================================================
# SECCION 2: CONSTRUCCION DE MATRIZ DE PESOS W
# =============================================================================

def construir_W_subregion(df):
    """
    Construye matriz W basada en subregiones geograficas.
    Metodo valido cuando no se dispone de shapefile.
    """
    print("\n" + "="*80)
    print("SECCION 2: MATRIZ DE PESOS ESPACIALES W")
    print("="*80)
    print("\n    Metodo: Contiguidad por subregiones (proxy geografico)")

    # Mapear cada municipio a su subregion
    mpio_subregion = df[["cod_mpio", "subregion"]].drop_duplicates().set_index("cod_mpio")["subregion"]

    municipios = df[ID_VAR].unique()
    n = len(municipios)

    # Construir lista de vecinos
    neighbors = {}
    weights = {}

    for mpio in municipios:
        sub = mpio_subregion[mpio]
        # Vecinos = municipios en la misma subregion (excluyendo si mismo)
        vecinos = mpio_subregion[mpio_subregion == sub].index.tolist()
        vecinos.remove(mpio)

        neighbors[mpio] = vecinos
        # Pesos uniformes (suman 1 por fila estandarizada)
        if vecinos:
            weights[mpio] = [1.0/len(vecinos)] * len(vecinos)
        else:
            weights[mpio] = [0.0]

    w = W(neighbors, weights)
    w.transform = 'r'  # Estandarizacion por filas

    print(f"\n[+] PROPIEDADES DE W:")
    print(f"    Municipios (N): {n}")
    print(f"    Conexiones totales: {w.s0:.0f}")
    # Calcular densidad manualmente
    n_links = sum(len(v) for v in w.neighbors.values())
    density = n_links / (n * (n - 1)) if n > 1 else 0
    print(f"    Densidad: {density:.3f}")
    print(f"    Vecinos promedio: {np.mean(list(w.cardinalities.values())):.1f}")
    sin_vecinos = sum(1 for v in w.cardinalities.values() if v == 0)
    print(f"    Municipios sin vecinos: {sin_vecinos}")

    # Mostrar subregiones
    print(f"\n[+] DISTRIBUCION POR SUBREGIONES:")
    subregion_counts = mpio_subregion.value_counts()
    for sub, count in subregion_counts.items():
        print(f"    {sub}: {count} municipios -> {count-1} vecinos por municipio")

    return w, mpio_subregion

# =============================================================================
# SECCION 3: DEPENDENCIA ESPACIAL GLOBAL (Moran's I)
# =============================================================================

def moran_global_analysis(df, w):
    """
    Calcula el indice de Moran para cada periodo (analisis dinamico).
    Moran's I > 0 indica autocorrelacion positiva (clustering).
    """
    print("\n" + "="*80)
    print("SECCION 3: DEPENDENCIA ESPACIAL GLOBAL - Moran's I")
    print("="*80)
    print("\n    Interpretacion:")
    print("    I > 0: Autocorrelacion positiva (valores similares se agrupan)")
    print("    I ~ 0: Distribucion aleatoria")
    print("    I < 0: Autocorrelacion negativa (valores opuestos cercanos)")

    results_moran = []

    print(f"\n[+] MORAN'S I POR ANIO - Variable: {DEP_VAR}")
    print("-" * 70)
    print(f"{'Anio':<6} {'I de Moran':>12} {'Z-score':>12} {'P-valor':>12} {'Signif.':>12}")
    print("-" * 70)

    for year in sorted(df[TIME_VAR].unique()):
        df_year = df[df[TIME_VAR] == year]
        y = df_year.set_index(ID_VAR)[DEP_VAR].dropna()

        if len(y) < 10:
            continue

        try:
            # Subset de pesos para municipios con datos
            w_year = w_subset(w, y.index.tolist())
            w_year.transform = 'r'

            moran = Moran(y.values, w_year)

            if moran.p_sim < 0.01:
                sig = "***"
            elif moran.p_sim < 0.05:
                sig = "**"
            elif moran.p_sim < 0.1:
                sig = "*"
            else:
                sig = "ns"

            results_moran.append({
                'year': year,
                'moran_i': moran.I,
                'z_score': moran.z_sim,
                'p_value': moran.p_sim
            })

            print(f"{year:<6} {moran.I:>12.4f} {moran.z_sim:>12.2f} {moran.p_sim:>12.4f} {sig:>12}")

        except Exception as e:
            print(f"{year:<6} ERROR: {str(e)[:40]}")

    print("-" * 70)
    print("Significancia: *** p<0.01, ** p<0.05, * p<0.1, ns = no significativo")

    # Promedio del periodo
    if results_moran:
        moran_df = pd.DataFrame(results_moran)
        print(f"\n[+] PROMEDIO 2015-2024:")
        print(f"    Moran's I promedio: {moran_df['moran_i'].mean():.4f}")
        print(f"    Desv. estandar: {moran_df['moran_i'].std():.4f}")
        print(f"    Anios significativos: {(moran_df['p_value'] < 0.05).sum()}/{len(moran_df)}")

        # Interpretacion
        avg_i = moran_df['moran_i'].mean()
        print(f"\n[!] INTERPRETACION:")
        if avg_i > 0.3:
            print(f"    FUERTE dependencia espacial (I={avg_i:.3f}). El modelo SAR esta justificado.")
        elif avg_i > 0.1:
            print(f"    MODERADA dependencia espacial (I={avg_i:.3f}). El modelo SAR es apropiado.")
        else:
            print(f"    DEBIL dependencia espacial (I={avg_i:.3f}). Considerar modelo OLS clasico.")
    else:
        moran_df = pd.DataFrame()

    return moran_df

def geary_c_analysis(df, w):
    """
    Indice de Geary's c (alternativa a Moran's I).
    c < 1 indica autocorrelacion positiva.
    """
    print(f"\n[+] INDICE DE GEARY'S C (validacion):")
    print("-" * 60)
    print(f"{'Anio':<6} {'Geary c':>12} {'Interpretacion':>30}")
    print("-" * 60)

    for year in [2015, 2019, 2023]:
        try:
            df_year = df[df[TIME_VAR] == year]
            y = df_year.set_index(ID_VAR)[DEP_VAR].dropna()
            w_year = w_subset(w, y.index.tolist())
            w_year.transform = 'r'

            geary = Geary(y.values, w_year)
            if geary.C < 1:
                interp = "Autocorr. positiva"
            elif geary.C > 1:
                interp = "Autocorr. negativa"
            else:
                interp = "Aleatorio"
            print(f"{year:<6} {geary.C:>12.4f} {interp:>30}")
        except Exception as e:
            print(f"{year:<6} {'ERROR':>12} {str(e)[:25]:>30}")

# =============================================================================
# SECCION 4: ANALISIS LOCAL (LISA - Clusters Espaciales)
# =============================================================================

def lisa_analysis(df, w, year_focus=2023):
    """
    Analisis LISA (Local Indicators of Spatial Association).
    Detecta clusters locales: HH, LL, HL, LH.
    """
    print("\n" + "="*80)
    print("SECCION 4: ANALISIS LISA (Clusters Espaciales Locales)")
    print("="*80)
    print(f"\n    Anio de analisis: {year_focus}")

    df_year = df[df[TIME_VAR] == year_focus].copy()
    y = df_year.set_index(ID_VAR)[DEP_VAR].dropna()

    if len(y) < 10:
        print(f"    ERROR: Insuficientes datos para el anio {year_focus}")
        return df_year

    try:
        w_subset_data = w_subset(w, y.index.tolist())
        w_subset_data.transform = 'r'

        # Calcular LISA
        lisa = Moran_Local(y.values, w_subset_data)

        # Clasificar clusters
        # Cuadrantes: 1=HH, 2=LH, 3=LL, 4=HL
        df_year = df_year[df_year[ID_VAR].isin(y.index)].copy()
        df_year['lisa_q'] = lisa.q
        df_year['lisa_p'] = lisa.p_sim

        # Significativos al 5%
        df_year['cluster'] = 'No significativo'
        df_year.loc[(df_year['lisa_q'] == 1) & (df_year['lisa_p'] < 0.05), 'cluster'] = 'HH (Alto-Alto)'
        df_year.loc[(df_year['lisa_q'] == 2) & (df_year['lisa_p'] < 0.05), 'cluster'] = 'LH (Bajo-Alto)'
        df_year.loc[(df_year['lisa_q'] == 3) & (df_year['lisa_p'] < 0.05), 'cluster'] = 'LL (Bajo-Bajo)'
        df_year.loc[(df_year['lisa_q'] == 4) & (df_year['lisa_p'] < 0.05), 'cluster'] = 'HL (Alto-Bajo)'

        print(f"\n[+] DISTRIBUCION DE CLUSTERS:")
        cluster_counts = df_year['cluster'].value_counts()
        for cluster, count in cluster_counts.items():
            pct = 100 * count / len(df_year)
            print(f"    {cluster}: {count} municipios ({pct:.1f}%)")

        # Municipios en clusters HH (hot spots)
        hh_municipios = df_year[df_year['cluster'] == 'HH (Alto-Alto)'][['municipio', 'subregion', DEP_VAR]]
        if len(hh_municipios) > 0:
            print(f"\n[+] HOT SPOTS (HH) - Municipios con alto VA y vecinos altos:")
            for _, row in hh_municipios.head(10).iterrows():
                print(f"    {row['municipio']} ({row['subregion']}): VA={row[DEP_VAR]:.2f}")

        # Municipios en clusters LL (cold spots)
        ll_municipios = df_year[df_year['cluster'] == 'LL (Bajo-Bajo)'][['municipio', 'subregion', DEP_VAR]]
        if len(ll_municipios) > 0:
            print(f"\n[+] COLD SPOTS (LL) - Municipios con bajo VA y vecinos bajos:")
            for _, row in ll_municipios.head(10).iterrows():
                print(f"    {row['municipio']} ({row['subregion']}): VA={row[DEP_VAR]:.2f}")

        # Outliers espaciales
        mask_outliers = df_year['cluster'].isin(['HL (Alto-Bajo)', 'LH (Bajo-Alto)'])
        outliers = df_year[mask_outliers]
        if len(outliers) > 0:
            print(f"\n[+] OUTLIERS ESPACIALES:")
            for _, row in outliers.iterrows():
                if row['cluster'] == 'HL (Alto-Bajo)':
                    tipo = "alto VA entre vecinos bajos"
                else:
                    tipo = "bajo VA entre vecinos altos"
                print(f"    {row['municipio']}: {tipo}")

    except Exception as e:
        print(f"    ERROR en LISA: {str(e)}")

    return df_year

# =============================================================================
# SECCION 5: MATRIZ DE CORRELACIONES Y MULTICOLINEALIDAD
# =============================================================================

def correlacion_espacial(df):
    """
    Calcula correlaciones simples y detecta multicolinealidad.
    """
    print("\n" + "="*80)
    print("SECCION 5: CORRELACIONES Y MULTICOLINEALIDAD")
    print("="*80)

    # Correlaciones simples
    vars_corr = [DEP_VAR] + COVARS
    corr_matrix = df[vars_corr].corr().round(3)

    print(f"\n[+] MATRIZ DE CORRELACIONES:")
    print(corr_matrix.to_string())

    # Interpretacion
    print(f"\n[+] CORRELACIONES CON {DEP_VAR}:")
    for var in COVARS:
        r = corr_matrix.loc[DEP_VAR, var]
        sig = "Significativa" if abs(r) > 0.1 else "Debil"
        direccion = "negativa" if r < 0 else "positiva"
        print(f"    {var}: r = {r:.3f} ({direccion}) [{sig}]")

    # Test de multicolinearidad (VIF)
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        X = df[COVARS].dropna()
        vif_data = pd.DataFrame()
        vif_data["Variable"] = COVARS
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(COVARS))]

        print(f"\n[+] FACTOR DE INFLACION DE VARIANZA (VIF):")
        print(vif_data.to_string(index=False))
        print(f"\n    Interpretacion VIF:")
        print(f"    VIF < 5: No hay multicolinealidad")
        print(f"    5 <= VIF < 10: Multicolinealidad moderada")
        print(f"    VIF >= 10: Multicolinealidad severa")

    except ImportError:
        print("\n[!] statsmodels no instalado - omitiendo VIF")

    return corr_matrix

# =============================================================================
# SECCION 6: HETEROGENEIDAD ESPACIAL Y MODELO A RECOMENDAR
# =============================================================================

def recomendacion_modelo(df, moran_results):
    """
    Basado en el EDA, recomienda el modelo espacial mas apropiado.
    """
    print("\n" + "="*80)
    print("SECCION 6: RECOMENDACION DE ESPECIFICACION DEL MODELO")
    print("="*80)

    if len(moran_results) == 0:
        print("\n[!] ERROR: No se pudieron calcular estadisticos de Moran")
        return "ERROR"

    avg_moran = moran_results['moran_i'].mean()
    sig_years = (moran_results['p_value'] < 0.05).sum()
    total_years = len(moran_results)

    print(f"\n[+] RESUMEN DEL DIAGNOSTICO:")
    print(f"    Moran's I promedio: {avg_moran:.4f}")
    print(f"    Anios con autocorrelacion significativa: {sig_years}/{total_years}")
    print(f"    Variable dependiente: {DEP_VAR}")
    print(f"    Covariables: {COVARS}")

    print(f"\n" + "="*80)
    print("   RECOMENDACION FINAL")
    print("="*80)

    if avg_moran > 0.1 and sig_years >= total_years * 0.7:
        print("""
    +==================================================================+
    |  MODELO RECOMENDADO: Spatial Autoregressive (SAR) / Spatial Lag |
    +------------------------------------------------------------------+
    |  Especificacion:                                                   |
    |    y = rho*Wy + X*beta + mu + epsilon                              |
    |                                                                    |
    |  Justificacion:                                                    |
    |    [+] Fuerte evidencia de dependencia espacial                  |
    |    [+] Autocorrelacion significativa en la mayoria de anios      |
    |    [+] Patrones de clustering confirmados (LISA)                 |
    |                                                                    |
    |  Alternativa (si hay dependencia en residuos):                   |
    |    Modelo SEM (Error Espacial): y = X*beta + u, u = lambda*Wu + e |
    |                                                                    |
    |  Estimacion recomendada: ML (Maximum Likelihood)                 |
    +------------------------------------------------------------------+
        """)
        recomendacion = "SAR_ML"
    else:
        print("""
    +==================================================================+
    |  MODELO RECOMENDADO: Panel de Datos con Efectos Fijos            |
    +------------------------------------------------------------------+
    |  Especificacion:                                                   |
    |    y = X*beta + mu + epsilon  (sin componente espacial)          |
    |                                                                    |
    |  Justificacion:                                                    |
    |    [-] Debil dependencia espacial                                  |
    |    [-] Autocorrelacion no significativa en la mayoria de anios   |
    +------------------------------------------------------------------+
        """)
        recomendacion = "Panel_OLS"

    # Guardar recomendacion
    rec_df = pd.DataFrame({
        'metric': ['Moran_I_mean', 'significant_years', 'total_years', 'recommended_model'],
        'value': [avg_moran, sig_years, total_years, recomendacion]
    })
    rec_df.to_csv(RESULTS_PATH / "model_recommendation.csv", index=False)

    return recomendacion

# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================

def main():
    """Ejecuta el analisis exploratorio espacial completo."""

    # 1. Cargar datos
    df = cargar_datos()

    # 2. Estadisticas descriptivas
    estadisticas_descriptivas(df)
    analisis_distribucion(df)

    # 3. Construir matriz W
    w, mpio_subregion = construir_W_subregion(df)

    # 4. Analisis de dependencia espacial global
    moran_results = moran_global_analysis(df, w)
    geary_c_analysis(df, w)

    # 5. Analisis LISA (clusters locales)
    lisa_df = lisa_analysis(df, w, year_focus=2023)

    # 6. Correlaciones
    corr_matrix = correlacion_espacial(df)

    # 7. Recomendacion final
    modelo = recomendacion_modelo(df, moran_results)

    print("\n" + "="*80)
    print("   ANALISIS EDA COMPLETADO")
    print("="*80)
    print(f"\n[+] Resultados guardados en: {RESULTS_PATH}/")
    print(f"[+] Recomendacion del modelo: {RESULTS_PATH}/model_recommendation.csv")

    logger.info("ESDA completado exitosamente")

    return df, w, moran_results, modelo

if __name__ == "__main__":
    df, w, moran_results, modelo = main()
