# -*- coding: utf-8 -*-
"""
ExoHabitAI - Exploratory Data Analysis & Preprocessing Pipeline
================================================================
Guidelines implemented:
  1.  Data Quality Assessment (missing values, nulls, duplicates, unit issues)
  2.  Summary statistics + missing-value heatmap
  3.  Handling missing data  (median / mode imputation, row removal)
  4.  Outlier Detection       (Z-Score + IQR; cap or remove)
  5.  Unit Standardisation    (Earth radii, Earth masses, AU, Kelvin)
  6.  Feature Engineering     (Habitability Score, Stellar Compatibility,
                               Orbital Stability)
  7.  Categorical Encoding    (One-Hot Encoding for star spectral type)
  8.  Feature Scaling         (StandardScaler on numeric features)
  9.  Target Variable Creation (binary + multi-class)
 10.  Export -> data/preprocessed/preprocessed.csv
"""

import os
import sys
import warnings

# Force UTF-8 stdout so special chars don't crash on Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "PS_2026.02.09_06.11.21.csv")
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "data", "preprocessed")
PLOTS_DIR     = os.path.join(PROJECT_ROOT, "plots")
OUTPUT_CSV    = os.path.join(OUTPUT_DIR, "preprocessed.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ---------------------------------------------------------------------------
# Feature catalogue
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "pl_rade",      # Planet radius       -> Earth radii
    "pl_bmasse",    # Planet mass         -> Earth masses
    "pl_orbper",    # Orbital period      -> days
    "pl_orbsmax",   # Semi-major axis     -> AU
    "pl_eqt",       # Equilibrium temp    -> K
    "pl_dens",      # Planet density      -> g/cm3
    "st_teff",      # Host star temp      -> K
    "st_lum",       # Star luminosity log -> converted to linear L_sun
    "st_met",       # Star metallicity    -> dex
    "st_spectype",  # Star spectral type  -> categorical
]

NUMERIC_COLS   = [c for c in FEATURE_COLS if c != "st_spectype"]
IDENTIFIER_COLS = ["pl_name", "hostname"]

PHYSICAL_LIMITS = {
    "pl_rade":    (0.1,    30.0),
    "pl_bmasse":  (0.01,   10_000.0),
    "pl_orbper":  (0.05,   200_000.0),
    "pl_orbsmax": (0.001,  5_000.0),
    "pl_eqt":     (30.0,   6_000.0),
    "pl_dens":    (0.0001, 100.0),
    "st_teff":    (1_000.0, 50_000.0),
    "st_lum":     (1e-4,   1e5),
    "st_met":     (-5.0,   2.0),
}

FEATURE_LABELS = {
    "pl_rade":    "Planet Radius (R_Earth)",
    "pl_bmasse":  "Planet Mass (M_Earth)",
    "pl_orbper":  "Orbital Period (days)",
    "pl_orbsmax": "Semi-Major Axis (AU)",
    "pl_eqt":     "Equilibrium Temp (K)",
    "pl_dens":    "Planet Density (g/cm3)",
    "st_teff":    "Star Temp (K)",
    "st_lum":     "Star Luminosity (L_sun)",
    "st_met":     "Star Metallicity (dex)",
}

SCALE_COLS = [
    "pl_rade", "pl_bmasse", "pl_orbper", "pl_orbsmax",
    "pl_eqt", "pl_dens", "st_teff", "st_lum", "st_met",
    "habitability_score", "stellar_compatibility", "orbital_stability",
]

SEP = "=" * 60
DASH = "-" * 60

# ===========================================================================
# STEP 0 - Load raw dataset
# ===========================================================================
def load_raw(path):
    df = pd.read_csv(path, comment="#", low_memory=False)
    keep = [c for c in IDENTIFIER_COLS + FEATURE_COLS if c in df.columns]
    return df[keep].copy()


# ===========================================================================
# STEP 1 - Data Quality Assessment
# ===========================================================================
def data_quality_assessment(df):
    print("\n" + DASH)
    print("STEP 1 . Data Quality Assessment")
    print(DASH)

    n_rows, n_cols = df.shape
    print("  Raw shape          : {:,} rows x {} columns".format(n_rows, n_cols))

    # Missing values
    missing     = df[FEATURE_COLS].isna().sum()
    missing_pct = (missing / n_rows * 100).round(2)
    mv_report   = pd.DataFrame({"missing_count": missing, "missing_%": missing_pct})
    print("\n  Missing-value summary (10 features):")
    print(mv_report.to_string())

    # Duplicates
    n_dup = int(df.duplicated().sum())
    print("\n  Duplicate rows     : {:,}".format(n_dup))

    # Impossible values
    print("\n  Physically impossible values (before cleaning):")
    found_any = False
    for col, (lo, hi) in PHYSICAL_LIMITS.items():
        numeric = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)
        neg_cnt  = int((numeric < lo).sum())
        over_cnt = int((numeric > hi).sum())
        if neg_cnt or over_cnt:
            print("    {:14s}  below {}: {:,}   above {}: {:,}".format(col, lo, neg_cnt, hi, over_cnt))
            found_any = True
    if not found_any:
        print("    None detected.")

    print("\n  Unit notes:")
    print("    st_lum    -> NASA stores log10(L_sun); will convert to linear L_sun")
    print("    pl_rade   -> already in Earth radii   (no conversion needed)")
    print("    pl_bmasse -> already in Earth masses  (no conversion needed)")
    print("    pl_orbsmax-> already in AU            (no conversion needed)")
    print("    pl_eqt    -> already in Kelvin        (no conversion needed)")

    return {"missing": missing, "missing_pct": missing_pct, "n_dup": n_dup}


# ===========================================================================
# STEP 2 - Summary statistics + missing-value heatmap
# ===========================================================================
def generate_summary_and_heatmap(df):
    print("\n" + DASH)
    print("STEP 2 . Summary Statistics & Missing-Value Heatmap")
    print(DASH)

    # Summary stats CSV
    summ      = df[NUMERIC_COLS].describe(percentiles=[.25, .5, .75]).transpose()
    summ_path = os.path.join(PLOTS_DIR, "summary_statistics.csv")
    summ.to_csv(summ_path)
    print("  Saved summary statistics       ->", summ_path)

    # Missing value heatmap (subsample for speed)
    sample = df[FEATURE_COLS].head(3000)
    plt.figure(figsize=(14, 6))
    sns.heatmap(sample.isna(), cbar=False, cmap="viridis", yticklabels=False)
    plt.title("Missing Value Heatmap (first 3 000 rows)", fontsize=14)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    hm_path = os.path.join(PLOTS_DIR, "missing_value_heatmap.png")
    plt.savefig(hm_path, dpi=150)
    plt.close()
    print("  Saved missing-value heatmap    ->", hm_path)


# ===========================================================================
# STEP 3 - Unit Standardisation
# ===========================================================================
def standardise_units(df):
    print("\n" + DASH)
    print("STEP 3 . Unit Standardisation")
    print(DASH)
    out = df.copy()

    for col in NUMERIC_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # st_lum: log10(L_sun) -> linear L_sun
    out["st_lum"] = np.where(out["st_lum"].notna(),
                              np.power(10.0, out["st_lum"]),
                              np.nan)

    print("  st_lum    : log10(L_sun) -> linear L_sun  [converted]")
    print("  pl_rade   : Earth radii  (no change)")
    print("  pl_bmasse : Earth masses (no change)")
    print("  pl_orbsmax: AU           (no change)")
    print("  pl_eqt    : Kelvin       (no change)")
    return out


# ===========================================================================
# STEP 4 - Remove duplicates & completely-missing rows
# ===========================================================================
def remove_bad_rows(df):
    print("\n" + DASH)
    print("STEP 4 . Removing Duplicates & Completely-Missing Rows")
    print(DASH)

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dup_removed = before - len(df)
    print("  Duplicate rows removed           : {:,}".format(n_dup_removed))

    all_missing_mask = df[NUMERIC_COLS + ["st_spectype"]].isna().all(axis=1)
    n_empty = int(all_missing_mask.sum())
    df = df.loc[~all_missing_mask].reset_index(drop=True)
    print("  All-feature-missing rows removed : {:,}".format(n_empty))
    print("  Shape after cleanup              : {}".format(df.shape))
    return df


# ===========================================================================
# STEP 5 - Apply hard physical limits (impossible values -> NaN)
# ===========================================================================
def apply_physical_limits(df):
    print("\n" + DASH)
    print("STEP 5 . Physical Limits (impossible values -> NaN)")
    print(DASH)
    out = df.copy()
    for col, (lo, hi) in PHYSICAL_LIMITS.items():
        if col not in out.columns:
            continue
        series = out[col]
        n_bad  = int(((series < lo) | (series > hi)).sum())
        out[col] = series.where(series.between(lo, hi), np.nan)
        if n_bad:
            print("  {:14s} -> set {:,} out-of-range values to NaN".format(col, n_bad))
    return out


# ===========================================================================
# STEP 6 - Outlier Detection: IQR + Z-Score, then cap
# ===========================================================================
def detect_and_cap_outliers(df):
    print("\n" + DASH)
    print("STEP 6 . Outlier Detection & Capping  (IQR + Z-Score, threshold=4 sigma)")
    print(DASH)
    out    = df.copy()
    bounds = {}

    for col in NUMERIC_COLS:
        series = out[col].dropna()
        if len(series) < 4:
            continue

        # IQR
        q1, q3 = series.quantile([0.25, 0.75])
        iqr     = q3 - q1
        iqr_lo  = (q1 - 1.5 * iqr) if iqr > 0 else q1
        iqr_hi  = (q3 + 1.5 * iqr) if iqr > 0 else q3

        # Z-Score (4 sigma)
        z_mean, z_std = series.mean(), series.std()
        z_lo = (z_mean - 4 * z_std) if z_std > 0 else series.min()
        z_hi = (z_mean + 4 * z_std) if z_std > 0 else series.max()

        # Physical hard limits
        hard_lo, hard_hi = PHYSICAL_LIMITS.get(col, (-np.inf, np.inf))

        # Combined: tightest range
        combined_lo = float(max(iqr_lo, z_lo, hard_lo))
        combined_hi = float(min(iqr_hi, z_hi, hard_hi))
        if combined_lo >= combined_hi:
            combined_lo, combined_hi = hard_lo, hard_hi

        bounds[col] = (combined_lo, combined_hi)
        n_outliers  = int(((out[col] < combined_lo) | (out[col] > combined_hi)).sum())
        out[col]    = out[col].clip(combined_lo, combined_hi)

        print("  {:14s}  IQR=[{:.3g},{:.3g}]  Z4=[{:.3g},{:.3g}]  capped={:,}".format(
            col, iqr_lo, iqr_hi, z_lo, z_hi, n_outliers))

    # Box plots after capping
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    for ax, col in zip(axes.flat, NUMERIC_COLS):
        ax.boxplot(out[col].dropna(), vert=True, patch_artist=True,
                   boxprops=dict(facecolor="#4C9BE8", alpha=0.7))
        ax.set_title(FEATURE_LABELS.get(col, col), fontsize=9)
        ax.tick_params(axis="both", labelsize=7)
    plt.suptitle("Feature Distributions After Outlier Capping", fontsize=13, y=1.01)
    plt.tight_layout()
    bp_path = os.path.join(PLOTS_DIR, "outlier_boxplots.png")
    plt.savefig(bp_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved box-plot              ->", bp_path)
    return out, bounds


# ===========================================================================
# STEP 7 - Impute Missing Values
# ===========================================================================
def _normalize_spectral_type(series):
    norm = series.fillna("").astype(str).str.strip().str.upper()
    norm = norm.replace({"": np.nan, "NAN": np.nan, "NONE": np.nan, "UNKNOWN": np.nan})
    norm = norm.str[0]
    norm = norm.where(norm.str.match(r"[A-Z]", na=False), np.nan)
    return norm


def impute_missing(df):
    print("\n" + DASH)
    print("STEP 7 . Imputing Missing Values")
    print(DASH)
    out              = df.copy()
    imputation_vals  = {}

    # Planetary physical values -> median
    planetary_cols = ["pl_rade", "pl_bmasse", "pl_orbper", "pl_orbsmax", "pl_eqt", "pl_dens"]
    for col in planetary_cols:
        med     = out[col].median()
        n_fill  = int(out[col].isna().sum())
        out[col] = out[col].fillna(med)
        imputation_vals[col] = float(med)
        print("  {:14s} -> median imputation  (filled {:,}, median={:.4g})".format(col, n_fill, med))

    # Star temperature -> median
    med_teff  = out["st_teff"].median()
    n_teff    = int(out["st_teff"].isna().sum())
    out["st_teff"] = out["st_teff"].fillna(med_teff)
    imputation_vals["st_teff"] = float(med_teff)
    print("  {:14s} -> median imputation  (filled {:,}, median={:.4g})".format("st_teff", n_teff, med_teff))

    # st_lum, st_met -> median
    for col in ["st_lum", "st_met"]:
        med    = out[col].median()
        n_fill = int(out[col].isna().sum())
        out[col] = out[col].fillna(med)
        imputation_vals[col] = float(med)
        print("  {:14s} -> median imputation  (filled {:,}, median={:.4g})".format(col, n_fill, med))

    # Star spectral type -> mode (categorical)
    spec_series = _normalize_spectral_type(out["st_spectype"])
    spec_mode   = spec_series.dropna().mode().iloc[0] if spec_series.dropna().any() else "G"
    n_spec      = int(spec_series.isna().sum())
    out["st_spectype"] = spec_series.fillna(spec_mode)
    imputation_vals["st_spectype"] = spec_mode
    print("  {:14s} -> mode  imputation  (filled {:,}, mode='{}')".format("st_spectype", n_spec, spec_mode))

    return out, imputation_vals


# ===========================================================================
# STEP 8 - Feature Engineering
# ===========================================================================
def _closeness(series, target, tol):
    """Gaussian-like proximity score in [0, 1]."""
    return np.exp(-np.abs(series - target) / tol)


def engineer_features(df):
    print("\n" + DASH)
    print("STEP 8 . Feature Engineering")
    print(DASH)
    out = df.copy()

    # Stellar radius proxy (R_sun) : R ~ sqrt(L) / (T/T_sun)^2
    safe_lum    = np.clip(out["st_lum"],  1e-6, None)
    safe_teff   = np.clip(out["st_teff"], 1.0,  None)
    stellar_rad = np.sqrt(safe_lum) / np.power(safe_teff / 5778.0, 2)

    # 8a. Habitability Score Index
    out["habitability_score"] = (
        _closeness(out["pl_eqt"],     288.0,  85.0)   # temp proximity to habitable range
        + _closeness(out["pl_rade"],  1.0,    0.8)    # radius similarity to Earth
        + _closeness(out["pl_orbsmax"],1.0,   0.9)    # distance from star (AU)
        + _closeness(out["st_lum"],   1.0,    1.25)   # stellar luminosity
    ) / 4.0
    print("  habitability_score     : temp + radius + distance + luminosity")

    # 8b. Stellar Compatibility Index
    out["stellar_compatibility"] = (
        _closeness(out["st_teff"],    5778.0, 1800.0)   # host star temperature
        + _closeness(stellar_rad,     1.0,    0.9)      # star size
        + _closeness(out["st_lum"],   1.0,    1.25)     # radiation stability
    ) / 3.0
    print("  stellar_compatibility  : star temp + size + luminosity")

    # 8c. Orbital Stability Factor
    out["orbital_stability"] = (
        _closeness(out["pl_orbper"],  365.0, 220.0)    # orbital period
        + _closeness(out["pl_orbsmax"], 1.0, 0.8)     # semi-major axis
    ) / 2.0
    print("  orbital_stability      : orbital period + semi-major axis")

    return out


# ===========================================================================
# STEP 9 - Target Variable Creation (binary + multi-class)
# ===========================================================================
def create_target(df):
    print("\n" + DASH)
    print("STEP 9 . Target Variable Creation")
    print(DASH)
    out = df.copy()

    # Binary: strict habitable zone criteria
    habitable_mask = (
        (out["pl_rade"]    <= 2.0)
        & out["st_teff"].between(3900, 7200)
        & out["pl_eqt"].between(180, 330)
        & out["pl_orbsmax"].between(0.4, 2.2)
        & out["st_lum"].between(0.08, 2.5)
    )
    out["habitability"] = habitable_mask.astype(int)

    # Multi-class: 0=non, 1=potentially, 2=highly habitable
    broad_mask = (
        out["pl_rade"].between(0.5, 3.0)
        & out["pl_eqt"].between(150, 400)
        & out["pl_orbsmax"].between(0.2, 3.0)
        & out["st_teff"].between(2600, 7600)
        & out["st_lum"].between(0.01, 5.0)
    )
    out["habitability_class"] = 0
    out.loc[broad_mask,    "habitability_class"] = 1
    out.loc[habitable_mask,"habitability_class"] = 2

    n_hab   = int(habitable_mask.sum())
    n_broad = int(broad_mask.sum()) - n_hab
    n_none  = len(out) - n_hab - n_broad
    print("  Binary   -> habitable: {:,}  |  non-habitable: {:,}".format(n_hab, len(out) - n_hab))
    print("  Multi-class -> class 0 (non): {:,}  |  class 1 (broad): {:,}  |  class 2 (strict): {:,}".format(
        n_none, n_broad, n_hab))
    return out


# ===========================================================================
# STEP 10 - Categorical Encoding (One-Hot for st_spectype)
# ===========================================================================
def encode_categorical(df):
    print("\n" + DASH)
    print("STEP 10 . Categorical Encoding  (One-Hot for star spectral type)")
    print(DASH)
    out    = df.copy()
    dummies = pd.get_dummies(out["st_spectype"], prefix="star_type", dtype=int)
    print("  One-Hot columns created:", list(dummies.columns))
    out = pd.concat([out, dummies], axis=1)
    return out


# ===========================================================================
# STEP 11 - Feature Scaling (StandardScaler)
# ===========================================================================
def scale_features(df):
    print("\n" + DASH)
    print("STEP 11 . Feature Scaling  (StandardScaler)")
    print(DASH)
    out         = df.copy()
    scaler      = StandardScaler()
    scaled_cols = [c for c in SCALE_COLS if c in out.columns]
    out[scaled_cols] = scaler.fit_transform(out[scaled_cols])
    print("  Scaled {:,} numeric columns via StandardScaler".format(len(scaled_cols)))
    return out, scaler


# ===========================================================================
# STEP 12 - Correlation heatmap
# ===========================================================================
def generate_correlation_heatmap(df):
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in ("habitability", "habitability_class")]
    corr = df[num_cols].corr()
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False,
                linewidths=0.3, square=False)
    plt.title("Feature Correlation Heatmap (Post-Processing)", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "feature_correlation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("  Saved correlation heatmap      ->", path)


# ===========================================================================
# STEP 13 - Data quality report
# ===========================================================================
def save_quality_report(raw_df, final_df, bounds):
    rows = []
    for col in FEATURE_COLS:
        raw_missing  = int(raw_df[col].isna().sum()) if col in raw_df.columns else len(raw_df)
        post_missing = int(final_df[col].isna().sum()) if col in final_df.columns else 0
        row = {"feature": col, "raw_missing": raw_missing, "post_missing": post_missing}
        if col in bounds:
            row["cap_lower"] = bounds[col][0]
            row["cap_upper"] = bounds[col][1]
        rows.append(row)
    report = pd.DataFrame(rows)
    path   = os.path.join(PLOTS_DIR, "data_quality_report.csv")
    report.to_csv(path, index=False)
    print("  Saved data quality report      ->", path)


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print(SEP)
    print("  ExoHabitAI - EDA & Preprocessing Pipeline")
    print(SEP)

    # 0. Load
    print("\nLoading raw dataset ...")
    raw_df = load_raw(RAW_DATA_PATH)
    print("  Raw shape:", raw_df.shape)

    # 1. Quality assessment
    data_quality_assessment(raw_df)

    # 2. Summary stats + heatmap (on raw)
    generate_summary_and_heatmap(raw_df)

    # 3. Unit standardisation
    df = standardise_units(raw_df)

    # 4. Remove duplicates / empty rows
    df = remove_bad_rows(df)

    # 5. Physical limits
    df = apply_physical_limits(df)

    # 6. Outlier capping
    df, bounds = detect_and_cap_outliers(df)

    # 7. Imputation
    df, imputation_vals = impute_missing(df)

    # 8. Feature engineering
    df = engineer_features(df)

    # 9. Target variable
    df = create_target(df)

    # 10. Categorical encoding
    df = encode_categorical(df)

    # 11. Feature scaling
    df_scaled, _ = scale_features(df)

    # 12. Correlation heatmap
    print("\n" + DASH)
    print("STEP 12 . Correlation Heatmap")
    print(DASH)
    generate_correlation_heatmap(df_scaled)

    # 13. Quality report
    print("\n" + DASH)
    print("STEP 13 . Data Quality Report")
    print(DASH)
    save_quality_report(raw_df, df, bounds)

    # 14. Save preprocessed.csv
    print("\n" + DASH)
    print("STEP 14 . Saving preprocessed.csv")
    print(DASH)
    df_scaled.to_csv(OUTPUT_CSV, index=False)
    print("  Saved ->", OUTPUT_CSV)
    print("  Final shape :", df_scaled.shape)
    print("  Columns     :", list(df_scaled.columns))

    print("\n" + SEP)
    print("  Pipeline completed successfully!")
    print(SEP)


if __name__ == "__main__":
    main()
