import os
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "PS_2026.02.09_06.11.21.csv")
PREPROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "preprocessed", "preprocessed.csv")
EDA_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "preprocessed", "preprocessed_eda.csv")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "exohabit_ml.csv")
RANKED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "habitability_ranked.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

IDENTIFIER_COLUMNS = ["pl_name", "hostname"]
RAW_FEATURE_COLUMNS = [
    "pl_rade",
    "pl_bmasse",
    "pl_orbper",
    "pl_orbsmax",
    "pl_eqt",
    "pl_dens",
    "st_teff",
    "st_lum",
    "st_met",
    "st_spectype",
]
NUMERIC_COLUMNS = [
    "pl_rade",
    "pl_bmasse",
    "pl_orbper",
    "pl_orbsmax",
    "pl_eqt",
    "pl_dens",
    "st_teff",
    "st_lum",
    "st_met",
]
POSITIVE_ONLY_COLUMNS = [
    "pl_rade",
    "pl_bmasse",
    "pl_orbper",
    "pl_orbsmax",
    "pl_eqt",
    "pl_dens",
    "st_teff",
    "st_lum",
]
ENGINEERED_COLUMNS = [
    "stellar_compatibility",
    "orbital_stability",
]
TARGET_COLUMN = "habitability"
NON_FEATURE_COLUMNS = IDENTIFIER_COLUMNS + [TARGET_COLUMN]

DEFAULT_NUMERIC_VALUES = {
    "pl_rade": 1.0,
    "pl_bmasse": 1.0,
    "pl_orbper": 365.25,
    "pl_orbsmax": 1.0,
    "pl_eqt": 288.0,
    "pl_dens": 5.5,
    "st_teff": 5778.0,
    "st_lum": 1.0,
    "st_met": 0.0,
}

PHYSICAL_LIMITS = {
    "pl_rade": (0.1, 30.0),
    "pl_bmasse": (0.01, 10000.0),
    "pl_orbper": (0.05, 200000.0),
    "pl_orbsmax": (0.001, 5000.0),
    "pl_eqt": (30.0, 6000.0),
    "pl_dens": (0.0001, 100.0),
    "st_teff": (1000.0, 50000.0),
    "st_lum": (1e-4, 1e5),
    "st_met": (-5.0, 2.0),
}

FIELD_ALIASES = {
    "name": "planet_name",
    "pl_name": "planet_name",
    "planet_name": "planet_name",
    "hostname": "host_name",
    "host_name": "host_name",
    "planet_radius": "planet_radius",
    "pl_rade": "planet_radius",
    "planet_mass": "planet_mass",
    "pl_bmasse": "planet_mass",
    "orbital_period": "orbital_period",
    "pl_orbper": "orbital_period",
    "semi_major_axis": "semi_major_axis",
    "pl_orbsmax": "semi_major_axis",
    "equilibrium_temperature": "equilibrium_temperature",
    "equilibrium_temp": "equilibrium_temperature",
    "surface_temperature": "equilibrium_temperature",
    "pl_eqt": "equilibrium_temperature",
    "planet_density": "planet_density",
    "pl_dens": "planet_density",
    "star_temperature": "star_temperature",
    "host_star_temperature": "star_temperature",
    "st_teff": "star_temperature",
    "star_luminosity": "star_luminosity",
    "stellar_luminosity": "star_luminosity",
    "star_luminosity_log": "star_luminosity_log",
    "stellar_luminosity_log": "star_luminosity_log",
    "st_lum": "star_luminosity_log",
    "star_metallicity": "star_metallicity",
    "stellar_metallicity": "star_metallicity",
    "st_met": "star_metallicity",
    "spectral_type": "spectral_type",
    "star_type": "spectral_type",
    "st_spectype": "spectral_type",
}

REQUIRED_API_FIELDS = [
    "orbital_period",
    "semi_major_axis",
    "star_temperature",
    "star_metallicity",
    "spectral_type",
]
NUMERIC_API_FIELDS = [
    "planet_radius",
    "planet_mass",
    "orbital_period",
    "semi_major_axis",
    "equilibrium_temperature",
    "planet_density",
    "star_temperature",
    "star_luminosity",
    "star_luminosity_log",
    "star_metallicity",
]


def ensure_directories() -> None:
    os.makedirs(os.path.dirname(PREPROCESSED_DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def load_raw_dataset(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", low_memory=False)


def _safe_column(frame: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def _build_base_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=raw_df.index)
    frame["pl_name"] = _safe_column(raw_df, "pl_name", pd.NA)
    frame["hostname"] = _safe_column(raw_df, "hostname", pd.NA)
    for column in RAW_FEATURE_COLUMNS:
        frame[column] = _safe_column(raw_df, column, pd.NA)
    return frame


def _standardize_units(df: pd.DataFrame) -> pd.DataFrame:
    standardized = df.copy()
    for column in NUMERIC_COLUMNS:
        standardized[column] = pd.to_numeric(standardized[column], errors="coerce")

    luminosity_log = standardized["st_lum"]
    standardized["st_lum"] = np.where(luminosity_log.notna(), np.power(10.0, luminosity_log), np.nan)
    return standardized


def _normalize_spectral_type(series: pd.Series, fill_value: str | None = None) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip().str.upper()
    normalized = normalized.replace({"": np.nan, "NAN": np.nan, "NONE": np.nan, "UNKNOWN": np.nan})
    normalized = normalized.str[0]
    normalized = normalized.where(normalized.str.match(r"[A-Z]", na=False), np.nan)
    if fill_value is not None:
        normalized = normalized.fillna(fill_value)
    return normalized


def _fill_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    completed = df.copy()
    completed["hostname"] = completed["hostname"].fillna("Unknown Host").astype(str).str.strip()

    name_series = completed["pl_name"].fillna("").astype(str).str.strip()
    missing_name_mask = name_series.eq("")
    generated_names = pd.Series(
        [f"Planet_{index + 1}" for index in range(len(completed))],
        index=completed.index,
    )
    completed["pl_name"] = name_series.where(~missing_name_mask, generated_names)
    return completed


def _apply_physical_limits(df: pd.DataFrame) -> pd.DataFrame:
    limited = df.copy()
    for column, (lower, upper) in PHYSICAL_LIMITS.items():
        series = limited[column]
        limited[column] = series.where(series.between(lower, upper), np.nan)
    return limited


def _compute_iqr_bounds(df: pd.DataFrame, columns: List[str]) -> Dict[str, Tuple[float, float]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            bounds[column] = (float(q1), float(q3))
            continue
        bounds[column] = (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))
    return bounds


def _compute_zscore_bounds(df: pd.DataFrame, columns: List[str], threshold: float = 4.0) -> Dict[str, Tuple[float, float]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    for column in columns:
        mean = df[column].mean()
        std = df[column].std()
        if pd.isna(std) or std == 0:
            bounds[column] = (float(df[column].min()), float(df[column].max()))
            continue
        bounds[column] = (float(mean - threshold * std), float(mean + threshold * std))
    return bounds


def _combine_bounds(
    iqr_bounds: Dict[str, Tuple[float, float]],
    zscore_bounds: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[float, float]]:
    combined: Dict[str, Tuple[float, float]] = {}
    for column in NUMERIC_COLUMNS:
        iqr_lower, iqr_upper = iqr_bounds.get(column, (-np.inf, np.inf))
        z_lower, z_upper = zscore_bounds.get(column, (-np.inf, np.inf))
        hard_lower, hard_upper = PHYSICAL_LIMITS[column]

        lower_candidates = [value for value in [iqr_lower, z_lower, hard_lower] if pd.notna(value)]
        upper_candidates = [value for value in [iqr_upper, z_upper, hard_upper] if pd.notna(value)]
        lower = max(lower_candidates) if lower_candidates else -np.inf
        upper = min(upper_candidates) if upper_candidates else np.inf

        if lower > upper:
            lower, upper = hard_lower, hard_upper

        combined[column] = (float(lower), float(upper))
    return combined


def apply_outlier_bounds(df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    clipped = df.copy()
    for column, (lower, upper) in bounds.items():
        if column in clipped.columns:
            clipped[column] = clipped[column].clip(lower, upper)
    return clipped


def _stellar_radius_proxy(df: pd.DataFrame) -> pd.Series:
    safe_luminosity = np.clip(df["st_lum"], 1e-6, None)
    safe_temperature = np.clip(df["st_teff"], 1.0, None)
    return np.sqrt(safe_luminosity) / np.power(safe_temperature / 5778.0, 2)


def engineer_features(base_df: pd.DataFrame) -> pd.DataFrame:
    df = base_df.copy()
    stellar_radius = _stellar_radius_proxy(df)

    # NOTE: habitability_score removed — it was computed from the same features
    # used to define the target label, causing direct data leakage (AUC=1.0).
    df["stellar_compatibility"] = (
        _closeness_score(df["st_teff"], 5778.0, 1800.0)
        + _closeness_score(stellar_radius, 1.0, 0.9)
        + _closeness_score(df["st_lum"], 1.0, 1.25)
    ) / 3.0
    df["orbital_stability"] = (
        _closeness_score(df["pl_orbper"], 365.0, 220.0)
        + _closeness_score(df["pl_orbsmax"], 1.0, 0.8)
    ) / 2.0
    return df


def _closeness_score(series: pd.Series, target: float, tolerance: float) -> pd.Series:
    return np.exp(-np.abs(series - target) / tolerance)


def _habitable_rule_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df["pl_rade"] <= 2.0)
        & (df["st_teff"].between(3900, 7200))
        & (df["pl_eqt"].between(180, 330))
        & (df["pl_orbsmax"].between(0.4, 2.2))
        & (df["st_lum"].between(0.08, 2.5))
    )


def _broad_plausibility_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["pl_rade"].between(0.5, 3.0)
        & df["pl_eqt"].between(150, 400)
        & df["pl_orbsmax"].between(0.2, 3.0)
        & df["st_teff"].between(2600, 7600)
        & df["st_lum"].between(0.01, 5.0)
    )


def _create_target(df: pd.DataFrame) -> pd.Series:
    return _habitable_rule_mask(df).astype(int)


def compute_physics_prior(feature_df: pd.DataFrame) -> pd.Series:
    prior = (
        _closeness_score(feature_df["pl_rade"], 1.0, 0.75)
        + _closeness_score(feature_df["pl_eqt"], 288.0, 80.0)
        + _closeness_score(feature_df["pl_orbsmax"], 1.0, 0.8)
        + _closeness_score(feature_df["st_teff"], 5778.0, 1800.0)
        + _closeness_score(feature_df["st_lum"], 1.0, 1.25)
    ) / 5.0
    prior = prior.pow(2)
    rule_mask = _habitable_rule_mask(feature_df)
    prior = pd.Series(np.where(rule_mask, np.maximum(prior, 0.8), prior), index=feature_df.index)
    return prior.clip(0.0, 1.0)


def blend_probabilities(model_probabilities, feature_df: pd.DataFrame):
    # NOTE: Physics-prior blending removed — the prior called _habitable_rule_mask
    # which re-encoded the exact label definition into predictions, causing AUC=1.0.
    # We now return raw model probabilities so the ROC curve reflects genuine learning.
    return np.clip(np.asarray(model_probabilities, dtype=float), 0.0, 1.0)


def preprocess_raw_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    df = _build_base_frame(raw_df)
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    df = _fill_identifier_columns(df)
    df = _standardize_units(df)

    complete_missing_rows = df[NUMERIC_COLUMNS + ["st_spectype"]].isna().all(axis=1)
    df = df.loc[~complete_missing_rows].copy()
    df = _apply_physical_limits(df)

    spectral_type_series = _normalize_spectral_type(df["st_spectype"])
    spectral_mode = spectral_type_series.dropna().mode().iloc[0] if spectral_type_series.dropna().any() else "G"
    df["st_spectype"] = spectral_type_series.fillna(spectral_mode)

    iqr_bounds = _compute_iqr_bounds(df, NUMERIC_COLUMNS)
    zscore_bounds = _compute_zscore_bounds(df, NUMERIC_COLUMNS)
    outlier_bounds = _combine_bounds(iqr_bounds, zscore_bounds)
    df = apply_outlier_bounds(df, outlier_bounds)

    numeric_medians = df[NUMERIC_COLUMNS].median()
    for column, default_value in DEFAULT_NUMERIC_VALUES.items():
        if pd.isna(numeric_medians.get(column)):
            numeric_medians[column] = default_value

    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].fillna(numeric_medians)
    df["st_spectype"] = df["st_spectype"].fillna(spectral_mode)
    df = engineer_features(df)
    df[TARGET_COLUMN] = _create_target(df)

    preprocessed_df = df.reset_index(drop=True)
    model_df = preprocessed_df.copy()
    preprocessing_profile = {
        "numeric_medians": {column: float(numeric_medians[column]) for column in NUMERIC_COLUMNS},
        "spectral_type_mode": spectral_mode,
        "iqr_bounds": iqr_bounds,
        "zscore_bounds": zscore_bounds,
        "outlier_bounds": outlier_bounds,
        "duplicate_rows_removed": duplicate_count,
        "completely_missing_rows_removed": int(complete_missing_rows.sum()),
    }
    return preprocessed_df, model_df, preprocessing_profile


def get_feature_columns(model_df: pd.DataFrame) -> List[str]:
    return [column for column in model_df.columns if column not in NON_FEATURE_COLUMNS]


def artifact_paths(base_dir: str = PROJECT_ROOT) -> Dict[str, str]:
    models_dir = os.path.join(base_dir, "models")
    return {
        "final_model": os.path.join(models_dir, "final_model.pkl"),
        "logistic_regression": os.path.join(models_dir, "logistic_regression.pkl"),
        "decision_tree": os.path.join(models_dir, "decision_tree.pkl"),
        "random_forest": os.path.join(models_dir, "random_forest.pkl"),
        "random_forest_tuned": os.path.join(models_dir, "random_forest_tuned.pkl"),
        "xgboost": os.path.join(models_dir, "xgboost.pkl"),
        "xgboost_tuned": os.path.join(models_dir, "xgboost_tuned.pkl"),
        "feature_cols": os.path.join(models_dir, "feature_cols.pkl"),
        "preprocessing_profile": os.path.join(models_dir, "preprocessing_profile.pkl"),
        "training_summary": os.path.join(models_dir, "training_summary.pkl"),
    }


def save_feature_metadata(feature_columns: List[str], preprocessing_profile: Dict[str, Any]) -> None:
    paths = artifact_paths()
    joblib.dump(feature_columns, paths["feature_cols"])
    joblib.dump(preprocessing_profile, paths["preprocessing_profile"])


def load_feature_metadata(base_dir: str = PROJECT_ROOT) -> Tuple[List[str], Dict[str, Any]]:
    paths = artifact_paths(base_dir)
    feature_columns = joblib.load(paths["feature_cols"])
    preprocessing_profile = joblib.load(paths["preprocessing_profile"])
    return feature_columns, preprocessing_profile


def normalize_payload(payload: Dict) -> Tuple[List[str], Dict]:
    errors: List[str] = []
    normalized: Dict[str, Any] = {}

    for key, value in payload.items():
        canonical_key = FIELD_ALIASES.get(key)
        if canonical_key is not None:
            normalized[canonical_key] = value

    for field in REQUIRED_API_FIELDS:
        if field not in normalized or normalized[field] in (None, ""):
            errors.append(f"Missing required field: {field}")

    for field in NUMERIC_API_FIELDS:
        if field not in normalized or normalized[field] in (None, ""):
            continue
        try:
            normalized[field] = float(normalized[field])
        except (TypeError, ValueError):
            errors.append(f"{field} must be a number")

    if "star_luminosity" in normalized and normalized["star_luminosity"] <= 0:
        errors.append("star_luminosity must be positive")

    if "spectral_type" in normalized:
        value = normalized["spectral_type"]
        if not isinstance(value, str) or not value.strip():
            errors.append("spectral_type must be a non-empty string")
        else:
            normalized["spectral_type"] = value.strip()[0].upper()

    if "planet_name" in normalized and normalized["planet_name"] is not None:
        normalized["planet_name"] = str(normalized["planet_name"]).strip() or "Custom Planet"
    if "host_name" in normalized and normalized["host_name"] is not None:
        normalized["host_name"] = str(normalized["host_name"]).strip() or "Custom Host"

    return errors, normalized


def _inference_defaults(profile: Dict[str, Any]) -> Dict[str, float]:
    defaults = DEFAULT_NUMERIC_VALUES.copy()
    defaults.update(profile.get("numeric_medians", {}))
    return defaults


def build_inference_raw_frame(normalized: Dict, preprocessing_profile: Dict[str, Any]) -> pd.DataFrame:
    defaults = _inference_defaults(preprocessing_profile)
    spectral_mode = preprocessing_profile.get("spectral_type_mode", "G")
    outlier_bounds = preprocessing_profile.get("outlier_bounds", {})

    if "star_luminosity" in normalized:
        star_luminosity = normalized["star_luminosity"]
    elif "star_luminosity_log" in normalized:
        star_luminosity = float(np.power(10.0, normalized["star_luminosity_log"]))
    else:
        star_luminosity = defaults["st_lum"]

    raw_row = pd.DataFrame(
        [
            {
                "pl_name": normalized.get("planet_name", "Custom Planet"),
                "hostname": normalized.get("host_name", "Custom Host"),
                "pl_rade": normalized.get("planet_radius", defaults["pl_rade"]),
                "pl_bmasse": normalized.get("planet_mass", defaults["pl_bmasse"]),
                "pl_orbper": normalized.get("orbital_period", defaults["pl_orbper"]),
                "pl_orbsmax": normalized.get("semi_major_axis", defaults["pl_orbsmax"]),
                "pl_eqt": normalized.get("equilibrium_temperature", defaults["pl_eqt"]),
                "pl_dens": normalized.get("planet_density", defaults["pl_dens"]),
                "st_teff": normalized.get("star_temperature", defaults["st_teff"]),
                "st_lum": star_luminosity,
                "st_met": normalized.get("star_metallicity", defaults["st_met"]),
                "st_spectype": normalized.get("spectral_type", spectral_mode),
            }
        ]
    )

    raw_row = _fill_identifier_columns(raw_row)
    raw_row["st_spectype"] = _normalize_spectral_type(raw_row["st_spectype"], fill_value=spectral_mode)

    for column in NUMERIC_COLUMNS:
        raw_row[column] = pd.to_numeric(raw_row[column], errors="coerce")
        lower_hard, upper_hard = PHYSICAL_LIMITS[column]
        raw_row[column] = raw_row[column].where(raw_row[column].between(lower_hard, upper_hard), np.nan)
        raw_row[column] = raw_row[column].fillna(defaults[column])

        lower, upper = outlier_bounds.get(column, (lower_hard, upper_hard))
        raw_row[column] = raw_row[column].clip(lower, upper)

    return raw_row


def build_inference_frame(
    normalized: Dict,
    feature_columns: List[str],
    preprocessing_profile: Dict[str, Any],
) -> pd.DataFrame:
    raw_row = build_inference_raw_frame(normalized, preprocessing_profile)
    model_row = engineer_features(raw_row)

    for column in feature_columns:
        if column not in model_row.columns:
            model_row[column] = 0.0

    return model_row[feature_columns]
