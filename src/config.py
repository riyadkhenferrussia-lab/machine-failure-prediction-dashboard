"""Central configuration: paths, column schema, physical limits and constants.

Everything that is a *decision* (thresholds, limits, seeds) lives here so that
the training code, the dashboard and the tests all read the same values.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "ai4i2020.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "failure_model.joblib"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"
IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
DATA_VALIDATION_PATH = REPORTS_DIR / "data_validation.json"
EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.json"
RECOMMENDATION_LEVELS_PATH = REPORTS_DIR / "recommendation_levels_holdout.csv"
RECOMMENDATION_DESIGN_PATH = REPORTS_DIR / "recommendation_design.csv"
MODEL_CARD_MD_PATH = REPORTS_DIR / "MODEL_CARD.md"
README_PATH = PROJECT_ROOT / "README.md"

# --------------------------------------------------------------------------- #
# Dataset source (UCI Machine Learning Repository, id 601, CC BY 4.0)
# --------------------------------------------------------------------------- #
UCI_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset"
UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
)
DATASET_DOI = "10.24432/C5HS5C"
DATASET_ROWS = 10_000

# --------------------------------------------------------------------------- #
# Column schema
# --------------------------------------------------------------------------- #
# Names in the published CSV  ->  canonical snake_case names used in the code.
RAW_TO_CANONICAL: dict[str, str] = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}
# Extra spellings accepted on input (the UCI web page calls the first column "UID").
COLUMN_ALIASES: dict[str, str] = {"uid": "udi"}

TARGET = "machine_failure"
IDENTIFIER_COLUMNS = ["udi", "product_id"]
FAILURE_MODE_COLUMNS = ["twf", "hdf", "pwf", "osf", "rnf"]
CATEGORICAL_FEATURE = "type"
NUMERIC_SENSOR_FEATURES = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
# The only columns a model is ever allowed to see.
RAW_FEATURES = [CATEGORICAL_FEATURE] + NUMERIC_SENSOR_FEATURES
ENGINEERED_FEATURES = ["temp_diff_k", "power_w", "strain_min_nm"]
# Columns that must never reach a model: they define or identify the label.
LEAKAGE_COLUMNS = [TARGET] + FAILURE_MODE_COLUMNS + IDENTIFIER_COLUMNS
REQUIRED_COLUMNS = list(RAW_TO_CANONICAL.values())

TYPE_VALUES = ["L", "M", "H"]

FEATURE_LABELS: dict[str, str] = {
    "type": "Product variant (Type)",
    "air_temp_k": "Air temperature [K]",
    "process_temp_k": "Process temperature [K]",
    "rotational_speed_rpm": "Rotational speed [rpm]",
    "torque_nm": "Torque [Nm]",
    "tool_wear_min": "Tool wear [min]",
    "temp_diff_k": "Temperature difference, process - air [K]",
    "power_w": "Mechanical power [W]",
    "strain_min_nm": "Overstrain index, tool wear x torque [min*Nm]",
}
SHORT_LABELS: dict[str, str] = {
    "type": "Type",
    "air_temp_k": "Air temp [K]",
    "process_temp_k": "Process temp [K]",
    "rotational_speed_rpm": "Speed [rpm]",
    "torque_nm": "Torque [Nm]",
    "tool_wear_min": "Tool wear [min]",
    "temp_diff_k": "Temp. difference [K]",
    "power_w": "Power [W]",
    "strain_min_nm": "Wear x torque [min*Nm]",
}
FAILURE_MODE_LABELS: dict[str, str] = {
    "twf": "Tool wear (TWF)",
    "hdf": "Heat dissipation (HDF)",
    "pwf": "Power (PWF)",
    "osf": "Overstrain (OSF)",
    "rnf": "Random (RNF)",
}

# Hard physical sanity limits for *inputs*. They are deliberately wide: their job
# is to catch typos and unit mistakes (e.g. Celsius instead of kelvin), not to
# describe the operating envelope. Values outside the range seen in training get
# a softer warning instead (see src/validation.py).
HARD_LIMITS: dict[str, tuple[float, float]] = {
    "air_temp_k": (200.0, 450.0),
    "process_temp_k": (200.0, 600.0),
    "rotational_speed_rpm": (0.0, 20_000.0),
    "torque_nm": (0.0, 1_000.0),
    "tool_wear_min": (0.0, 5_000.0),
}
MAX_BATCH_ROWS = 50_000
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# --------------------------------------------------------------------------- #
# Engineering rules used by the recommendation system (src/recommend.py)
# --------------------------------------------------------------------------- #
# Failure criteria as documented by the dataset authors (UCI page / Matzka 2020).
# They were re-derived from the CSV and reproduce the HDF, PWF and OSF labels
# exactly (see reports/data_validation.json). For real equipment, limits must
# come from the OEM specification and maintenance engineering, not from here.
POWER_MIN_W = 3_500.0
POWER_MAX_W = 9_000.0
HDF_TEMP_DIFF_MAX_K = 8.6
HDF_SPEED_MAX_RPM = 1_380.0
OSF_LIMITS_MIN_NM: dict[str, float] = {"L": 11_000.0, "M": 12_000.0, "H": 13_000.0}
# Tool wear: the documentation describes tool replacement/failure at a randomly
# chosen wear time between 200 and 240 min. This is a probabilistic window, not a
# deterministic criterion, so it only ever raises a "watch" status (from the start of
# the window; the training data show no raised failure rate below it).
TOOL_WEAR_WINDOW_START_MIN = 200.0
TOOL_WEAR_WATCH_MIN = TOOL_WEAR_WINDOW_START_MIN
# An indicator is "approaching" (status WATCH) when it is within this fraction of its
# limit. Heat dissipation uses a tighter margin because its watch region is large: on
# the training split a 10% margin put 17% of the machines on watch with hardly any
# raised failure rate (see reports/recommendation_design.csv for the comparison).
APPROACH_MARGIN = 0.10
HEAT_APPROACH_MARGIN = 0.05
# Model score levels, as multiples of the operating threshold.
ELEVATED_SCORE_FRACTION = 0.5

# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SEARCH_ITERATIONS = 24
# Operating point: maximise F-beta on out-of-fold training predictions. With
# beta = 2 recall counts four times as much as precision, reflecting the
# assumption that a missed failure is costlier than an unnecessary inspection.
THRESHOLD_BETA = 2.0

# Hyper-parameters used when training without the search (``--no-search``) and
# when the dashboard has to rebuild a missing model. These are the parameters the
# full random search selected for the final model (cross-validated PR-AUC on the
# training set only), as recorded in models/model_card.json.
DEFAULT_RF_PARAMS: dict[str, object] = {
    "n_estimators": 400,
    "max_depth": 24,
    "min_samples_leaf": 1,
    "max_features": 0.5,
    "class_weight": None,
}
DEFAULT_LR_PARAMS: dict[str, object] = {"C": 1.0}

MODEL_BUNDLE_FORMAT = 1
