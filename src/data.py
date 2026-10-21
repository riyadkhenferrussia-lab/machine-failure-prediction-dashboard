"""Dataset acquisition, loading, validation and the train/test split.

Command line::

    python -m src.data                   # validate the local CSV (default)
    python -m src.data --download        # fetch the official CSV from UCI if missing
    python -m src.data --verify-official # compare the local CSV with the official download
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as cfg
from .schema import standardize_columns
from .utils import sha256_of_file, write_json
from .validation import DataValidationError, ValidationReport, validate_dataset


class DatasetDownloadError(RuntimeError):
    """The official dataset could not be downloaded or unpacked."""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def read_raw_csv(path: Path | str | None = None) -> pd.DataFrame:
    """Read the CSV exactly as published (original column names)."""
    path = Path(path) if path is not None else cfg.RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python -m src.data --download` or download "
            f"ai4i2020.csv manually from {cfg.UCI_DATASET_PAGE} and place it there."
        )
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"{path} is not a readable CSV file: {exc}") from exc


def load_and_validate(path: Path | str | None = None) -> tuple[pd.DataFrame, ValidationReport]:
    """Load the dataset with canonical column names and return it with its report."""
    frame = standardize_columns(read_raw_csv(path))
    report = validate_dataset(frame, expected_rows=cfg.DATASET_ROWS)
    if report.ok:
        ordered = [c for c in cfg.REQUIRED_COLUMNS if c in frame.columns]
        extra = [c for c in frame.columns if c not in ordered]
        frame = frame[ordered + extra].reset_index(drop=True)
    return frame, report


def load_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Load and validate the dataset; raises ``DataValidationError`` on blocking problems."""
    frame, report = load_and_validate(path)
    report.raise_if_errors()
    return frame


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def split_dataset(
    df: pd.DataFrame,
    test_size: float = cfg.TEST_SIZE,
    seed: int = cfg.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified hold-out split on the failure label.

    The rows are independent observations (no timestamps), so a random stratified
    split is appropriate. Both parts keep all columns (ids and failure-mode flags
    included) so results can be traced and analysed; models only ever receive
    ``cfg.RAW_FEATURES`` (see ``features_and_target``).
    """
    train, test = train_test_split(
        df, test_size=test_size, stratify=df[cfg.TARGET], random_state=seed
    )
    train = train.sort_values("udi").reset_index(drop=True)
    test = test.sort_values("udi").reset_index(drop=True)
    overlap = set(train["udi"]) & set(test["udi"])
    if overlap:  # pragma: no cover - guards against a future change to the split
        raise RuntimeError(f"Train/test overlap detected for {len(overlap)} rows")
    return train, test


def features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select the model inputs (raw sensor columns + Type) and the binary target."""
    return df[cfg.RAW_FEATURES].copy(), df[cfg.TARGET].astype(int).copy()


# --------------------------------------------------------------------------- #
# Download / verification against the official source
# --------------------------------------------------------------------------- #
def _extract_csv(payload: bytes) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise DatasetDownloadError("The downloaded archive does not contain a CSV file.")
            chosen = next((n for n in names if Path(n).name.lower() == "ai4i2020.csv"), names[0])
            return archive.read(chosen)
    except zipfile.BadZipFile as exc:
        raise DatasetDownloadError("The downloaded file is not a valid ZIP archive.") from exc


def fetch_official_csv(url: str = cfg.UCI_DATASET_URL, timeout: float = 60.0) -> bytes:
    """Download the official ZIP and return the CSV bytes."""
    request = Request(url, headers={"User-Agent": "machine-failure-dashboard/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https URL
            payload = response.read()
    except OSError as exc:  # URLError, HTTPError, timeouts and TLS errors are OSErrors
        raise DatasetDownloadError(
            f"Could not download the dataset from {url}: {exc}. Download ai4i2020.csv manually "
            f"from {cfg.UCI_DATASET_PAGE} and place it in {cfg.RAW_DATA_PATH}."
        ) from exc
    return _extract_csv(payload)


def download_dataset(
    dest: Path | str | None = None, url: str = cfg.UCI_DATASET_URL, force: bool = False
) -> Path:
    """Download the official CSV to ``dest`` (validated before it is written)."""
    dest = Path(dest) if dest is not None else cfg.RAW_DATA_PATH
    if dest.exists() and not force:
        return dest
    content = fetch_official_csv(url)
    frame = standardize_columns(pd.read_csv(io.BytesIO(content)))
    validate_dataset(frame, expected_rows=cfg.DATASET_ROWS).raise_if_errors()
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(dest)
    return dest


def verify_against_official(
    path: Path | str | None = None, url: str = cfg.UCI_DATASET_URL
) -> dict[str, object]:
    """Compare the local CSV with a fresh download of the official file (kept in memory)."""
    local_path = Path(path) if path is not None else cfg.RAW_DATA_PATH
    official_bytes = fetch_official_csv(url)
    local_bytes = local_path.read_bytes()
    local_df = standardize_columns(pd.read_csv(local_path)).reset_index(drop=True)
    official_df = standardize_columns(pd.read_csv(io.BytesIO(official_bytes))).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(local_df, official_df, check_dtype=False)
        same_content = True
    except AssertionError:
        same_content = False
    return {
        "identical_bytes": official_bytes == local_bytes,
        "identical_content": same_content,
        "local_sha256": sha256_of_file(local_path),
        "official_rows": int(len(official_df)),
        "local_rows": int(len(local_df)),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.data", description=__doc__.split("\n\n")[0])
    parser.add_argument("--download", action="store_true", help="download the official CSV from UCI")
    parser.add_argument("--force", action="store_true", help="overwrite an existing local CSV")
    parser.add_argument(
        "--verify-official",
        action="store_true",
        help="download the official CSV into memory and compare it with the local file",
    )
    args = parser.parse_args(argv)

    try:
        if args.download:
            path = download_dataset(force=args.force)
            print(f"Dataset available at {path}")
        if args.verify_official:
            result = verify_against_official()
            print(result)
            return 0 if result["identical_content"] else 1
        frame, report = load_and_validate()
    except (DatasetDownloadError, FileNotFoundError, DataValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    write_json(cfg.DATA_VALIDATION_PATH, report.to_dict())
    print(f"Rows: {len(frame):,}   columns: {frame.shape[1]}   sha256: {sha256_of_file(cfg.RAW_DATA_PATH)}")
    print(f"Validation: {'PASSED' if report.ok else 'FAILED'} "
          f"({len(report.errors)} errors, {len(report.warnings)} warnings)")
    for issue in report.issues:
        print(f"  [{issue.level}] {issue.code}: {issue.message}")
    print(f"Report written to {cfg.DATA_VALIDATION_PATH}")
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
