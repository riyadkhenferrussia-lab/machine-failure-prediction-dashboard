"""Column-name handling: map published / user-supplied names to canonical ones."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from . import config as cfg


def _norm(name: object) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


_LOOKUP: dict[str, str] = {}
for _raw, _canonical in cfg.RAW_TO_CANONICAL.items():
    _LOOKUP[_norm(_raw)] = _canonical
    _LOOKUP[_norm(_canonical)] = _canonical
for _alias, _canonical in cfg.COLUMN_ALIASES.items():
    _LOOKUP[_norm(_alias)] = _canonical


def canonical_column_name(name: object) -> str | None:
    """Return the canonical name for ``name`` or ``None`` if it is not a known column."""
    return _LOOKUP.get(_norm(name))


def column_mapping(columns: Iterable[object]) -> dict[object, str]:
    """Map original column labels to canonical names (unknown columns are skipped).

    Raises ``ValueError`` if two different columns resolve to the same canonical name.
    """
    mapping: dict[object, str] = {}
    seen: dict[str, object] = {}
    for column in columns:
        canonical = canonical_column_name(column)
        if canonical is None:
            continue
        if canonical in seen:
            raise ValueError(
                f"Columns {seen[canonical]!r} and {column!r} both map to {canonical!r}"
            )
        seen[canonical] = column
        mapping[column] = canonical
    return mapping


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known columns to their canonical snake_case names; keep unknown ones as is."""
    return df.rename(columns=column_mapping(df.columns))


def accepted_names(canonical: str) -> list[str]:
    """Human-readable list of the spellings accepted for a canonical column."""
    names = [canonical]
    for raw, canon in cfg.RAW_TO_CANONICAL.items():
        if canon == canonical and raw != canonical:
            names.append(raw)
    return names
