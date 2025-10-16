# nanoftirtools — AFM & NEASPEC utilities
# Copyright (C) 2025  Till Leissner (SDU)
#
# This file is part of nanoftirtools.
#
# nanoftirtools is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# nanoftirtools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with nanoftirtools.  If not, see <https://www.gnu.org/licenses/>.


from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------
# Input normalization helpers
# ---------------------------

def _as_peaks_array(peaks: Any, column: str = "x") -> np.ndarray:
    """
    Accepts:
      - PeakResult (with .peaks DataFrame and a column 'x')
      - pandas DataFrame with a column named `column` (default 'x')
      - 1D array-like of wavenumbers
    Returns a sorted 1D numpy array (float64).
    """
    if peaks is None:
        return np.array([], dtype=float)
    # PeakResult (from nanoftirtools.peaks)
    if hasattr(peaks, "peaks") and isinstance(getattr(peaks, "peaks"), pd.DataFrame):
        df = peaks.peaks
        if column not in df.columns:
            raise KeyError(f"PeakResult.peaks lacks a '{column}' column.")
        arr = np.asarray(df[column], dtype=float)
        return np.sort(arr[np.isfinite(arr)])
    # DataFrame
    if isinstance(peaks, pd.DataFrame):
        if column not in peaks.columns:
            raise KeyError(f"peaks DataFrame lacks a '{column}' column.")
        arr = np.asarray(peaks[column], dtype=float)
        return np.sort(arr[np.isfinite(arr)])
    # 1D array-like
    arr = np.asarray(peaks, dtype=float).ravel()
    return np.sort(arr[np.isfinite(arr)])


@dataclass
class ReferenceEntry:
    name: str
    peaks: np.ndarray          # 1D array of wavenumbers (cm^-1)
    weights: Optional[np.ndarray] = None  # optional weights/intensities


def load_library_csv(path: str | Path) -> list[ReferenceEntry]:
    """
    Load a reference IR library from CSV.

    Supported formats:

    (A) Wide 'peaks-as-string' format:
        columns: name, peaks[, weights]
        - 'peaks' is a comma/semicolon-separated string of numbers (cm^-1)
        - 'weights' optional: same count as 'peaks'

    (B) Long 'tidy' format:
        columns: name, wavenumber[, weight]
        - one row per peak; grouped by 'name'
    """
    path = Path(path)

    # First attempt: ignore comment lines starting with '#'
    try:
        df = pd.read_csv(path, comment="#", engine="python")
    except Exception:
        # Fallback plain read
        df = pd.read_csv(path, engine="python")

    # If columns don't include 'name' but the file looks like 2–3 plain columns,
    # assume wide format without header and assign names.
    lower_cols = [c.lower().strip().lstrip("#").strip() for c in df.columns]
    if "name" not in lower_cols and df.shape[1] in (2, 3):
        df = pd.read_csv(
            path,
            comment="#",
            header=None,
            names=["name", "peaks"] + (["weights"] if df.shape[1] == 3 else []),
            engine="python",
        )
        lower_cols = [c.lower() for c in df.columns]

    # Normalize column dict (strip leading '#', spaces)
    cols = {c.lower().strip().lstrip("#").strip(): c for c in df.columns}
    lower = set(cols.keys())

    entries: list[ReferenceEntry] = []

    def _split_nums(s: str) -> np.ndarray:
        if not isinstance(s, str):
            return np.array([], dtype=float)
        s = s.strip().strip('"').strip("'")
        parts = [p for p in s.replace(";", ",").split(",") if p.strip() != ""]
        arr = np.array([float(p) for p in parts], dtype=float) if parts else np.array([], dtype=float)
        return np.sort(arr)

    # Wide format
    if {"name", "peaks"} <= lower:
        name_col = cols["name"]
        peaks_col = cols["peaks"]
        weights_col = cols.get("weights")

        for _, row in df.iterrows():
            name = str(row[name_col]).strip().strip('"').strip("'")
            peaks_arr = _split_nums(str(row[peaks_col]))
            weights_arr = None
            if weights_col is not None and isinstance(row[weights_col], str):
                w = _split_nums(str(row[weights_col]))
                if w.size and w.size == peaks_arr.size:
                    weights_arr = w
            entries.append(ReferenceEntry(name=name, peaks=peaks_arr, weights=weights_arr))
        return entries

    # Long format
    if {"name", "wavenumber"} <= lower:
        name_col = cols["name"]
        wn_col = cols["wavenumber"]
        weight_col = cols.get("weight")

        grouped = df.groupby(name_col, sort=False)
        for name, g in grouped:
            nm = str(name).strip().strip('"').strip("'")
            wn = np.asarray(g[wn_col], dtype=float)
            wn = np.sort(wn[np.isfinite(wn)])
            weights_arr = None
            if weight_col is not None and weight_col in g.columns:
                w = np.asarray(g[weight_col], dtype=float)
                if w.size == wn.size:
                    weights_arr = w
            entries.append(ReferenceEntry(name=nm, peaks=wn, weights=weights_arr))
        return entries

    raise ValueError(
        "CSV format not recognized. Provide either:\n"
        "  (A) columns: name, peaks[, weights]\n"
        "  (B) columns: name, wavenumber[, weight]"
    )

# ---------------------------
# Matching & scoring
# ---------------------------

@dataclass
class MatchResult:
    table: pd.DataFrame            # ranked matches (one row per reference)
    matches_by_ref: dict[str, list[tuple[float, float, float]]]  
    # dict[name] -> list of (query_peak, ref_peak, delta_cm1)


def _pairwise_match(query: np.ndarray, ref: np.ndarray, tol: float) -> list[tuple[float, float, float]]:
    """
    Greedy nearest-neighbor matching within tolerance.
    Returns list of (q, r, |q-r|). Each query/ref used at most once.
    """
    if query.size == 0 or ref.size == 0:
        return []
    qi = 0
    ri = 0
    pairs: list[tuple[float, float, float]] = []
    while qi < len(query) and ri < len(ref):
        q = query[qi]
        r = ref[ri]
        d = q - r
        if abs(d) <= tol:
            pairs.append((q, r, abs(d)))
            qi += 1
            ri += 1
        elif d > 0:
            ri += 1
        else:
            qi += 1
    return pairs


def match_peaks_to_library(
    peaks: Any,
    library: list[ReferenceEntry] | pd.DataFrame | str | Path,
    *,
    tol: float = 5.0,                 # ± cm^-1 tolerance
    xcol_in_peaks: str = "x",         # column name if peaks is a DataFrame or PeakResult
    min_hits: int = 1,                # require at least this many matched peaks
    sort_by: str = "f1"               # 'f1' | 'recall' | 'precision' | 'hits'
) -> MatchResult:
    """
    Match detected peaks against a reference library.
    Returns a ranked table with precision/recall/F1 and tie-breaker stats.

    - peaks: PeakResult, DataFrame (with xcol_in_peaks), or 1D array-like of wavenumbers.
    - library: 
        * list[ReferenceEntry] (already parsed), or
        * long DataFrame with columns ('name','wavenumber'[, 'weight']), or
        * path to a CSV accepted by load_library_csv().
    """
    q = _as_peaks_array(peaks, column=xcol_in_peaks)

    # Normalize library
    entries: list[ReferenceEntry]
    if isinstance(library, list) and all(isinstance(e, ReferenceEntry) for e in library):
        entries = library  # already parsed
    elif isinstance(library, (str, Path)):
        entries = load_library_csv(library)
    elif isinstance(library, pd.DataFrame):
        # Treat as long-form DataFrame
        tmp_path = Path("_in_memory_.csv")  # not actually written; use a branch
        # We can parse directly without writing:
        # Ensure required columns:
        cols = {c.lower(): c for c in library.columns}
        if not {"name", "wavenumber"} <= set(cols.keys()):
            raise ValueError("library DataFrame must have columns: 'name', 'wavenumber'[, 'weight']")
        name_col = cols["name"]; wn_col = cols["wavenumber"]; weight_col = cols.get("weight")
        entries = []
        for name, g in library.groupby(name_col, sort=False):
            wn = np.sort(np.asarray(g[wn_col], dtype=float))
            weights_arr = None
            if weight_col is not None and weight_col in g.columns:
                w = np.asarray(g[weight_col], dtype=float)
                if w.size == wn.size:
                    weights_arr = w
            entries.append(ReferenceEntry(str(name), wn, weights_arr))
    else:
        raise TypeError("Unsupported library type. Pass a list[ReferenceEntry], a DataFrame, or a CSV path.")

    rows = []
    by_ref: dict[str, list[tuple[float, float, float]]] = {}

    for ref in entries:
        pairs = _pairwise_match(q, ref.peaks, tol=tol)
        hits = len(pairs)
        if hits < min_hits:
            continue

        # Metrics
        precision = hits / max(len(q), 1)
        recall = hits / max(len(ref.peaks), 1)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        # spread of mismatches (lower is better)
        mad = float(np.mean([d for _, _, d in pairs])) if hits else np.inf

        rows.append({
            "name": ref.name,
            "hits": hits,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_abs_delta": mad,
            "ref_peaks": len(ref.peaks),
            "query_peaks": len(q),
        })
        by_ref[ref.name] = pairs

    if not rows:
        return MatchResult(
            table=pd.DataFrame(columns=["name","hits","precision","recall","f1","mean_abs_delta","ref_peaks","query_peaks"]),
            matches_by_ref={}
        )

    result = pd.DataFrame(rows)

    # Sort policy
    if sort_by == "recall":
        result = result.sort_values(["recall","precision","hits", "mean_abs_delta"], ascending=[False, False, False, True])
    elif sort_by == "precision":
        result = result.sort_values(["precision","recall","hits", "mean_abs_delta"], ascending=[False, False, False, True])
    elif sort_by == "hits":
        result = result.sort_values(["hits","f1","mean_abs_delta"], ascending=[False, False, True])
    else:  # f1
        result = result.sort_values(["f1","precision","recall","hits","mean_abs_delta"], ascending=[False, False, False, False, True])

    result = result.reset_index(drop=True)
    return MatchResult(table=result, matches_by_ref=by_ref)
