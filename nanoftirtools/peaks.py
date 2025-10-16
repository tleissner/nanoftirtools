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
import numpy as np
import pandas as pd
from typing import Literal, Optional, Tuple, Dict, Any

from scipy.signal import find_peaks, peak_widths, savgol_filter

try:
    # pybaselines >=1.0
    from pybaselines import Baseline
    _HAS_PYBASELINES = True
except Exception:
    _HAS_PYBASELINES = False


@dataclass
class PeakResult:
    peaks: pd.DataFrame          # table of peaks (x, y, prominence, fwhm, etc.)
    baseline: Optional[np.ndarray]  # estimated baseline (same length as y) or None
    y_corrected: np.ndarray      # y - baseline (or original y if baseline not used)


def detect_peaks(
    df: pd.DataFrame,
    *,
    xcol: str = "Wavenumber",
    ycol: str = "O1A",

    # --- preprocessing ---
    smooth_window: Optional[int] = 11,   # Savitzky–Golay window (odd). None to disable.
    polyorder: int = 3,

    # --- baseline correction (pybaselines) ---
    use_baseline: bool = True,
    baseline_method: Literal["asls", "airpls", "iasls", "penalized"] = "asls",
    baseline_kwargs: Optional[Dict[str, Any]] = None,  # e.g. {"lam": 1e6, "p": 0.001, "max_iter": 25}

    # --- peak picking (scipy.signal.find_peaks) ---
    height: Optional[float] = None,
    prominence: Optional[float] = 0.0,
    distance: Optional[int] = None,
    width: Optional[int] = None,
) -> PeakResult:
    """
    Detect peaks in df[ycol] vs df[xcol] with optional baseline correction (pybaselines)
    and optional Savitzky–Golay smoothing. Returns PeakResult with a peaks DataFrame.

    Peaks table columns: ['x', 'y', 'prominence', 'height', 'left_x', 'right_x', 'fwhm_x', 'index'].
    """
    if xcol not in df.columns or ycol not in df.columns:
        raise KeyError(f"Missing columns: {xcol!r} or {ycol!r}")

    x = df[xcol].to_numpy()
    y_raw = df[ycol].to_numpy(dtype=float)

    # Optional smoothing (before baseline)
    if smooth_window and smooth_window > 2 and smooth_window % 2 == 1 and smooth_window <= len(y_raw):
        y_smooth = savgol_filter(y_raw, smooth_window, polyorder)
    else:
        y_smooth = y_raw

    # Baseline correction
    baseline = None
    y_for_peaks = y_smooth

    if use_baseline:
        if not _HAS_PYBASELINES:
            raise ImportError("pybaselines is not installed. Run: pip install pybaselines")
        bl = Baseline(x)  # x is optional for most methods but fine to pass
        baseline_kwargs = baseline_kwargs or {}
        # choose method
        if baseline_method == "asls":
            baseline, _ = bl.asls(y_smooth, **({"lam": 1e6, "p": 0.001, "max_iter": 25} | baseline_kwargs))
        elif baseline_method == "airpls":
            baseline, _ = bl.airpls(y_smooth, **({"lam": 1e5, "porder": 1, "itermax": 50} | baseline_kwargs))
        elif baseline_method == "iasls":
            baseline, _ = bl.iasls(y_smooth, **({"lam": 1e6, "p": 0.001, "max_iter": 25} | baseline_kwargs))
        elif baseline_method == "penalized":
            baseline, _ = bl.penalized(y_smooth, **({"lam": 1e5} | baseline_kwargs))
        else:
            raise ValueError(f"Unsupported baseline_method: {baseline_method}")
        y_for_peaks = y_smooth - baseline

    # Peak picking on corrected signal
    peaks_idx, props = find_peaks(
        y_for_peaks,
        height=height,
        prominence=prominence,
        distance=distance,
        width=width
    )

    # FWHM (in index coords), then convert to x-units
    if len(peaks_idx):
        w_res = peak_widths(y_for_peaks, peaks_idx, rel_height=0.5)
        left_x  = np.interp(w_res[2], np.arange(len(x)), x)
        right_x = np.interp(w_res[3], np.arange(len(x)), x)
        fwhm_x  = np.abs(right_x - left_x)
    else:
        left_x = right_x = fwhm_x = np.array([])

    peaks_df = pd.DataFrame({
        "x": x[peaks_idx],
        "y": y_for_peaks[peaks_idx],  # NOTE: corrected amplitude
        "prominence": props.get("prominences", np.array([])),
        "height": props.get("peak_heights", y_for_peaks[peaks_idx] if len(peaks_idx) else np.array([])),
        "left_x": left_x,
        "right_x": right_x,
        "fwhm_x": fwhm_x,
        "index": peaks_idx,
    }).sort_values("x" if x[0] <= x[-1] else "x", ascending=True).reset_index(drop=True)

    return PeakResult(peaks=peaks_df, baseline=baseline, y_corrected=y_for_peaks)
