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
from pathlib import Path
from typing import Iterable, Sequence, Pattern, Literal, Callable, Optional
import re, fnmatch
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .spectrum import load_neaspec_spectrum, Spectrum
from .plotting import _nonzero_window  
from .spectrum_utils import list_spectra, NF_S_RE

from .spectrum import load_neaspec_spectrum, Spectrum
from .spectrum_utils import list_spectra, NF_S_RE

def _norm(s: str) -> str:
    """Normalize spaces (incl. NBSP) and trim; keep original case handling to caller."""
    return s.replace("\u00A0", " ").strip()

def _lower_norm(s: str) -> str:
    return _norm(s).lower()

def load_many_from_folder(
    folder: str | Path,
    *,
    recursive: bool = True,
    regex = NF_S_RE,                           # base filter for *NF S.txt
    # ---- NEW include filters (OR-combined) ----
    filenames: Iterable[str] | None = None,    # match by file name, e.g. "2025-10-07 144814 NF S.txt"
    stems: Iterable[str] | None = None,        # match by stem (no extension), e.g. "2025-10-07 144814 NF S"
    patterns: Iterable[str] | None = None,     # glob patterns for names, e.g. "*1448*NF S.txt"
    regex_name: Pattern[str] | None = None,    # regex for names
    custom: Callable[[Path], bool] | None = None,  # any user predicate(Path) -> bool
    case_sensitive: bool = False,
    # ---- NEW exclude filters (OR-combined) ----
    exclude_filenames: Iterable[str] | None = None,
    exclude_stems: Iterable[str] | None = None,
    exclude_patterns: Iterable[str] | None = None,
    exclude_regex_name: Pattern[str] | None = None,
) -> list[Spectrum]:
    """
    Return Spectrum[] from a folder, with optional include/exclude filters.
    If no include/exclude given, behaves like before (loads all *NF S.txt).

    Matching semantics:
      - All *include* filters are OR-combined: a file is kept if it matches ANY include.
      - All *exclude* filters are OR-combined: a file kept so far is dropped if it matches ANY exclude.
      - By default matches are case-insensitive (set case_sensitive=True to change).
      - NBSP vs normal space is handled for name comparisons.
    """
    files = list_spectra(folder, recursive=recursive, regex=regex)

    # -- Fast return if nothing to filter --
    if not any([filenames, stems, patterns, regex_name, custom,
                exclude_filenames, exclude_stems, exclude_patterns, exclude_regex_name]):
        return [load_neaspec_spectrum(p) for p in sorted(files)]

    # Precompute normalized lookup sets
    def _maybe_lower(s: str) -> str:
        return _norm(s) if case_sensitive else _lower_norm(s)

    inc_names = { _maybe_lower(n) for n in (filenames or []) }
    inc_stems = { _maybe_lower(n) for n in (stems or []) }
    exc_names = { _maybe_lower(n) for n in (exclude_filenames or []) }
    exc_stems = { _maybe_lower(n) for n in (exclude_stems or []) }

    # Helper: name/stem/glob/regex checks
    def _matches_include(p: Path) -> bool:
        name = _maybe_lower(p.name)
        stem = _maybe_lower(p.stem)
        # names / stems
        if inc_names and name in inc_names: return True
        if inc_stems and stem in inc_stems: return True
        # glob patterns
        if patterns:
            for pat in patterns:
                patn = _norm(pat) if case_sensitive else _lower_norm(pat)
                tgt  = _norm(p.name) if case_sensitive else name
                if fnmatch.fnmatchcase(tgt, patn):
                    return True
        # regex
        if regex_name:
            # test on normalized (case depends on compiled regex flags)
            tgt = _norm(p.name)
            if regex_name.search(tgt):
                return True
        # custom callable
        if custom and custom(p): return True
        # If no include filters were provided at all, treat as match
        return not any([filenames, stems, patterns, regex_name, custom])

    def _matches_exclude(p: Path) -> bool:
        name = _maybe_lower(p.name)
        stem = _maybe_lower(p.stem)
        if exc_names and name in exc_names: return True
        if exc_stems and stem in exc_stems: return True
        if exclude_patterns:
            for pat in exclude_patterns:
                patn = _norm(pat) if case_sensitive else _lower_norm(pat)
                tgt  = _norm(p.name) if case_sensitive else name
                if fnmatch.fnmatchcase(tgt, patn):
                    return True
        if exclude_regex_name:
            tgt = _norm(p.name)
            if exclude_regex_name.search(tgt):
                return True
        return False

    # Apply filters
    kept: list[Path] = []
    for p in files:
        if not _matches_include(p):
            continue
        if _matches_exclude(p):
            continue
        kept.append(p)

    if not kept:
        raise FileNotFoundError("No spectra matched your include/exclude filters.")

    return [load_neaspec_spectrum(p) for p in sorted(kept)]


#specs = load_many_from_folder(
#    folder="../../2025-10-07_tests",
#    filenames=[
#        "2025-10-07 113626 NF S.txt",
#        "2025-10-07 144814 NF S.txt",
#    ],
#)

### Examples 
#specs = load_many_from_folder(
#    "../../2025-10-07_tests",
#    stems=["2025-10-07 113626 NF S", "2025-10-07 144814 NF S"],
#)

#specs = load_many_from_folder(
#    "../../2025-10-07_tests",
#    patterns=["*1136*NF S.txt", "*1448*NF S.txt"],
#)

#specs = load_many_from_folder(
#    "../../2025-10-07_tests",
#    patterns=["*NF S.txt"],                # include all
#    exclude_filenames=["2025-10-07 113626 NF S.txt"],  # …except this one
#)

#import re
#specs = load_many_from_folder(
#    "../../2025-10-07_tests",
#    regex_name=re.compile(r"^2025-10-07\s+1(13|44)\d+.*NF S\.txt$", re.I),
#)

# or pass any callable(Path)->bool:
#specs = load_many_from_folder(
#    "../../2025-10-07_tests",
#    custom=lambda p: "113626" in p.stem or "144814" in p.stem,
#)

#--- Stitching ---

try:
    from scipy.signal import savgol_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

@dataclass
class StitchReport:
    stitched: pd.DataFrame      # columns: [xcol, channel]
    scales: list[float]         # per-spec scale, anchor=1.0
    overlaps: list[tuple[float,float]]
    order: list[int]
    preferred: int
    x_window: tuple[float, float] | None   # effective window used (lo, hi)

def _resample_to(x_src: np.ndarray, y_src: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    if x_src[0] > x_src[-1]:
        x_src = x_src[::-1]; y_src = y_src[::-1]
    return np.interp(x_ref, x_src, y_src, left=np.nan, right=np.nan)

def _smooth_1d(y: np.ndarray,
               kind: Literal["none","savgol","gaussian"] = "none",
               *, window: int = 11, polyorder: int = 3, sigma: float = 2.0) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if kind == "none":
        return y
    if kind == "savgol":
        if not _HAS_SCIPY or len(y) < 5:
            return y
        w = max(3, min(int(window), len(y) - (1 - len(y) % 2)))
        if w % 2 == 0: w -= 1
        if w < 3 or w <= polyorder:
            return y
        ok = np.isfinite(y)
        out = y.copy()
        if ok.sum() >= w:
            out[ok] = savgol_filter(y[ok], window_length=w, polyorder=min(polyorder, w - 1))
        return out
    if kind == "gaussian":
        n = int(max(3, window))
        if n % 2 == 0: n += 1
        if n > len(y): n = len(y) if len(y) % 2 == 1 else len(y) - 1
        if n < 3: return y
        r = (n - 1) // 2
        xs = np.arange(-r, r + 1, dtype=float)
        ker = np.exp(-0.5 * (xs / float(max(1e-6, sigma)))**2); ker /= ker.sum()
        ok = np.isfinite(y)
        yfill = np.where(ok, y, 0.0); norm = np.where(ok, 1.0, 0.0)
        conv = np.convolve(yfill, ker, mode="same")
        normc = np.convolve(norm, ker, mode="same")
        with np.errstate(invalid="ignore", divide="ignore"):
            out = conv / np.where(normc == 0, np.nan, normc)
        return out
    return y

def stitch_spectra(
    specs: Sequence,  # Sequence[Spectrum]
    *,
    channel: str = "O1A",
    xcol: str = "Wavenumber",
    # grid building
    grid: Optional[np.ndarray] = None,
    spacing: Optional[float] = None,
    # overlap/scale merge
    method: Literal["overlap-scale-mean", "overlap-scale-first"] = "overlap-scale-mean",
    min_overlap: int = 10,
    robust: bool = True,
    preferred: Optional[int] = 0,
    prefer_weight: float = 0.7,
    order: Optional[Sequence[int]] = None,
    # smoothing (pre-merge, post-resample)
    smooth: Literal["none","savgol","gaussian"] = "savgol",
    smooth_window: int = 11,
    smooth_polyorder: int = 3,
    smooth_sigma: float = 2.0,
    x_window: Tuple[float, float] | None = None,       # explicit (lo, hi)
    auto_window: bool = False,                         # derive from non-zero y
    window_mode: Literal["union","intersection"] = "union",
    eps: float = 0.0,
    fill_value: float = 0.0,
) -> StitchReport:
    """
    Stitch spectra to a common grid with optional smoothing and windowing.

    Windowing:
      - If x_window=(lo, hi) is given, grid/data are clipped to that range.
      - Else if auto_window=True, compute per-spectrum non-zero windows of `channel`
        via _nonzero_window and combine across spectra using:
            * 'union'        -> [min(lo_i), max(hi_i)]
            * 'intersection' -> [max(lo_i), min(hi_i)]
      - If neither is set, uses full min..max coverage across inputs.
    """
    if not specs:
        raise ValueError("No spectra provided.")
    n = len(specs)

    pref_idx = 0 if preferred is None else int(preferred)
    if not (0 <= pref_idx < n):
        raise IndexError(f"preferred index {preferred} out of range 0..{n-1}")

    # Axis direction from preferred
    x0 = specs[pref_idx].df[xcol].to_numpy()
    descending = x0[0] > x0[-1]

    # Determine global raw bounds across all specs
    mins = [np.nanmin(s.df[xcol].to_numpy()) for s in specs]
    maxs = [np.nanmax(s.df[xcol].to_numpy()) for s in specs]
    raw_lo, raw_hi = float(np.nanmin(mins)), float(np.nanmax(maxs))

    # Compute effective window
    eff_lo, eff_hi = None, None
    if x_window is not None:
        lo, hi = float(min(x_window)), float(max(x_window))
        eff_lo, eff_hi = lo, hi
    elif auto_window:
        los, his = [], []
        for s in specs:
            if channel not in s.df.columns: continue
            lo_i, hi_i, _ = _nonzero_window(s.df, xcol, [channel], eps=eps)
            if lo_i is not None and hi_i is not None:
                los.append(lo_i); his.append(hi_i)
        if los and his:
            if window_mode == "intersection":
                eff_lo, eff_hi = float(np.max(los)), float(np.min(his))
            else:  # union
                eff_lo, eff_hi = float(np.min(los)), float(np.max(his))
    # Fallback to full range
    if eff_lo is None or eff_hi is None or eff_hi <= eff_lo:
        eff_lo, eff_hi = (raw_lo, raw_hi)

    # Build grid (clipped to window)
    if grid is None:
        if spacing is None:
            dx = np.median(np.abs(np.diff(specs[pref_idx].df[xcol].to_numpy())))
            spacing = float(dx if np.isfinite(dx) and dx > 0 else 1.0)
        if not descending:
            grid = np.arange(eff_lo, eff_hi + spacing, spacing, dtype=float)
        else:
            grid = np.arange(eff_hi, eff_lo - spacing, -spacing, dtype=float)
    else:
        # Clip provided grid to window and keep direction
        if grid[0] <= grid[-1]:
            grid = grid[(grid >= eff_lo) & (grid <= eff_hi)]
        else:
            grid = grid[(grid <= eff_lo) & (grid >= eff_hi)]
        if grid.size == 0:
            raise ValueError("Provided grid has no points inside the requested window.")

    # Resample then smooth (on-window grid)
    Ys = []
    for s in specs:
        df = s.df
        if xcol not in df or channel not in df:
            raise KeyError(f"Spectrum missing {xcol!r} or {channel!r}")
        x = df[xcol].to_numpy()
        y = df[channel].to_numpy(dtype=float)
        yr = _resample_to(x, y, grid)
        ys = _smooth_1d(yr, kind=smooth, window=smooth_window,
                        polyorder=smooth_polyorder, sigma=smooth_sigma)
        Ys.append(ys.astype(float))

    # Determine order
    if order is not None:
        order = list(order)
        if set(order) != set(range(n)):
            raise ValueError("order must be a permutation of 0..n-1")
    else:
        order = [pref_idx] + list(range(pref_idx+1, n)) + list(range(pref_idx-1, -1, -1))

    scales = [np.nan] * n
    overlaps: list[tuple[float,float]] = [(np.nan, np.nan)] * n
    scales[pref_idx] = 1.0
    stitched = Ys[pref_idx].copy()

    def overlap_scale(prev: np.ndarray, cur: np.ndarray) -> tuple[float, tuple[float,float], np.ndarray]:
        mask = np.isfinite(prev) & np.isfinite(cur)
        if mask.sum() < max(min_overlap, 1):
            return 1.0, (np.nan, np.nan), cur
        x_overlap = grid[mask]
        a, b = prev[mask], cur[mask]
        if robust:
            denom = np.where(b == 0, np.nan, b)
            ratios = a / denom
            s = float(np.nanmedian(ratios))
            if not np.isfinite(s) or s == 0: s = 1.0
        else:
            num = float(np.nansum(b * a))
            den = float(np.nansum(b * b))
            s = (num / den) if den > 0 else 1.0
        return s, (float(np.nanmin(x_overlap)), float(np.nanmax(x_overlap))), s * cur

    for idx in order:
        if idx == pref_idx:
            continue
        cur = Ys[idx]
        s, rng, cur_scaled = overlap_scale(stitched, cur)
        scales[idx] = s
        overlaps[idx] = rng

        if method == "overlap-scale-mean":
            both = np.isfinite(stitched) & np.isfinite(cur_scaled)
            only_prev = np.isfinite(stitched) & ~np.isfinite(cur_scaled)
            only_cur  = ~np.isfinite(stitched) & np.isfinite(cur_scaled)
            out = np.empty_like(stitched); out[:] = np.nan
            if np.any(both):
                w = float(min(max(prefer_weight, 0.0), 1.0))
                out[both] = w * stitched[both] + (1.0 - w) * cur_scaled[both]
            out[only_prev] = stitched[only_prev]
            out[only_cur]  = cur_scaled[only_cur]
            stitched = out
        else:  # "overlap-scale-first"
            out = stitched.copy()
            fill = ~np.isfinite(out) & np.isfinite(cur_scaled)
            out[fill] = cur_scaled[fill]
            stitched = out

    out_df = pd.DataFrame({xcol: grid, channel: stitched})

    out_df[channel] = np.nan_to_num(out_df[channel].to_numpy(dtype=float),
                                nan=fill_value, posinf=fill_value, neginf=fill_value)
    scales = [1.0 if not np.isfinite(v) else float(v) for v in scales]
    return StitchReport(
        stitched=out_df, scales=scales, overlaps=overlaps,
        order=order, preferred=pref_idx, x_window=(eff_lo, eff_hi)
    )


# ---------- Plot multiple spectra in one figure ----------

def plot_many_specs(
    specs: Sequence[Spectrum],
    *,
    channel: str = "O1A",
    xcol: str = "Wavenumber",
    title: str | None = None,
    start_hidden: bool = True,
    window: bool = True,
    eps: float = 0.0,
    label: Literal["stem","name","project"] = "stem",
) -> go.Figure:
    """
    Plot `channel` from many Spectrum objects in one Plotly figure.
    Each trace can be toggled via legend. Optionally crops to non-zero window
    across ALL spectra (combined).
    """
    if not specs:
        raise ValueError("No spectra provided.")

    fig = go.Figure()
    global_lo, global_hi = np.inf, -np.inf
    plotted = 0

    # determine combined non-zero window if requested
    if window:
        los, his = [], []
        for s in specs:
            df = s.df
            if channel not in df.columns or xcol not in df.columns:
                continue
            lo, hi, _ = _nonzero_window(df, xcol, [channel], eps=eps)
            if lo is not None and hi is not None:
                los.append(lo); his.append(hi)
        if los and his:
            global_lo = float(np.min(los))
            global_hi = float(np.max(his))

    for s in specs:
        df = s.df
        if xcol not in df.columns or channel not in df.columns:
            continue

        # label
        if label == "project":
            name = s.header.get("meta", {}).get("Project", "spectrum")
        elif label == "name":
            name = Path(s.header.get("path", s.header.get("meta", {}).get("filename", "spectrum"))).name
        else:  # stem
            p = Path(s.header.get("path", s.header.get("meta", {}).get("filename", "spectrum")))
            name = p.stem

        x = df[xcol].to_numpy()
        y = df[channel].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if window and np.isfinite(global_lo) and np.isfinite(global_hi) and global_hi > global_lo:
            m = valid & (x >= global_lo) & (x <= global_hi)
        else:
            m = valid

        fig.add_trace(go.Scatter(
            x=x[m], y=y[m], name=name, mode="lines",
            visible=("legendonly" if start_hidden else True),
            hovertemplate=f"{name}<br>{xcol}: %{{x}}<br>{channel}: %{{y}}<extra></extra>",
        ))
        plotted += 1

    if plotted == 0:
        raise RuntimeError(f"No valid data to plot for channel {channel!r}.")

    if window and np.isfinite(global_lo) and np.isfinite(global_hi) and global_hi > global_lo:
        # small margin
        pad = 0.01 * (global_hi - global_lo)
        fig.update_xaxes(range=[global_lo - pad, global_hi + pad])

    fig.update_layout(
        title=title or f"{channel} across {plotted} spectra",
        xaxis_title=xcol,
        yaxis_title=channel,
        hovermode="x unified",
        legend_title_text="Spectrum",
        updatemenus=[{
            "type": "buttons", "direction": "right",
            "x": 1.0, "y": 1.15, "xanchor": "right",
            "buttons": [
                {"label": "Show all", "method": "update",
                 "args": [{"visible": [True]*plotted}]},
                {"label": "Hide all", "method": "update",
                 "args": [{"visible": ["legendonly"]*plotted}]},
            ],
        }],
    )
    return fig