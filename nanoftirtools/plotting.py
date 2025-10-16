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


# nanoftirtools/plotting.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import re
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from typing import Pattern, Iterable, Sequence, Mapping

from .spectrum import load_neaspec_spectrum
from .spectrum_utils import list_spectra, NF_S_RE



# -------- Channel discovery --------

# Matches: O1A, O2P, M0A, R-O3P, r-o1a, etc.
CHAN_RE = re.compile(r"^(?:R-)?[OM]\d+[AP]$", re.IGNORECASE)

def list_channels(df: pd.DataFrame, *, xcol: str = "Wavenumber") -> list[str]:
    """Return columns that look like NEASPEC channels, excluding xcol."""
    chans = [c for c in df.columns if c != xcol and CHAN_RE.match(str(c))]
    # keep order as in the DataFrame
    return chans

def choose_channel(df: pd.DataFrame,
                   preferred: Iterable[str] = ("O1A","O2A","O0A","O1P","O2P"),
                   *,
                   xcol: str = "Wavenumber") -> str:
    """Pick a sensible default channel; fall back to first detected channel."""
    for c in preferred:
        if c in df.columns:
            return c
    chans = list_channels(df, xcol=xcol)
    if chans:
        return chans[0]
    raise ValueError("No suitable channel columns found.")

def plot_channel_for_folder_plotly(folder: str | Path,
                                   channel: str,
                                   *,
                                   xcol: str = "Wavenumber",
                                   recursive: bool = False,
                                   eps: float = 0.0,
                                   label_by: str = "stem",   # 'stem' | 'name' | 'project'
                                   title: str | None = None,
                                   start_hidden: bool = True,
                                   regex: Pattern[str] = NF_S_RE) -> go.Figure:
    """
    Plot `channel` vs `xcol` for all spectra matching `regex` (default NF[space/NBSP/_/-]*S.txt).
    """
    files = list_spectra(folder, recursive=recursive, regex=regex)
    if not files:
        raise FileNotFoundError(f"No spectra matched in {folder!s} with pattern {regex.pattern!r}")

    fig = go.Figure()
    global_xmin, global_xmax = np.inf, -np.inf
    plotted = 0

    for f in sorted(files):
        try:
            spec = load_neaspec_spectrum(f)
            df = spec.df
        except Exception as e:
            print(f"[skip] {f.name}: load failed ({e})")
            continue

        if xcol not in df.columns or channel not in df.columns:
            print(f"[skip] {f.name}: missing {xcol!r} or {channel!r}")
            continue

        x = df[xcol].to_numpy()
        y = df[channel].to_numpy(dtype=float)

        valid = np.isfinite(x) & np.isfinite(y)
        nz = np.abs(y) > eps
        mask = valid & nz
        if not np.any(mask):
            print(f"[skip] {f.name}: no non-zero data in {channel!r} (eps={eps})")
            continue

        xmin, xmax = np.min(x[mask]), np.max(x[mask])
        global_xmin = min(global_xmin, xmin)
        global_xmax = max(global_xmax, xmax)

        # small per-trace margin
        m = 0.01 * (xmax - xmin) if np.isfinite(xmax - xmin) else 0.0
        lo, hi = xmin - m, xmax + m
        window = valid & (x >= lo) & (x <= hi)

        meta = spec.header.get("meta", {})
        if label_by == "project":
            label = meta.get("Project", f.stem)
        elif label_by == "name":
            label = f.name
        else:
            label = f.stem

        fig.add_trace(go.Scatter(
            x=x[window],
            y=y[window],
            mode="lines",
            name=label,
            hovertemplate=f"{label}<br>{xcol}: %{{x}}<br>{channel}: %{{y}}<extra></extra>",
            visible=("legendonly" if start_hidden else True),
        ))
        plotted += 1

    if plotted == 0:
        raise RuntimeError("Nothing to plot for this channel.")

    if np.isfinite(global_xmin) and np.isfinite(global_xmax) and global_xmax > global_xmin:
        m = 0.01 * (global_xmax - global_xmin)
        fig.update_xaxes(range=[global_xmin - m, global_xmax + m])

    fig.update_layout(
        title=title or f"{channel} across {plotted} file(s)",
        xaxis_title=xcol, yaxis_title=channel,
        hovermode="x unified",
        legend_title_text="File",
        updatemenus=[{
            "type": "buttons",
            "direction": "right",
            "x": 1.0, "y": 1.15, "xanchor": "right",
            "buttons": [
                {"label": "Show all", "method": "update", "args": [{"visible": [True]*plotted}]} ,
                {"label": "Hide all", "method": "update", "args": [{"visible": ["legendonly"]*plotted}]}
            ]
        }]
    )
    return fig


def plot_channel_for_folder(folder: str | Path,
                            channel: str,
                            *,
                            xcol: str = "Wavenumber",
                            recursive: bool = False,
                            eps: float = 0.0,
                            label_by: str = "stem",   # 'stem' | 'name' | 'project'
                            title: str | None = None):
    folder = Path(folder)
    it = folder.rglob("*.txt") if recursive else folder.glob("*.txt")
    files = [p for p in it if NF_S_RE.search(p.name)]
    if not files:
        print("No matching files. Debug hint: list txt files:",
              [repr(p.name) for p in (folder.rglob('*.txt') if recursive else folder.glob('*.txt'))][:10])
        return None

    fig, ax = plt.subplots()
    global_xmin, global_xmax = np.inf, -np.inf
    plotted = 0

    for f in sorted(files):
        try:
            spec = load_neaspec_spectrum(f)
            df = spec.df
        except Exception as e:
            print(f"[skip] {f.name}: load failed ({e})"); continue

        if xcol not in df or channel not in df:
            print(f"[skip] {f.name}: missing columns ({xcol!r} or {channel!r})"); continue

        x = df[xcol].to_numpy()
        y = df[channel].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        nz = np.abs(y) > eps
        mask = valid & nz
        if not np.any(mask):
            print(f"[skip] {f.name}: no non-zero data in {channel!r} (eps={eps})"); continue

        xmin, xmax = x[mask].min(), x[mask].max()
        global_xmin, global_xmax = min(global_xmin, xmin), max(global_xmax, xmax)

        margin = 0.01 * (xmax - xmin) if np.isfinite(xmax - xmin) else 0.0
        lo, hi = xmin - margin, xmax + margin
        window = valid & (x >= lo) & (x <= hi)

        if label_by == "name":
            label = f.name
        elif label_by == "project":
            label = spec.header.get("meta", {}).get("Project", f.stem)
        else:
            label = f.stem

        ax.plot(x[window], y[window], label=label)
        plotted += 1

    if plotted == 0:
        print("Nothing to plot: all files lacked non-zero data for this channel.")
        return None

    if np.isfinite(global_xmin) and np.isfinite(global_xmax) and global_xmax > global_xmin:
        m = 0.01 * (global_xmax - global_xmin)
        ax.set_xlim(global_xmin - m, global_xmax + m)

    ax.set_xlabel(xcol)
    ax.set_ylabel(channel)
    ax.set_title(title or f"{channel} across {plotted} file(s)")
    ax.legend()
    plt.show()
    return ax

# -------- Windowing utility --------

def _nonzero_window(df: pd.DataFrame, xcol: str, ycols: Sequence[str], eps: float = 0.0):
    """
    Return (lo, hi, mask) for rows where ANY y in ycols has |y| > eps.
    If nothing is non-zero, returns (None, None, valid_mask).
    """
    ycols = [c for c in ycols if c in df.columns]
    if not ycols:
        raise ValueError("No valid y columns found in DataFrame for non-zero window.")
    x = df[xcol].to_numpy()
    Y = df[ycols].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.all(np.isfinite(Y), axis=1)
    any_nz = np.any(np.abs(Y) > eps, axis=1)
    nz_mask = valid & any_nz
    if not np.any(nz_mask):
        return None, None, valid
    xmin = float(np.min(x[nz_mask])); xmax = float(np.max(x[nz_mask]))
    margin = 0.01 * (xmax - xmin) if np.isfinite(xmax - xmin) else 0.0
    lo, hi = xmin - margin, xmax + margin
    window = valid & (x >= lo) & (x <= hi)
    return lo, hi, window

def df_plot_plotly(df: pd.DataFrame,
                   *,
                   x: str,
                   y: str | Sequence[str],
                   eps: float = 0.0,
                   title: str | None = None,
                   start_hidden: bool = False,
                   window: bool = True) -> go.Figure:
    """Plot like pandas df.plot but (optionally) crop x-range to non-zero region."""
    ycols = [y] if isinstance(y, str) else list(y)
    lo = hi = None
    if window:
        lo, hi, mask = _nonzero_window(df, x, ycols, eps=eps)
    else:
        mask = np.isfinite(df[x].to_numpy())
        for c in ycols:
            if c in df:
                mask &= np.isfinite(df[c].to_numpy(dtype=float))
    d = df.loc[mask, [x, *[c for c in ycols if c in df.columns]]]
    fig = px.line(d, x=x, y=[c for c in ycols if c in d.columns], title=title)
    if start_hidden:
        for tr in fig.data:
            tr.visible = "legendonly"
    if window and lo is not None and hi is not None:
        fig.update_xaxes(range=[lo, hi])
    fig.update_layout(
        hovermode="x unified",
        legend_title_text="Channel",
        updatemenus=[{
            "type": "buttons", "direction": "right", "x": 1.0, "y": 1.15, "xanchor": "right",
            "buttons": [
                {"label": "Show all", "method": "update", "args": [{"visible": [True]*len(fig.data)}]},
                {"label": "Hide all", "method": "update", "args": [{"visible": ["legendonly"]*len(fig.data)}]},
            ],
        }],
    )
    return fig

# -------- Helper to accept PeakResult OR DataFrame --------

def _as_peaks_df(peaks_any: Any) -> pd.DataFrame | None:
    if peaks_any is None:
        return None
    if hasattr(peaks_any, "peaks") and isinstance(getattr(peaks_any, "peaks"), pd.DataFrame):
        return peaks_any.peaks
    if isinstance(peaks_any, pd.DataFrame):
        return peaks_any
    raise TypeError("peaks must be a pandas DataFrame, PeakResult, or None")


def _nonzero_window(df: pd.DataFrame, xcol: str, ycols: Sequence[str], eps: float = 0.0):
    ycols = [c for c in ycols if c in df.columns]
    if not ycols:
        raise ValueError("No valid y columns found in DataFrame for non-zero window.")
    x = df[xcol].to_numpy()
    Y = df[ycols].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.all(np.isfinite(Y), axis=1)
    any_nz = np.any(np.abs(Y) > eps, axis=1)
    nz_mask = valid & any_nz
    if not np.any(nz_mask):
        return None, None, valid
    xmin = np.min(x[nz_mask]); xmax = np.max(x[nz_mask])
    margin = 0.01 * (xmax - xmin) if np.isfinite(xmax - xmin) else 0.0
    lo, hi = xmin - margin, xmax + margin
    window = valid & (x >= lo) & (x <= hi)
    return lo, hi, window

def plot_spectrum_with_peaks(df: pd.DataFrame,
                             peaks: Any,
                             *,
                             xcol: str = "Wavenumber",
                             ycol: str = "O1A",
                             eps: float = 0.0,
                             title: str | None = None,
                             start_hidden: bool = False,
                             window: bool = True) -> go.Figure:
    """Plot a single channel with peaks (Plotly). Windowing optional."""
    if xcol not in df.columns or ycol not in df.columns:
        raise KeyError(f"Missing {xcol!r} or {ycol!r} in DataFrame.")
    lo = hi = None
    x_vals = df[xcol].to_numpy()
    y_vals = df[ycol].to_numpy(dtype=float)
    valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    if window:
        lo2, hi2, mask2 = _nonzero_window(df, xcol, [ycol], eps=eps)
        lo, hi = lo2, hi2
        mask = mask2 if mask2 is not None else valid_mask
    else:
        mask = valid_mask
    x = df.loc[mask, xcol].to_numpy()
    y = df.loc[mask, ycol].to_numpy(dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name=ycol,
        visible=("legendonly" if start_hidden else True),
        hovertemplate=f"{xcol}: %{{x}}<br>{ycol}: %{{y}}<extra></extra>",
    ))

    peaks_df = _as_peaks_df(peaks)
    if peaks_df is not None and not peaks_df.empty:
        if window and (lo is not None) and (hi is not None):
            p_mask = (peaks_df["x"] >= lo) & (peaks_df["x"] <= hi)
        else:
            p_mask = np.ones(len(peaks_df), dtype=bool)
        fig.add_trace(go.Scatter(
            x=peaks_df.loc[p_mask, "x"],
            y=peaks_df.loc[p_mask, "y"],
            mode="markers", name="peaks",
            marker=dict(size=9, symbol="x"),
            hovertemplate=f"Peak<br>{xcol}: %{{x}}<br>{ycol}: %{{y}}<extra></extra>",
        ))
        if {"left_x", "right_x", "y"}.issubset(peaks_df.columns):
            xs, ys = [], []
            for _, r in peaks_df.loc[p_mask].iterrows():
                xs += [r["left_x"], r["right_x"], None]
                ys += [r["y"],      r["y"],       None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="FWHM",
                                     line=dict(dash="dot"), hoverinfo="skip"))

    if window and (lo is not None) and (hi is not None):
        fig.update_xaxes(range=[lo, hi])
    fig.update_layout(
        title=title or f"{ycol} with peaks",
        xaxis_title=xcol, yaxis_title=ycol,
        hovermode="x unified",
        legend_title_text="Trace",
        updatemenus=[{
            "type": "buttons", "direction": "right", "x": 1.0, "y": 1.15, "xanchor": "right",
            "buttons": [
                {"label": "Show all", "method": "update",
                 "args": [{"visible": [True]*len(fig.data)}]},
                {"label": "Hide all", "method": "update",
                 "args": [{"visible": ["legendonly"]*len(fig.data)}]},
            ],
        }],
    )
    return fig

def plot_channel_dropdown(df: pd.DataFrame,
                          *,
                          xcol: str = "Wavenumber",
                          channels: Sequence[str] | None = None,
                          preferred: Iterable[str] = ("O1A","O2A","O0A","O1P","O2P"),
                          eps: float = 0.0,
                          window: bool = True,
                          start_hidden: bool = True,
                          title: str | None = None) -> go.Figure:
    """
    Build an interactive Plotly figure with a dropdown to select a single channel.
    If `window=True`, switching the channel also updates the x-range to that
    channel's non-zero window.
    """
    if xcol not in df.columns:
        raise KeyError(f"{xcol!r} not in DataFrame")
    if channels is None:
        channels = list_channels(df, xcol=xcol)
    if not channels:
        raise ValueError("No channel columns found to plot.")

    x = df[xcol].to_numpy()
    fig = go.Figure()

    # Precompute per-channel windows (lo/hi) and masks
    ch_windows: dict[str, tuple[float | None, float | None, np.ndarray]] = {}
    for c in channels:
        if window:
            lo, hi, mask = _nonzero_window(df, xcol, [c], eps=eps)
        else:
            y = df[c].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            lo = hi = None
        ch_windows[c] = (lo, hi, mask)

    # Add traces (one per channel)
    for c in channels:
        lo, hi, mask = ch_windows[c]
        fig.add_trace(go.Scatter(
            x=df.loc[mask, xcol], y=df.loc[mask, c],
            mode="lines", name=c,
            visible=("legendonly" if start_hidden else True),
            hovertemplate=f"{xcol}: %{{x}}<br>{c}: %{{y}}<extra></extra>",
        ))

    # Build dropdown buttons that show exactly one channel (others hidden)
    buttons = []
    for i, c in enumerate(channels):
        vis = ["legendonly"] * len(channels)
        vis[i] = True
        lo, hi, _ = ch_windows[c]
        # Update x-axis range per channel if window available
        args = [{"visible": vis}]
        if window and (lo is not None) and (hi is not None):
            args.append({"xaxis": {"range": [lo, hi]}, "title": {"text": c}})
        else:
            args.append({"title": {"text": c}})
        buttons.append({"label": c, "method": "update", "args": args})

    # Default title: chosen preferred or first channel
    try:
        default_ch = choose_channel(df, preferred=preferred, xcol=xcol)
    except Exception:
        default_ch = channels[0]

    fig.update_layout(
        title=title or f"Channel: {default_ch}",
        xaxis_title=xcol, yaxis_title="Signal",
        hovermode="x unified",
        updatemenus=[{
            "type": "dropdown", "direction": "down",
            "x": 1.0, "y": 1.15, "xanchor": "right",
            "buttons": buttons
        }],
        legend_title_text="Channel"
    )
    return fig


#---Plot position on AFM


def _len_scale(u: str | None) -> float:
    if not u: return 1e-6
    u = str(u).strip().lower().replace("μ", "µ")
    if u in {"m", "meter", "meters"}: return 1.0
    if u in {"µm","um","micrometer","micrometers"}: return 1e-6
    if u in {"nm","nanometer","nanometers"}: return 1e-9
    return 1e-6

def _convert_len(val: float, from_unit: str | None, to_unit: str) -> float:
    return float(val) * (_len_scale(from_unit) / _len_scale(to_unit))

def _extract_scp(spec_header: Mapping[str, Any]) -> tuple[Optional[float], Optional[float], Optional[str]]:
    """Return (X, Y, unit) from spec.header['parameters']['Scanner Center Position'] if available."""
    try:
        params = spec_header.get("parameters", {})
        scp = params.get("Scanner Center Position")
        if isinstance(scp, dict):
            unit = scp.get("unit")
            val = scp.get("value", {})
            if isinstance(val, dict) and "X" in val and "Y" in val:
                return float(val["X"]), float(val["Y"]), unit
            raw = scp.get("raw", {})
            if isinstance(raw, dict) and isinstance(raw.get("values"), (list, tuple)) and len(raw["values"]) >= 2:
                return float(str(raw["values"][0]).replace(",", ".")), float(str(raw["values"][1]).replace(",", ".")), raw.get("unit", unit)
    except Exception:
        pass
    return None, None, None

def plot_afm_with_spectrum_marker(
    img,   # AFMImage: .data (ny,nx), .sx, .sy, .unit_xy, .meta (optional GWY fields)
    spec,  # Spectrum: .header dict with 'parameters' -> 'Scanner Center Position'
    *,
    title: Optional[str] = None,
    marker_color: str = "r",
    marker_size: int = 90,
    marker_symbol: str = "x",
    annotate: bool = True,
    coords: str = "auto",           # "auto" | "pixels" | "absolute"
    pixel_offset: float = 0.5,      # 0.5 marks pixel center; use 0.0 for upper-left corner
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot AFM image and mark the spectrum position.

    coords="pixels": interpret X,Y as pixel indices with origin at the TOP-LEFT of the image data.
                     Conversion to plot coordinates (imshow origin='lower'):
                       x_plot = x0 + (x_px + pixel_offset) * sx
                       y_plot = y1 - (y_px + pixel_offset) * sy

    coords="absolute": interpret X,Y in physical units (unit taken from the header, e.g. µm)
                       and convert to the image's display unit.

    coords="auto": if unit is 'px' (or values look like pixel indices), use "pixels",
                   otherwise use "absolute".
    """
    # --- image geometry ---
    ny, nx = img.data.shape
    unit_xy = getattr(img, "unit_xy", "µm")
    sx = float(img.sx)
    sy = float(img.sy)

    # default extent (no offsets): [0..W] × [0..H]
    x0 = 0.0
    y0 = 0.0
    x1 = nx * sx
    y1 = ny * sy

    # prefer GWY offsets/sizes (meters) if available
    meta = getattr(img, "meta", {}) or {}
    xoff_m, yoff_m = meta.get("xoff_m"), meta.get("yoff_m")
    xreal_m, yreal_m = meta.get("xreal_m"), meta.get("yreal_m")
    if all(v is not None for v in (xoff_m, yoff_m, xreal_m, yreal_m)):
        x0 = _convert_len(float(xoff_m), "m", unit_xy)
        y0 = _convert_len(float(yoff_m), "m", unit_xy)
        x1 = x0 + _convert_len(float(xreal_m), "m", unit_xy)
        y1 = y0 + _convert_len(float(yreal_m), "m", unit_xy)

    extent = (x0, x1, y0, y1)

    # --- read spectrum center ---
    x_val, y_val, unit = _extract_scp(getattr(spec, "header", {}) or {})
    if x_val is None or y_val is None:
        # Graceful message and plot without marker if not present
        if ax is None:
            _, ax = plt.subplots(figsize=(5.6, 5.2))
        im = ax.imshow(img.data, origin="lower", extent=extent, cmap=getattr(img, "cmap", "viridis"))
        cbar = plt.colorbar(im, ax=ax); cbar.set_label(getattr(img, "zlabel", "Value"))
        ax.set_xlabel(f"X [{unit_xy}]"); ax.set_ylabel(f"Y [{unit_xy}]")
        if title: ax.set_title(title)
        ax.text(0.02, 0.98, "No 'Scanner Center Position' in spectrum header",
                transform=ax.transAxes, va="top", ha="left", fontsize=9, color="w",
                bbox=dict(facecolor="0.2", alpha=0.6, edgecolor="none"))
        return ax

    # --- choose coord interpretation ---
    use_pixels = False
    if coords == "pixels":
        use_pixels = True
    elif coords == "absolute":
        use_pixels = False
    else:  # auto
        if isinstance(unit, str) and unit.strip().lower() in {"px", "pixel", "pixels"}:
            use_pixels = True
        else:
            # heuristic: looks like pixel indices if within image bounds (with small slack)
            if 0 <= x_val <= nx + 1 and 0 <= y_val <= ny + 1:
                use_pixels = True

    # --- compute plot coordinates ---
    if use_pixels:
        # pixel coords: (0,0) is TOP-LEFT of array; imshow uses origin="lower"
        # so X is direct; Y must be flipped.
        x_plot = x0 + (x_val + float(pixel_offset)) * sx
        y_plot = y1 - (y_val + float(pixel_offset)) * sy
        label_unit = "px→" + unit_xy
    else:
        # absolute units from header (e.g., µm) → convert
        x_plot = _convert_len(x_val, unit, unit_xy)
        y_plot = _convert_len(y_val, unit, unit_xy)
        label_unit = unit_xy

    # --- plot ---
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 5.2))
    im = ax.imshow(img.data, origin="lower", extent=extent, cmap=getattr(img, "cmap", "viridis"))
    cbar = plt.colorbar(im, ax=ax); cbar.set_label(getattr(img, "zlabel", "Value"))
    ax.set_xlabel(f"X [{unit_xy}]"); ax.set_ylabel(f"Y [{unit_xy}]")
    if title: ax.set_title(title)

    ax.scatter([x_plot], [y_plot], s=marker_size, c=marker_color, marker=marker_symbol, zorder=3)
    if annotate:
        ax.annotate(f"({x_plot:.2f}, {y_plot:.2f}) {label_unit}",
                    (x_plot, y_plot), textcoords="offset points", xytext=(8, 8),
                    fontsize=9, color=marker_color,
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    # optional diagnostic if outside
    if not (min(x0, x1) <= x_plot <= max(x0, x1) and min(y0, y1) <= y_plot <= max(y0, y1)):
        ax.text(0.02, 0.92, "Marker outside extent (check coords/units)",
                transform=ax.transAxes, va="top", ha="left", fontsize=9, color="yellow",
                bbox=dict(facecolor="0.2", alpha=0.6, edgecolor="none"))

    return ax

