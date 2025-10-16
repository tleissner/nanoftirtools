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
import re
import pandas as pd
import numpy as np

NBSP = "\xa0"
KEY_LABELS_RE = re.compile(r"^(?P<key>.+?)\s*\((?P<labels>[^)]+)\)\s*$")
UNIT_RE = re.compile(r"^\[(?P<unit>.+?)\]$")

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace(NBSP, " ").strip())

def _parse_unit(tok: str | None):
    if not tok:
        return None
    m = UNIT_RE.match(tok)
    return m.group("unit") if m else None

def _num(tok: str):
    if tok is None:
        return None
    t = tok.replace(NBSP, "").strip()
    if "," in t and "." in t:
        t = t.replace(",", "")
    elif "," in t and t.replace(",", "").isdigit():
        t = t.replace(",", "")
    try:
        if re.fullmatch(r"[+-]?\d+", t):
            return int(t)
        return float(t)
    except ValueError:
        return tok

def _split_fields(line: str) -> list[str]:
    if "\t" in line:
        parts = line.split("\t")
    else:
        parts = re.split(r"\s{2,}", line)
    parts = [_clean(p) for p in parts]
    return [p for p in parts if p not in ("", "—", "-")]

def parse_neaspec_header(lines: list[str]) -> dict:
    meta: dict = {}
    parameters: dict = {}

    for raw in lines:
        if not raw.lstrip().startswith("#"):
            continue
        line = raw.lstrip()[1:].strip()
        if not line:
            continue

        parts = _split_fields(line)

        if ":" not in parts[0] and "www." in parts[0]:
            meta["Source"] = _clean(parts[0])
            continue

        if ":" in parts[0]:
            key, after = parts[0].split(":", 1)
            key = _clean(key)
            values = []
            if after.strip():
                values.append(_clean(after))
            values.extend(parts[1:])
            unit = _parse_unit(values[0]) if values else None
            if unit:
                values = values[1:]
        else:
            key = parts[0]
            unit = _parse_unit(parts[1]) if len(parts) > 1 else None
            values = parts[2:] if unit is not None else parts[1:]

        labels = None
        m = KEY_LABELS_RE.match(key)
        if m:
            key = _clean(m.group("key"))
            labels = [t.strip() for t in m.group("labels").split(",")]

        parsed_vals = [_num(v) for v in values]

        if labels and len(labels) == len(parsed_vals):
            value = {labels[i]: parsed_vals[i] for i in range(len(labels))}
        elif len(parsed_vals) == 1:
            value = parsed_vals[0]
        else:
            while parsed_vals and parsed_vals[-1] in ("", None):
                parsed_vals.pop()
            value = parsed_vals

        entry = {"unit": unit, "value": value, "raw": {"key": key, "unit": unit, "values": values}}

        if key in {"Scan", "Project", "Description", "Date", "Reference",
                   "Laser Source", "Detector", "Version"}:
            meta[key] = value
        else:
            parameters[key] = entry

    return {"meta": meta, "parameters": parameters}


@dataclass
class Spectrum:
    df: pd.DataFrame
    header: dict

def load_neaspec_spectrum(path: str | Path) -> Spectrum:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    header_lines = [ln for ln in lines if ln.lstrip().startswith("#")]
    first_table_idx = next(i for i, ln in enumerate(lines) if not ln.lstrip().startswith("#"))

    header = parse_neaspec_header(header_lines)
    header["meta"]["_filename"] = path.name

    df = pd.read_csv(
        path,
        sep="\t",
        engine="python",
        comment="#",
        skip_blank_lines=True
    )
    df.columns = [c.strip() for c in df.columns]
    return Spectrum(df=df, header=header)



