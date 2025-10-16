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


# nanoftirtools/spectrum_utils.py
from __future__ import annotations
from pathlib import Path
import re
from typing import Iterable, Pattern

# Accept "NF S.txt" where the space might be normal space or NBSP (U+00A0),
# and be tolerant to underscores/dashes and multiple spaces:
NF_S_PATTERN: str = r"(?i)\bNF[\s\u00A0_-]*S\.txt$"
NF_S_RE: Pattern[str] = re.compile(NF_S_PATTERN)

def list_spectra(folder: str | Path,
                 *,
                 recursive: bool = False,
                 regex: Pattern[str] = NF_S_RE) -> list[Path]:
    """
    Return *.txt files in `folder` matching `regex` (default: NF[space/NBSP/_/-]*S.txt).
    """
    folder = Path(folder)
    it = folder.rglob("*.txt") if recursive else folder.glob("*.txt")
    return [p for p in it if regex.search(p.name)]
