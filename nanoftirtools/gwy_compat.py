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


"""
Compatibility helpers for using the 'gwyfile' package with NumPy >= 1.25.
Call patch_numpy_for_gwyfile() BEFORE importing 'gwyfile'.
"""
from __future__ import annotations
import numpy as _np

_patched = False
_old_fromstring = _np.fromstring

def patch_numpy_for_gwyfile() -> None:
    """
    Monkey-patch numpy.fromstring to emulate the old binary mode that
    'gwyfile' relies on (bytes input with sep=''). Safe no-op if already patched.
    """
    global _patched, _old_fromstring
    if _patched:
        return

    def _fromstring_compat(s, dtype=float, count=-1, sep=''):
        if sep == '' and isinstance(s, (bytes, bytearray, memoryview)):
            return _np.frombuffer(s, dtype=dtype, count=count)
        return _old_fromstring(s, dtype=dtype, count=count, sep=sep)

    _np.fromstring = _fromstring_compat  # type: ignore[assignment]
    _patched = True
