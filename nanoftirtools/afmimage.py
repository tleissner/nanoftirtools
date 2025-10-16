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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import numpy as np
import re
import matplotlib.pyplot as plt

@dataclass
class AFMImage:
    data: np.ndarray
    meta: dict = field(default_factory=dict)
    sx: float = 1.0
    sy: float = 1.0
    unit_xy: str = ""
    zlabel: str = "Value"
    cmap: str = "viridis"

    # ---------- helpers ----------
    @staticmethod
    def _unitstr(u) -> str:
        if hasattr(u, "unitstr"):
            return u.unitstr or ""
        if isinstance(u, str):
            return u
        return ""

    @staticmethod
    def _convert_scale(s_m: float | None, target: str = "µm") -> tuple[float, str]:
        if s_m is None:
            return 1.0, target
        if target == "µm":
            return s_m * 1e6, "µm"
        if target == "nm":
            return s_m * 1e9, "nm"
        if target in ("m", "", None):
            return s_m, ("m" if target == "m" else "")
        return s_m, target

    @staticmethod
    def _parse_channel_title(title: str) -> tuple[str, str | None]:
        TITLE_RE = re.compile(
            r"""^\s*
                (?P<prefix>R-)?
                (?:
                    (?P<fam>[MO])\s*[-_]? (?P<order>\d+)\s*(?P<ap>[AP])
                  | (?P<z>Z)
                )
                (?:\s+(?P<variant>raw|C))?
                \s*$""",
            re.IGNORECASE | re.VERBOSE
        )
        m = TITLE_RE.match(title or "")
        if not m:
            return (title, None)
        prefix = (m.group("prefix") or "").upper()
        if m.group("z"):
            ch = f"{prefix}Z".strip()
        else:
            fam  = m.group("fam").upper()
            order= int(m.group("order"))
            ap   = m.group("ap").upper()
            ch = f"{prefix}{fam}{order}{ap}".strip()
        variant = m.group("variant")
        return ch, (variant.lower() if variant else None)

    @property
    def extent(self):
        ny, nx = self.data.shape
        return [0, nx * self.sx, 0, ny * self.sy]

    def plot(self, *, title: str | None = None, equal_axes: bool = True):
        fig, ax = plt.subplots()
        im = ax.imshow(
            self.data,
            origin="lower",
            extent=self.extent,
            cmap=self.cmap,
            aspect=("equal" if equal_axes else "auto")
        )
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(self.zlabel)
        ax.set_xlabel(f"X ({self.unit_xy})" if self.unit_xy else "X")
        ax.set_ylabel(f"Y ({self.unit_xy})" if self.unit_xy else "Y")
        if title:
            ax.set_title(title)
        fig.tight_layout()
        return fig, ax
    
    # ---------- constructors ----------
    @classmethod
    def from_gwy(cls,
                 path: str | Path,
                 *,
                 unit_xy: str = "µm",
                 zlabel_default: str = "Value",
                 cmap: str = "viridis",
                 key_policy: str = "title"   # 'title' | 'channel' | 'channel+variant'
                 ) -> dict[str, "AFMImage"]:
        """
        Load channels from a .gwy file via gwyfile.util.get_datafields().
        Keys default to the ORIGINAL TITLES (key_policy='title').
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        try:
            import gwyfile  # type: ignore
        except Exception as e:
            raise ImportError("pip install gwyfile") from e

        obj = gwyfile.load(str(path))
        try:
            dfields = gwyfile.util.get_datafields(obj)  # {title: GwyDataField}
        except Exception:
            dfields = {}

        out: dict[str, AFMImage] = {}
        collisions: dict[str, int] = {}

        for idx, (title, df) in enumerate(dfields.items()):
            arr = np.asarray(df.data)
            if arr.ndim > 2: arr = np.squeeze(arr)
            if arr.ndim != 2: continue

            yres, xres = arr.shape
            xreal = getattr(df, "xreal", None)
            yreal = getattr(df, "yreal", None)

            sx_m = (float(xreal)/xres) if xreal else None
            sy_m = (float(yreal)/yres) if yreal else None
            sx, unit_eff = cls._convert_scale(sx_m, unit_xy)
            sy, _        = cls._convert_scale(sy_m, unit_xy)

            unit_x = cls._unitstr(getattr(df, "si_unit_x", None)) or "m"
            unit_y = cls._unitstr(getattr(df, "si_unit_y", None)) or "m"
            unit_z = cls._unitstr(getattr(df, "si_unit_z", None)) or ""

            ch, variant = cls._parse_channel_title(title)

            # Choose dict key
            if key_policy == "title":
                key = title
            elif key_policy == "channel+variant":
                key = f"{ch} ({variant})" if variant else ch
            else:  # 'channel'
                key = ch or title

            if key in out:
                collisions[key] = collisions.get(key, 1) + 1
                key = f"{key}#{collisions[key]}"

            meta = {
                "filename": path.name, "path": str(path), "source": "gwy",
                "index": idx, "title": title,
                "channel": title,           # keep original title as canonical
                "short_label": ch, "variant": variant,
                "xres": xres, "yres": yres, "xreal_m": xreal, "yreal_m": yreal,
                "unit_x": unit_x, "unit_y": unit_y, "unit_z": unit_z,
            }

            out[key] = cls(arr, meta, sx, sy, unit_eff, (unit_z or zlabel_default), cmap)

        return out

    @classmethod
    def from_gsf_folder(cls,
                        folder: str | Path,
                        *,
                        sx: float = 1.0,
                        sy: float = 1.0,
                        unit_xy: str = "",
                        zlabel: str = "Value",
                        cmap: str = "viridis",
                        extra_meta: dict | None = None,
                        recursive: bool = False,
                        key_policy: str = "title"   # 'title' | 'channel' | 'stem'
                        ) -> dict[str, "AFMImage"]:
        """
        Load all *.gsf files in folder and return dict with ORIGINAL TITLES as keys by default.
        """
        folder = Path(folder)
        pattern = "**/*.gsf" if recursive else "*.gsf"
        files = sorted(folder.glob(pattern))

        try:
            from gsffile import read_gsf  # expects your environment's reader
        except Exception as e:
            raise ImportError("Could not import 'read_gsf' from 'gsffile'. Ensure it is installed or available.") from e

        out: dict[str, AFMImage] = {}
        collisions: dict[str, int] = {}

        for p in files:
            # read_gsf can return array; (array, meta); or dict
            try:
                res = read_gsf(str(p))
            except Exception:
                continue

            if isinstance(res, np.ndarray):
                arr, hdr = res, {}
            elif isinstance(res, (list, tuple)) and len(res) >= 1:
                arr = res[0]
                hdr = res[1] if (len(res) >= 2 and isinstance(res[1], dict)) else {}
            elif isinstance(res, dict):
                arr = None
                for k in ("data", "array", "Z", "image"):
                    if k in res:
                        arr = res[k]; break
                if arr is None: continue
                hdr = {k: v for k, v in res.items() if k not in ("data", "array", "Z", "image")}
            else:
                continue

            arr = np.asarray(arr)
            if arr.ndim > 2: arr = np.squeeze(arr)
            if arr.ndim != 2: continue

            # Scaling from header if present (meters -> unit)
            xreal = hdr.get("XReal"); yreal = hdr.get("YReal")
            xres  = hdr.get("XRes") or hdr.get("YResIncomplete") or hdr.get("Neaspec_XRes")
            yres  = hdr.get("YRes") or hdr.get("YResIncomplete") or hdr.get("Neaspec_YRes")
            try:
                xres_i = int(xres) if xres is not None else arr.shape[1]
                yres_i = int(yres) if yres is not None else arr.shape[0]
            except Exception:
                xres_i, yres_i = arr.shape[1], arr.shape[0]

            def _scale_from_hdr(_real, _res):
                if _real is None or _res in (None, 0): return None
                try:
                    return float(_real) / int(_res)
                except Exception:
                    return None

            sx_m = _scale_from_hdr(xreal, xres_i)
            sy_m = _scale_from_hdr(yreal, yres_i)

            sx_eff, unit_eff = cls._convert_scale(sx_m, unit_xy)
            sy_eff, _        = cls._convert_scale(sy_m, unit_xy)

            # Pick original title if available; fall back to stem
            title = (
                hdr.get("Title") or hdr.get("title") or hdr.get("ChannelTitle") or
                hdr.get("Name") or hdr.get("Channel") or p.stem
            )

            ch, variant = cls._parse_channel_title(str(title))

            if key_policy == "title":
                key = str(title)
            elif key_policy == "channel":
                key = ch or p.stem
            else:  # 'stem'
                key = p.stem

            if key in out:
                collisions[key] = collisions.get(key, 1) + 1
                key = f"{key}#{collisions[key]}"

            meta = {
                "filename": p.name, "path": str(p), "folder": str(folder), "source": "gsf",
                "title": str(title), "channel": str(title), "short_label": ch, "variant": variant,
                "xreal_m": xreal, "yreal_m": yreal, "xres": xres_i, "yres": yres_i,
            }
            if extra_meta:
                meta.update(extra_meta)
            if isinstance(hdr, dict):
                meta.update({k: v for k, v in hdr.items() if k not in ("data","array","Z","image")})

            out[key] = cls(arr, meta, sx_eff, sy_eff, unit_eff, zlabel, cmap)

        return out

    @classmethod
    def from_gsf_file(cls,
                      path: str | Path,
                      *,
                      unit_xy: str = "",
                      zlabel: str = "Value",
                      cmap: str = "viridis",
                      key_policy: str = "title"   # 'title' | 'channel' | 'stem'
                      ) -> dict[str, "AFMImage"]:
        """
        Load a single .gsf file and return {key: AFMImage}.
        Keys default to the ORIGINAL title when present.
        """
        from pathlib import Path
        import numpy as np
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
    
        try:
            from gsffile import read_gsf
        except Exception as e:
            raise ImportError("Could not import 'read_gsf' from 'gsffile'.") from e
    
        res = read_gsf(str(path))
    
        # Coerce result to (array, header)
        if isinstance(res, np.ndarray):
            arr, hdr = res, {}
        elif isinstance(res, (list, tuple)) and len(res) >= 1:
            arr = res[0]
            hdr = res[1] if (len(res) >= 2 and isinstance(res[1], dict)) else {}
        elif isinstance(res, dict):
            arr = None
            for k in ("data", "array", "Z", "image"):
                if k in res:
                    arr = res[k]; break
            if arr is None:
                raise ValueError("GSF dict did not contain data under keys: data/array/Z/image")
            hdr = {k: v for k, v in res.items() if k not in ("data","array","Z","image")}
        else:
            raise ValueError("Unsupported return type from read_gsf")
    
        arr = np.asarray(arr)
        if arr.ndim > 2: arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D data, got {arr.shape}")
    
        # Scaling
        xreal = hdr.get("XReal"); yreal = hdr.get("YReal")
        xres  = hdr.get("XRes") or hdr.get("YResIncomplete") or hdr.get("Neaspec_XRes")
        yres  = hdr.get("YRes") or hdr.get("YResIncomplete") or hdr.get("Neaspec_YRes")
        xres_i = int(xres) if xres is not None else arr.shape[1]
        yres_i = int(yres) if yres is not None else arr.shape[0]
    
        def _scale_from_hdr(_real, _res):
            if _real is None or _res in (None, 0): return None
            try: return float(_real) / int(_res)   # meters/px
            except Exception: return None
    
        sx_m = _scale_from_hdr(xreal, xres_i)
        sy_m = _scale_from_hdr(yreal, yres_i)
    
        sx_eff, unit_eff = cls._convert_scale(sx_m, unit_xy)
        sy_eff, _        = cls._convert_scale(sy_m, unit_xy)
    
        # Title & key
        title = (hdr.get("Title") or hdr.get("title") or hdr.get("ChannelTitle")
                 or hdr.get("Name") or hdr.get("Channel") or path.stem)
        ch, variant = cls._parse_channel_title(str(title))
    
        if key_policy == "title":
            key = str(title)
        elif key_policy == "channel":
            key = ch or path.stem
        else:
            key = path.stem
    
        meta = {
            "filename": path.name, "path": str(path), "folder": str(path.parent), "source": "gsf",
            "title": str(title), "channel": str(title), "short_label": ch, "variant": variant,
            "xreal_m": xreal, "yreal_m": yreal, "xres": xres_i, "yres": yres_i,
        }
        if isinstance(hdr, dict):
            meta.update({k: v for k, v in hdr.items() if k not in ("data","array","Z","image")})
    
        return {key: cls(arr, meta, sx_eff, sy_eff, unit_eff, zlabel, cmap)}


    @classmethod
    def load_auto(cls, path: str | Path, **kwargs) -> dict[str, "AFMImage"]:
        """
        Smart loader:
          - if path is a .gwy file -> from_gwy
          - if path is a .gsf file -> from_gsf_file
          - if path is a folder:
              * if it contains any .gwy -> load the first .gwy (or choose one)
              * else load all .gsf in the folder via from_gsf_folder
        """
        p = Path(path)
        if p.is_file():
            if p.suffix.lower() == ".gwy":
                return cls.from_gwy(p, **kwargs)
            if p.suffix.lower() == ".gsf":
                return cls.from_gsf_file(p, **kwargs)
            raise ValueError(f"Unsupported file type: {p.suffix}")
    
        if p.is_dir():
            gwys = sorted(p.glob("*.gwy"))
            if gwys:
                return cls.from_gwy(gwys[0], **kwargs)
            return cls.from_gsf_folder(p, **kwargs)
    
        raise FileNotFoundError(p)

