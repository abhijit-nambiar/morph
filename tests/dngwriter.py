"""Tiny uncompressed-DNG writer so RAW tests run without shipping multi-MB camera files."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# XYZ(D65) -> linear sRGB: with this ColorMatrix the "camera" is an sRGB-primaries sensor.
_XYZ_TO_SRGB = [3.2406, -1.5372, -0.4986, -0.9689, 1.8758, 0.0415, 0.0557, -0.2040, 1.0570]
_TYPES = {"B": (1, "B"), "A": (2, "B"), "H": (3, "H"), "L": (4, "I"), "R": (5, "II"), "SR": (10, "ii")}


def _rational(v, signed=False):
    d = 10000
    return (round(v * d), d)


def write_dng(path: str | Path, rgb_linear: np.ndarray, neutral=(0.5, 1.0, 0.7), date="2024:05:01 10:00:00",
              exposure=(1, 250), iso=400, make="Nikon", model="Z 6 test", flip=1) -> Path:
    """Mosaic ``rgb_linear`` (HxWx3, 0..1 scene-linear, even H/W) into an RGGB DNG.

    ``neutral`` is AsShotNeutral (camera RGB of a grey), so a correct decoder using camera WB
    gets the input colours back.
    """
    h, w = rgb_linear.shape[:2]
    cam = np.clip(rgb_linear * np.asarray(neutral, np.float32), 0, 1)
    bayer = np.empty((h, w), np.uint16)
    s = 64000
    bayer[0::2, 0::2] = cam[0::2, 0::2, 0] * s
    bayer[0::2, 1::2] = cam[0::2, 1::2, 1] * s
    bayer[1::2, 0::2] = cam[1::2, 0::2, 1] * s
    bayer[1::2, 1::2] = cam[1::2, 1::2, 2] * s
    data = bayer.astype("<u2").tobytes()

    def asc(t):
        return ("A", list(t.encode() + b"\0"))

    exif = {33434: ("R", [exposure]), 34855: ("H", [iso]), 36867: asc(date), 33437: ("R", [(28, 10)])}
    ifd0 = {
        254: ("L", [0]), 256: ("L", [w]), 257: ("L", [h]), 258: ("H", [16]), 259: ("H", [1]), 262: ("H", [32803]),
        271: asc(make), 272: asc(model), 273: ("L", [0]), 274: ("H", [flip]), 277: ("H", [1]), 278: ("L", [h]),
        279: ("L", [len(data)]), 284: ("H", [1]), 306: asc(date), 33421: ("H", [2, 2]), 33422: ("B", [0, 1, 1, 2]),
        34665: ("L", [0]), 50706: ("B", [1, 4, 0, 0]), 50708: asc(f"{make} {model}"), 50714: ("L", [0]),
        50717: ("L", [s]), 50721: ("SR", [_rational(v) for v in _XYZ_TO_SRGB]),
        50728: ("R", [_rational(v) for v in neutral]), 50778: ("H", [21]),
    }

    def ifd_bytes(tags, offset, patch=None):
        n = len(tags)
        head = struct.pack("<H", n)
        extra = b""
        entries = b""
        extra_off = offset + 2 + 12 * n + 4
        for tag in sorted(tags):
            typ, vals = tags[tag]
            code, fmt = _TYPES[typ]
            if typ in ("R", "SR"):
                payload = b"".join(struct.pack("<" + fmt, *v) for v in vals)
            else:
                payload = struct.pack("<" + fmt * len(vals), *vals)
            count = len(vals)
            if len(payload) <= 4:
                entries += struct.pack("<HHI", tag, code, count) + payload.ljust(4, b"\0")
            else:
                if len(extra) % 2:
                    extra += b"\0"
                entries += struct.pack("<HHII", tag, code, count, extra_off + len(extra))
                extra += payload
        return head + entries + struct.pack("<I", 0) + extra

    # layout: header | ifd0 | exif ifd | pixel data
    size0 = len(ifd_bytes(ifd0, 8))
    exif_off = 8 + size0 + (size0 % 2)
    size1 = len(ifd_bytes(exif, exif_off))
    data_off = exif_off + size1 + (size1 % 2)
    ifd0[273] = ("L", [data_off])
    ifd0[34665] = ("L", [exif_off])
    b0 = ifd_bytes(ifd0, 8)
    b1 = ifd_bytes(exif, exif_off)
    out = b"II*\0" + struct.pack("<I", 8) + b0 + b"\0" * (size0 % 2) + b1 + b"\0" * (size1 % 2) + data
    Path(path).write_bytes(out)
    return Path(path)
