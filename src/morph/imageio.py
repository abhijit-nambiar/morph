"""Image loading and export that preserves geometry, EXIF and ICC metadata.

Camera RAW files (Nikon NEF/NRW and the other formats LibRaw reads) are decoded with ``rawpy``
using the camera's white balance and a fixed, camera-like base rendering (see :func:`render_raw`),
so frames shot at the same exposure render identically across an album.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# Formats exports can keep (with EXIF/ICC); other readable rasters export as JPEG.
WRITABLE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
RASTER_SUFFIXES = WRITABLE_SUFFIXES | {".bmp", ".avif", ".heic", ".heif"}
RAW_SUFFIXES = {
    ".nef", ".nrw",  # Nikon
    ".cr2", ".cr3", ".crw", ".arw", ".srf", ".sr2", ".raf", ".orf", ".rw2", ".pef", ".dng",
    ".srw", ".3fr", ".fff", ".iiq", ".erf", ".kdc", ".dcr", ".mrw", ".mos", ".x3f", ".rwl", ".raw",
}
IMAGE_SUFFIXES = RASTER_SUFFIXES | RAW_SUFFIXES
PAIR_POLICIES = ("raw", "jpeg", "both")

_EXIF_IFD = 0x8769
_TAG_DATETIME_ORIGINAL = 0x9003
_TAG_DATETIME = 0x0132
_TAG_EXPOSURE_TIME = 0x829A
_TAG_FNUMBER = 0x829D
_TAG_ISO = 0x8827
_TAG_ORIENTATION = 0x0112
# TIFF layout tags that Pillow's getexif() returns for TIFF sources; copying them into another file's
# EXIF (strip offsets, bit depth, compression, ...) corrupts the written image.
_TIFF_LAYOUT_TAGS = {254, 255, 256, 257, 258, 259, 262, 266, 273, 277, 278, 279, 280, 281, 284, 317, 320,
                     322, 323, 324, 325, 338, 339, 340, 341, 347, 530, 531, 532}


@dataclass
class PhotoInfo:
    path: Path
    name: str
    width: int
    height: int
    captured: datetime | None = None
    exposure_time: float | None = None
    f_number: float | None = None
    iso: float | None = None
    sha256: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def exposure_value(self) -> float | None:
        """Scene-referred EV100 implied by the camera settings (higher = brighter scene)."""
        if not (self.exposure_time and self.f_number and self.iso):
            return None
        return float(np.log2(self.f_number**2 / self.exposure_time) - np.log2(self.iso / 100.0))

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "captured": self.captured.isoformat() if self.captured else None,
            "exposure_time": self.exposure_time,
            "f_number": self.f_number,
            "iso": self.iso,
            "exposure_value": self.exposure_value,
            "sha256": self.sha256,
        }


def is_raw(path: str | Path) -> bool:
    return Path(path).suffix.lower() in RAW_SUFFIXES


def list_images(folder: str | Path, pairs: str = "raw") -> list[Path]:
    """Images in ``folder`` (RAW and raster mixed).

    When a RAW and a JPEG/TIFF share a stem (camera RAW+JPEG mode), ``pairs`` picks one:
    ``"raw"`` (default: more latitude, and every RAW renders with the same base look),
    ``"jpeg"`` (the camera's own rendering), or ``"both"``.
    """
    if pairs not in PAIR_POLICIES:
        raise ValueError(f"pairs must be one of {PAIR_POLICIES}")
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"album folder not found: {folder}")
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".") and p.is_file())
    if pairs != "both":
        raw_stems = {p.stem for p in files if is_raw(p)}
        raster_stems = {p.stem for p in files if not is_raw(p)}
        paired = raw_stems & raster_stems
        drop_raw = pairs == "jpeg"
        files = [p for p in files if p.stem not in paired or is_raw(p) != drop_raw]
    if not files:
        raise ValueError(f"no images found in {folder}")
    return files


def raw_pairs(folder: str | Path) -> dict[str, Path]:
    """``{raw stem: raster sibling}`` for RAW+JPEG pairs, e.g. to score the camera JPEG instead of decoding the RAW."""
    files = [p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")]
    rasters = {p.stem: p for p in sorted(files) if p.suffix.lower() in (".jpg", ".jpeg")}
    return {p.stem: rasters[p.stem] for p in files if is_raw(p) and p.stem in rasters}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rational(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def read_info(path: str | Path, with_hash: bool = True) -> PhotoInfo:
    path = Path(path)
    if is_raw(path):
        info = _read_raw_info(path)
        if with_hash:
            info.sha256 = sha256_file(path)
        return info
    _register_heif(path)
    with Image.open(path) as im:
        exif = im.getexif()
        size = im.size
        orientation = exif.get(_TAG_ORIENTATION, 1)
        if orientation in (5, 6, 7, 8):
            size = (size[1], size[0])
        sub = exif.get_ifd(_EXIF_IFD) if exif else {}
        stamp = sub.get(_TAG_DATETIME_ORIGINAL) or exif.get(_TAG_DATETIME)
        captured = None
        if stamp:
            try:
                captured = datetime.strptime(str(stamp).strip("\x00 "), "%Y:%m:%d %H:%M:%S")
            except ValueError:
                captured = None
        iso = sub.get(_TAG_ISO)
        if isinstance(iso, tuple):
            iso = iso[0] if iso else None
        info = PhotoInfo(
            path=path,
            name=path.name,
            width=size[0],
            height=size[1],
            captured=captured,
            exposure_time=_rational(sub.get(_TAG_EXPOSURE_TIME)),
            f_number=_rational(sub.get(_TAG_FNUMBER)),
            iso=_rational(iso),
        )
    if with_hash:
        info.sha256 = sha256_file(path)
    return info


def load_rgb(path: str | Path, max_side: int | None = None) -> np.ndarray:
    """Load an oriented RGB image as float32 in [0, 1], optionally downscaled.

    RAW files always go through :func:`render_raw`, at every size, so analysis copies and
    full-resolution exports share exactly the same base rendering.
    """
    if is_raw(path):
        return render_raw(path, max_side)
    _register_heif(path)
    with Image.open(path) as im:
        if _high_bit_depth(im):
            img = _load_high_bit_depth(path, im, max_side)
            if img is not None:
                return img
        if max_side:
            im.draft("RGB", (max_side, max_side))
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max_side and max(im.size) > max_side:
            scale = max_side / max(im.size)
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
        return np.asarray(im, dtype=np.float32) / 255.0


def _register_heif(path: str | Path) -> None:
    if Path(path).suffix.lower() in (".heic", ".heif"):
        try:
            from pillow_heif import register_heif_opener
        except ImportError as e:
            raise RuntimeError(f"{Path(path).name}: HEIC/HEIF needs the pillow-heif package (pip install pillow-heif)") from e
        register_heif_opener()


def _high_bit_depth(im: Image.Image) -> bool:
    """16-bit PNG/TIFF: Pillow would truncate RGB to 8 bits and clip greyscale ``I;16`` to white."""
    if im.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N", "F"):
        return True
    raw = im.tile[0][3] if im.tile else None  # decoder args; a plain tuple before Pillow 11
    raw = raw[0] if isinstance(raw, tuple) else raw
    return isinstance(raw, str) and ";16" in raw


def _load_high_bit_depth(path: str | Path, im: Image.Image, max_side: int | None) -> np.ndarray | None:
    """Full-precision decode of 16-bit PNG/TIFF; ``None`` when only an 8-bit decode is possible."""
    import cv2

    orient = im.getexif().get(_TAG_ORIENTATION, 1)  # read before decoding: Pillow drops TIFF tags on load
    if im.mode.startswith(("I", "F")):  # greyscale: Pillow hands over the raw values
        arr = np.asarray(im)
        if im.format == "TIFF":
            orient = 1  # Pillow's TIFF reader applies the orientation tag itself
        scale = 65535.0 if im.mode.startswith("I;16") else None
    else:  # 16-bit RGB: Pillow truncates to 8 bits, so decode with OpenCV and orient from Pillow's tag below
        arr = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if arr is None:
            return None
        arr = arr[..., ::-1] if arr.ndim == 3 else arr  # BGR -> RGB
        scale = {np.dtype(np.uint16): 65535.0, np.dtype(np.uint8): 255.0}.get(arr.dtype)
    img = arr.astype(np.float32) / scale if scale else np.clip(arr.astype(np.float32), 0, None)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, -1)
    if scale is None and img.max() > 1.0:  # float / 32-bit integer data: normalise to the image's own range
        img = img / img.max()
    img = {2: lambda a: a[:, ::-1], 3: lambda a: a[::-1, ::-1], 4: lambda a: a[::-1], 5: lambda a: np.rot90(a, 1)[::-1],
           6: lambda a: np.rot90(a, -1), 7: lambda a: np.rot90(a, -1)[::-1], 8: lambda a: np.rot90(a, 1)}.get(orient, lambda a: a)(img)
    img = np.ascontiguousarray(np.clip(img, 0.0, 1.0), dtype=np.float32)
    if max_side and max(img.shape[:2]) > max_side:
        img = resize(img, fit_size(img.shape[1], img.shape[0], max_side))
    return img


def load_rgb_bytes(data: bytes, filename: str = "upload.png") -> np.ndarray:
    """:func:`load_rgb` for an uploaded file (any supported format, decided by the file name's extension)."""
    import tempfile

    suffix = Path(filename).suffix.lower()
    suffix = suffix if suffix in IMAGE_SUFFIXES else ".png"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / f"upload{suffix}"
        f.write_bytes(data)
        return load_rgb(f)


def resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize float image to (width, height)."""
    import cv2

    w, h = size
    if img.shape[1] == w and img.shape[0] == h:
        return img
    interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)


def fit_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    scale = min(1.0, max_side / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def _quality_for_size(pil: Image.Image, fmt: str, kwargs: dict, quality: int, max_bytes: int) -> dict:
    """Highest quality (<= ``quality``) whose file fits ``max_bytes``; if none does, the smallest file we can make.

    Resolution is never reduced. Below quality 85, JPEG chroma is 4:2:0 as Lightroom does.
    """
    def size(q: int) -> tuple[int, dict]:
        kw = {**kwargs, "quality": q}
        if fmt == "JPEG":
            kw["subsampling"] = 0 if q >= 85 else 2
            kw.pop("optimize", None)
        buf = io.BytesIO()
        pil.save(buf, fmt, **{k: v for k, v in kw.items() if k not in ("exif", "icc_profile", "xmp")})
        return buf.tell() + len(kwargs.get("exif", b"")) + len(kwargs.get("icc_profile", b"") or b"") + len(kwargs.get("xmp", b"") or b""), kw

    lo, hi = 10, quality
    n, best = size(hi)
    if n <= max_bytes:
        return best
    n, kw = size(lo)
    if n > max_bytes:
        return kw
    best = kw
    while hi - lo > 1:
        mid = (lo + hi) // 2
        n, kw = size(mid)
        lo, hi, best = (mid, hi, kw) if n <= max_bytes else (lo, mid, best)
    return best


_TIFF_ASCII_TAGS = (270, 271, 272, 305, 306, 315, 33432)  # description, make, model, software, datetime, artist, copyright


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    import struct
    import zlib

    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _save_16bit(img: np.ndarray, out: Path, icc: bytes | None, exif: Image.Exif | None, xmp) -> Path:
    """16-bit RGB PNG/TIFF (Pillow can only write 8-bit RGB), keeping ICC, XMP and EXIF (TIFF: the baseline tags)."""
    arr = np.rint(np.clip(img, 0.0, 1.0) * 65535.0).astype(np.uint16)
    xmp = xmp.encode() if isinstance(xmp, str) else xmp
    if out.suffix.lower() == ".png":
        import zlib

        import cv2

        ok, buf = cv2.imencode(".png", np.ascontiguousarray(arr[..., ::-1]))
        if not ok:
            raise OSError(f"could not encode {out.name}")
        data = buf.tobytes()
        extra = b""
        if icc:
            extra += _png_chunk(b"iCCP", b"ICC Profile\0\0" + zlib.compress(icc))
        if exif:
            raw = exif.tobytes()
            extra += _png_chunk(b"eXIf", raw[6:] if raw.startswith(b"Exif\0\0") else raw)
        if xmp:
            extra += _png_chunk(b"iTXt", b"XML:com.adobe.xmp\0\0\0\0\0" + xmp)
        ihdr_end = 8 + 4 + 4 + 13 + 4
        out.write_bytes(data[:ihdr_end] + extra + data[ihdr_end:])
        return out
    import tifffile

    tags = []
    if icc:
        tags.append((34675, "B", len(icc), icc, True))
    if xmp:
        tags.append((700, "B", len(xmp), xmp, True))
    base = {t: exif.get(t) for t in _TIFF_ASCII_TAGS} if exif else {}
    if exif and not base.get(306):
        base[306] = exif.get_ifd(_EXIF_IFD).get(_TAG_DATETIME_ORIGINAL)
    for t, v in base.items():
        if isinstance(v, str) and v.strip("\0 ") and t != 305:
            tags.append((t, "s", 0, v.strip("\0 "), True))
    subs = {t: dict(exif.get_ifd(t)) for t in (_EXIF_IFD, _GPS_IFD)} if exif else {}
    subs = {t: d for t, d in subs.items() if d}
    # tifffile refuses the EXIF/GPS pointer tags: write placeholders one tag number lower (keeps the IFD
    # sorted), then retag them and fill in the offsets once the sub-IFDs are appended
    tags += [(t - 1, "I", 1, 0, True) for t in subs]
    tifffile.imwrite(out, arr, photometric="rgb", compression="zlib", extratags=tags, metadata=None, software="Morph",
                     byteorder="<")
    if subs:
        _append_sub_ifds(out, subs)
    return out


def _append_sub_ifds(path: Path, subs: dict[int, dict]) -> None:
    """Write EXIF/GPS sub-IFDs (which tifffile cannot) at the end of a TIFF and point IFD0's tags at them."""
    import struct

    import tifffile
    from PIL.TiffImagePlugin import ImageFileDirectory_v2

    with tifffile.TiffFile(path) as t:
        slots = {k: t.pages[0].tags[k - 1].valueoffset for k in subs}
    data = bytearray(path.read_bytes())
    for k, at in slots.items():
        data[at - 8 : at - 6] = struct.pack("<H", k)
    for k, d in subs.items():
        data += b"\0" * (-len(data) % 4)
        ifd = ImageFileDirectory_v2(prefix=b"II")
        for tag, v in d.items():
            try:
                ifd[tag] = v
            except (TypeError, ValueError):
                continue
        off = len(data)
        data += ifd.tobytes(off)
        data[slots[k] : slots[k] + 4] = struct.pack("<I", off)
    path.write_bytes(bytes(data))


def save_image(img: np.ndarray, out: str | Path, source: str | Path | None = None, quality: int = 95,
               max_bytes: int | None = None) -> Path:
    """Save float RGB [0,1] keeping the source's EXIF (orientation reset) and ICC profile.

    TIFF is always 16-bit, PNG is 16-bit from a RAW or 16-bit source; JPEG/WebP are 8-bit.

    ``max_bytes`` (JPEG/WebP): pick the highest quality that keeps the file under that size.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.rint(np.clip(img, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr, "RGB")
    kwargs: dict = {}
    xmp = None
    exif_obj = None
    deep = False  # source carries more than 8 bits per channel
    if source is not None and is_raw(source):
        exif_obj = raw_exif(source)
        kwargs["exif"] = exif_obj.tobytes()
        kwargs["icc_profile"] = srgb_icc()
        deep = True
    elif source is not None:
        with Image.open(source) as src:
            deep = _high_bit_depth(src)
            icc = src.info.get("icc_profile")
            exif = exif_obj = src.getexif()
            for tag in _TIFF_LAYOUT_TAGS & set(exif):
                del exif[tag]
            if exif:
                exif[_TAG_ORIENTATION] = 1
                kwargs["exif"] = exif.tobytes()
            if icc:
                kwargs["icc_profile"] = icc
            xmp = src.info.get("xmp")
    suffix = out.suffix.lower()
    if suffix in (".tif", ".tiff") or (deep and suffix == ".png"):  # TIFF is the print format: always 16-bit
        return _save_16bit(img, out, kwargs.get("icc_profile"), exif_obj if kwargs.get("exif") else None, xmp)
    if suffix in (".jpg", ".jpeg"):
        kwargs.update(quality=quality, subsampling=0, optimize=True)  # dropped below if libjpeg overflows
        if xmp:
            kwargs["xmp"] = xmp
    elif suffix in (".tif", ".tiff"):
        kwargs.update(compression="tiff_deflate")
    elif suffix == ".webp":
        kwargs.update(quality=quality, method=4)
    if max_bytes and suffix in (".jpg", ".jpeg", ".webp"):
        kwargs = _quality_for_size(pil, "WEBP" if suffix == ".webp" else "JPEG", kwargs, quality, max_bytes)
    try:
        try:
            pil.save(out, **kwargs)
        except OSError:  # libjpeg's optimize pass can overflow on high-entropy or CMYK-derived images
            if not kwargs.pop("optimize", False):
                raise
            pil.save(out, **kwargs)
    except RuntimeError:  # libtiff (used for compression) cannot write EXIF sub-IFDs: keep the metadata, drop compression
        kwargs.pop("compression", None)
        pil.save(out, **kwargs)
    return out


def save_png_u8(arr: np.ndarray, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)).save(out)
    return out


def jpeg_bytes(img: np.ndarray, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


# -- camera RAW ------------------------------------------------------------------------------------
# Base tone curve, as (scene-linear fraction of sensor saturation -> display-linear) control points.
# Calibrated against Nikon in-camera JPEGs (mid grey lands slightly darker than Picture Control
# "Standard", shadows keep more detail) and joined with a monotone cubic so it has no kinks.
# Raw clipping maps to 1.0, so highlights roll off smoothly instead of hard-clipping.
RAW_CURVE_X = (0.0, 0.0025, 0.01, 0.03, 0.09, 0.2, 0.4, 0.7, 1.0)
RAW_CURVE_Y = (0.0, 0.0035, 0.017, 0.074, 0.272, 0.57, 0.835, 0.962, 1.0)
RAW_SATURATION = 1.1  # camera-matrix colour is accurate but looks a touch flat next to camera JPEGs
RAW_RENDER_VERSION = "raw-v2"
RAW_MATCH_EV = 2.0  # most a RAW is brightened or darkened to meet its camera JPEG
_RAW_LUT: np.ndarray | None = None
_LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _raw_lut() -> np.ndarray:
    """65536-entry LUT: linear 16-bit -> tone-mapped, sRGB-encoded float."""
    global _RAW_LUT
    if _RAW_LUT is None:
        from scipy.interpolate import PchipInterpolator

        x = np.linspace(0.0, 1.0, 65536)
        y = np.clip(PchipInterpolator(RAW_CURVE_X, RAW_CURVE_Y)(x), 0.0, 1.0)
        _RAW_LUT = np.where(y <= 0.0031308, y * 12.92, 1.055 * np.power(y, 1 / 2.4) - 0.055).astype(np.float32)
    return _RAW_LUT


def _raw_geometry(raw) -> tuple[tuple[int, int, int, int], int]:
    """(top, left, height, width) of the camera's default crop in the unrotated full-size frame, and the flip code."""
    s = raw.sizes
    if s.crop_width and s.crop_height:
        box = (s.crop_top_margin, s.crop_left_margin, s.crop_height, s.crop_width)
    else:
        box = (0, 0, s.height, s.width)
    return box, s.flip


def _orient(img: np.ndarray, flip: int) -> np.ndarray:
    if flip == 3:
        return img[::-1, ::-1]
    if flip == 5:  # 90 degrees counter-clockwise
        return np.rot90(img, 1)
    if flip == 6:  # 90 degrees clockwise
        return np.rot90(img, -1)
    return img


def _thumb_luma(raw) -> float | None:
    """Mean sRGB luma of the camera's embedded JPEG: how bright the camera rendered this frame."""
    import rawpy

    try:
        th = raw.extract_thumb()
    except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
        return None
    if th.format != rawpy.ThumbFormat.JPEG:
        return None
    try:
        with Image.open(io.BytesIO(th.data)) as im:
            if max(im.size) < 320:
                return None
            im.draft("RGB", (512, 512))
            a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    except OSError:
        return None
    return float((a @ _LUM).mean())


def _shoulder16(v: np.ndarray, knee: float = 0.6 * 65535) -> np.ndarray:
    over = np.maximum(v - knee, 0)
    return np.where(v > knee, knee + (65535 - knee) * np.tanh(over / (65535 - knee)), v)


def _match_exposure(lin: np.ndarray, target: float) -> np.ndarray:
    """Scale linear sensor data so the rendered mean luma meets the camera JPEG's (within +-RAW_MATCH_EV).

    Camera makers render RAW very differently (Canon/Sony/Nikon Z JPEGs sit ~1 EV above a neutral
    decode); the camera JPEG is what the photographer saw and exposed for. Pushes roll off into white.
    """
    step = max(1, int(np.sqrt(lin.shape[0] * lin.shape[1] / 120_000)))
    sm = lin[::step, ::step].astype(np.float32)
    lut = _raw_lut()

    def mean_luma(ev):
        v = _shoulder16(sm * 2.0**ev) if ev > 0 else sm * 2.0**ev
        return float((lut[np.clip(v, 0, 65535).astype(np.uint16)] @ _LUM).mean())

    lo, hi = -RAW_MATCH_EV, RAW_MATCH_EV
    if not mean_luma(lo) < target < mean_luma(hi):
        ev = lo if target <= mean_luma(lo) else hi
    else:
        for _ in range(14):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if mean_luma(mid) < target else (lo, mid)
        ev = (lo + hi) / 2
    if abs(ev) < 0.05:
        return lin
    v = lin.astype(np.float32) * np.float32(2.0**ev)
    if ev > 0:
        v = _shoulder16(v)
    return np.clip(v, 0, 65535).astype(np.uint16)


def _embedded_full_jpeg(path: str | Path) -> np.ndarray | None:
    import rawpy

    try:
        with rawpy.imread(str(path)) as raw:
            (_, _, h, w), flip = _raw_geometry(raw)
            th = raw.extract_thumb()
    except rawpy.LibRawError:
        return None
    if th.format != rawpy.ThumbFormat.JPEG:
        return None
    with Image.open(io.BytesIO(th.data)) as im:
        if max(im.size) < 0.9 * max(h, w):
            return None
        img = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    if (img.shape[1] > img.shape[0]) == (w > h):
        img = np.ascontiguousarray(_orient(img, flip))
    return img


def render_raw(path: str | Path, max_side: int | None = None) -> np.ndarray:
    """Decode a RAW file to oriented sRGB float32 [0, 1].

    Camera white balance, camera colour matrix to sRGB, highlight blending (no magenta clipped
    skies), no auto-brightening (so a burst or a whole album renders consistently), then a
    fixed camera-like tone curve and a gentle saturation lift. Sizes up to half the sensor
    resolution use LibRaw's fast half-size decode, which goes through the same colour pipeline.
    """
    import rawpy

    try:
        with rawpy.imread(str(path)) as raw:
            (top, left, h, w), flip = _raw_geometry(raw)
            half = bool(max_side) and max_side <= max(h, w) // 2
            lin = raw.postprocess(
                half_size=half, use_camera_wb=True, no_auto_bright=True, gamma=(1, 1), output_bps=16,
                output_color=rawpy.ColorSpace.sRGB, highlight_mode=rawpy.HighlightMode.Blend, user_flip=0,
            )
            preview = _thumb_luma(raw)
    except (rawpy.LibRawFileUnsupportedError, rawpy.LibRawDataError) as e:
        # e.g. Nikon High Efficiency NEF, which LibRaw cannot demosaic: use the camera's full-size JPEG instead.
        img = _embedded_full_jpeg(path)
        if img is None:
            raise
        if max_side and max(img.shape[:2]) > max_side:
            img = resize(img, fit_size(img.shape[1], img.shape[0], max_side))
        return img
    k = 2 if half else 1
    lin = lin[top // k : (top + h) // k, left // k : (left + w) // k]
    if preview is not None:
        lin = _match_exposure(lin, preview)
    if RAW_SATURATION != 1.0:
        f = lin.astype(np.float32)
        y = f @ _LUM
        f = y[..., None] + (f - y[..., None]) * RAW_SATURATION
        lin = np.clip(f, 0, 65535).astype(np.uint16)
    img = np.ascontiguousarray(_orient(_raw_lut()[lin], flip))
    if max_side and max(img.shape[:2]) > max_side:
        img = resize(img, fit_size(img.shape[1], img.shape[0], max_side))
    return img


def load_preview(path: str | Path, max_side: int = 2048) -> np.ndarray:
    """Fast oriented RGB float32 [0, 1] for culling and thumbnails.

    For RAW files this is the camera's embedded JPEG preview when it is large enough (Nikon
    bodies embed a full-size one), so no demosaicing is needed; otherwise a half-size decode.
    Not used for analysis or export, whose colours must match the RAW rendering exactly.
    """
    if not is_raw(path):
        return load_rgb(path, max_side)
    import rawpy

    with rawpy.imread(str(path)) as raw:
        (_, _, h, w), flip = _raw_geometry(raw)
        try:
            th = raw.extract_thumb()
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            th = None
    img = None
    if th is not None and th.format == rawpy.ThumbFormat.JPEG:
        with Image.open(io.BytesIO(th.data)) as im:
            if max(im.size) >= min(max_side, max(h, w) // 2):
                im.draft("RGB", (max_side, max_side))
                img = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    elif th is not None and th.format == rawpy.ThumbFormat.BITMAP and max(th.data.shape[:2]) >= min(max_side, max(h, w) // 2):
        img = th.data.astype(np.float32) / 255.0
    if img is None:
        return render_raw(path, max_side)
    # Embedded previews are stored in sensor orientation; follow the RAW's orientation, not the preview's.
    if (img.shape[1] > img.shape[0]) == (w > h):
        img = np.ascontiguousarray(_orient(img, flip))
    if max(img.shape[:2]) > max_side:
        img = resize(img, fit_size(img.shape[1], img.shape[0], max_side))
    return img


def _clean(v):
    return v.strip("\x00 ") if isinstance(v, str) else v


def _read_raw_info(path: Path) -> PhotoInfo:
    import rawpy

    with rawpy.imread(str(path)) as raw:
        (_, _, h, w), flip = _raw_geometry(raw)
        other = raw.other
    if flip in (5, 6):
        w, h = h, w
    sub = raw_exif(path).get_ifd(_EXIF_IFD)
    captured = None
    stamp = _clean(sub.get(_TAG_DATETIME_ORIGINAL))
    if stamp:
        try:
            captured = datetime.strptime(stamp, "%Y:%m:%d %H:%M:%S")
        except ValueError:
            captured = None
    if captured is None and other.timestamp:
        captured = other.timestamp
    pos = lambda v: float(v) if v and v > 0 else None
    return PhotoInfo(path=path, name=path.name, width=w, height=h, captured=captured,
                     exposure_time=pos(other.shutter_speed), f_number=pos(other.aperture), iso=pos(other.iso_speed),
                     extra={"raw": True})


# EXIF carried from a RAW into exports: identity, capture settings, lens, GPS. Nikon's MakerNote
# holds offsets into the RAW file and the TIFF structure tags describe the RAW's thumbnail, so
# both are dropped.
_RAW_IFD0_TAGS = (0x010F, 0x0110, 0x0131, 0x0132, 0x013B, 0x8298)
_RAW_SKIP_EXIF = {0x927C, 0xA005}
_GPS_IFD = 0x8825


def raw_exif(path: str | Path) -> Image.Exif:
    """EXIF for files rendered from a RAW (orientation normalised to 1)."""
    import rawpy

    out = Image.Exif()
    src_exif = None
    try:  # TIFF-based RAWs (NEF, DNG, ARW, ...) expose their EXIF to Pillow directly
        with Image.open(path) as im:
            src_exif = im.getexif()
            sub, gps = dict(src_exif.get_ifd(_EXIF_IFD)), dict(src_exif.get_ifd(_GPS_IFD))
    except Exception:  # noqa: BLE001 - NRW/CR3/RAF/...: fall back to what LibRaw parsed
        src_exif = None
    if src_exif:
        for t in _RAW_IFD0_TAGS:
            if t in src_exif:
                out[t] = _clean(src_exif[t])
        out[_TAG_ORIENTATION] = 1
        if sub:
            out.get_ifd(_EXIF_IFD).update({k: _clean(v) for k, v in sub.items() if k not in _RAW_SKIP_EXIF})
        if len(gps) > 1:
            out.get_ifd(_GPS_IFD).update(gps)
        return out
    with rawpy.imread(str(path)) as raw:
        other, lens = raw.other, raw.lens
    out[_TAG_ORIENTATION] = 1
    if other.artist:
        out[0x013B] = other.artist
    e = out.get_ifd(_EXIF_IFD)
    if other.timestamp:
        e[_TAG_DATETIME_ORIGINAL] = other.timestamp.strftime("%Y:%m:%d %H:%M:%S")
    if other.shutter_speed > 0:
        e[_TAG_EXPOSURE_TIME] = float(other.shutter_speed)
    if other.aperture > 0:
        e[_TAG_FNUMBER] = float(other.aperture)
    if other.iso_speed > 0:
        e[_TAG_ISO] = int(other.iso_speed)
    if other.focal_length > 0:
        e[0x920A] = float(other.focal_length)
    if lens.model:
        e[0xA434] = lens.model
    return out


def srgb_icc() -> bytes:
    from PIL import ImageCms

    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
