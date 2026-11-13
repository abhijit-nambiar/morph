"""Image loading and export that preserves geometry, EXIF and ICC metadata."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

_EXIF_IFD = 0x8769
_TAG_DATETIME_ORIGINAL = 0x9003
_TAG_DATETIME = 0x0132
_TAG_EXPOSURE_TIME = 0x829A
_TAG_FNUMBER = 0x829D
_TAG_ISO = 0x8827
_TAG_ORIENTATION = 0x0112


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


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"album folder not found: {folder}")
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("."))
    if not files:
        raise ValueError(f"no images found in {folder}")
    return files


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
    """Load an oriented RGB image as float32 in [0, 1], optionally downscaled."""
    with Image.open(path) as im:
        if max_side:
            im.draft("RGB", (max_side, max_side))
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max_side and max(im.size) > max_side:
            scale = max_side / max(im.size)
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
        return np.asarray(im, dtype=np.float32) / 255.0


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


def save_image(img: np.ndarray, out: str | Path, source: str | Path | None = None, quality: int = 95) -> Path:
    """Save float RGB [0,1] keeping the source's EXIF (orientation reset) and ICC profile."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.rint(np.clip(img, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr, "RGB")
    kwargs: dict = {}
    if source is not None:
        with Image.open(source) as src:
            icc = src.info.get("icc_profile")
            exif = src.getexif()
            if exif:
                exif[_TAG_ORIENTATION] = 1
                kwargs["exif"] = exif.tobytes()
            if icc:
                kwargs["icc_profile"] = icc
            xmp = src.info.get("xmp")
    else:
        xmp = None
    suffix = out.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        kwargs.update(quality=quality, subsampling=0, optimize=True)
        if xmp:
            kwargs["xmp"] = xmp
    elif suffix in (".tif", ".tiff"):
        kwargs.update(compression="tiff_deflate")
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
