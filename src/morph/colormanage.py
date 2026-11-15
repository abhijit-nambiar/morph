"""LittleCMS conversions between embedded RGB profiles and the sRGB working space."""

import io
import logging
from functools import lru_cache

import imagecodecs
import numpy as np
from PIL import ImageCms

log = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def rgb_profile(profile: bytes) -> bytes | None:
    """Return a non-sRGB RGB profile, or None for unmanaged/unsupported inputs."""
    try:
        info = ImageCms.ImageCmsProfile(io.BytesIO(profile)).profile
        if info.xcolor_space.strip() != 'RGB':
            return None
        if 'srgb' in (info.profile_description or '').lower():
            return None
        return profile
    except Exception as exc:
        log.warning("Invalid ICC profile; keeping native RGB values: %s", exc)
        return None


def source_profile(im) -> bytes | None:
    profile = im.info.get('icc_profile')
    if profile:
        return rgb_profile(profile)
    sub = im.getexif().get_ifd(0x8769)
    interop = im.getexif().get_ifd(0xA005) if 0xA005 in sub else {}
    if sub.get(0xA001) == 0xFFFF and interop.get(1) in ('R03', b'R03', b'R03\0'):
        return imagecodecs.cms_profile('adobergb')
    return None


@lru_cache(maxsize=1)
def working_profile() -> bytes:
    return imagecodecs.cms_profile('srgb')


def convert(img: np.ndarray, profile: bytes, *, exporting: bool = False) -> np.ndarray:
    try:
        src, dst = (working_profile(), profile) if exporting else (profile, working_profile())
        return np.clip(imagecodecs.cms_transform(
            np.ascontiguousarray(img), src, dst, outdtype=np.float32, intent=1,
            colorspace='rgb', outcolorspace='rgb', planar=False, outplanar=False,
        ), 0, 1)
    except Exception as exc:
        log.warning("ICC conversion failed; keeping native RGB values: %s", exc)
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255
        return img.astype(np.float32)
