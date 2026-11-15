import io
from pathlib import Path

import imagecodecs as cms
import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms, ImageOps

from morph.colorspace import delta_e2000, srgb_to_lab
from morph.imageio import load_rgb, save_image, srgb_icc


def profile(name):
    if name != 'displayp3':
        return cms.cms_profile(name)
    linear = cms.cms_profile('rgb', whitepoint=(.3127, .3290),
                             primaries=(.64, .33, .30, .60, .15, .06), gamma=1)
    ramp = np.repeat(np.linspace(0, 1, 4096, dtype=np.float32)[None, :, None], 3, axis=2)
    curve = transform(ramp, cms.cms_profile('srgb'), linear)[0, :, 0]
    return cms.cms_profile('rgb', whitepoint=(.3127, .3290),
                           primaries=(.68, .32, .265, .69, .15, .06), transferfunction=curve)


def description(icc):
    return ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc)))


def transform(a, src, dst):
    return np.clip(cms.cms_transform(np.ascontiguousarray(a), src, dst,
                                    outdtype=np.float32, intent=1, colorspace='rgb',
                                    outcolorspace='rgb', planar=False, outplanar=False), 0, 1)


@pytest.fixture
def photo():
    path = Path(__file__).parents[1] / "Morph/originals/DSC_0901-1.jpg"
    if not path.exists() or path.stat().st_size < 512:
        pytest.skip('real photo requires Git LFS assets')
    with Image.open(path) as im:
        im.thumbnail((640, 640))
        return np.asarray(im.convert('RGB')).copy()


@pytest.mark.parametrize('name', ['adobergb', 'displayp3'])
def test_real_photo(tmp_path, photo, name):
    src, dst = profile('srgb'), profile(name)
    wide = transform(photo, src, dst)
    Image.fromarray(photo).save(tmp_path/'s.jpg', quality=100, subsampling=0, icc_profile=src)
    Image.fromarray(np.rint(wide*255).astype('uint8')).save(
        tmp_path/'w.jpg', quality=100, subsampling=0, icc_profile=dst)
    a, b = load_rgb(tmp_path/'s.jpg'), load_rgb(tmp_path/'w.jpg')
    assert delta_e2000(srgb_to_lab(a), srgb_to_lab(b)).mean() < 1


@pytest.mark.parametrize('dtype', ['uint16', 'float32'])
def test_deep_precision(tmp_path, dtype):
    profile = cms.cms_profile('adobergb')
    a = np.broadcast_to(np.linspace(.1, .9, 2048, dtype=np.float32)[None, :, None], (3, 2048, 3)).copy()
    a = np.rint(a*65535).astype(dtype) if dtype == 'uint16' else a
    tifffile.imwrite(tmp_path/'wide.tif', a, photometric='rgb', iccprofile=profile)
    expected = transform(a, profile, cms.cms_profile('srgb'))
    actual = load_rgb(tmp_path/'wide.tif')
    assert len(np.unique(actual[..., 0])) > 1000
    np.testing.assert_allclose(actual, expected, atol=2e-6)


@pytest.mark.parametrize('suffix', ['.jpg', '.tif', '.png'])
def test_export_roundtrip(tmp_path, photo, suffix):
    icc = profile('displayp3')
    wide = transform(photo, cms.cms_profile('srgb'), icc)
    src = tmp_path/'source.tif'
    tifffile.imwrite(src, np.rint(wide*65535).astype('uint16'), photometric='rgb', iccprofile=icc)
    a = load_rgb(src)
    out = save_image(a, tmp_path/('out'+suffix), source=src, quality=100)
    with Image.open(out) as im:
        assert im.info['icc_profile'] == icc
        assert im.getexif().get_ifd(0x8769)[0xA001] == 0xFFFF
    b = load_rgb(out)
    if suffix == '.jpg':
        assert delta_e2000(srgb_to_lab(a), srgb_to_lab(b)).mean() < 1
    else:
        np.testing.assert_allclose(b, a, atol=.0003)


def test_exif_adobe(tmp_path, photo):
    exif = Image.Exif()
    exif[0x8769] = {0x9003: '2024:05:01 10:00:00', 0xA001: 0xFFFF,
                    0xA005: {1: 'R03'}}
    src = tmp_path/'adobe.jpg'
    Image.fromarray(photo).save(src, exif=exif, quality=100, subsampling=0)
    with Image.open(src) as im:
        expected = transform(np.asarray(im), cms.cms_profile('adobergb'), cms.cms_profile('srgb'))
    np.testing.assert_allclose(load_rgb(src), expected, atol=1e-7)
    for suffix in ('.jpg', '.tif'):
        out = save_image(expected, tmp_path/('out'+suffix), source=src)
        with Image.open(out) as im:
            sub = im.getexif().get_ifd(0x8769)
            assert sub[0x9003] == '2024:05:01 10:00:00'
            assert sub[0xA001] == 0xFFFF
            assert ImageCms.ImageCmsProfile(io.BytesIO(im.info['icc_profile'])).profile.xcolor_space.strip() == 'RGB'


@pytest.mark.parametrize('profile', [None, srgb_icc(), b'broken ICC'])
@pytest.mark.parametrize('max_side', [None, 80])
def test_unchanged(tmp_path, photo, profile, max_side, caplog):
    src = tmp_path/'source.png'
    Image.fromarray(photo).save(src, icc_profile=profile)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        if max_side:
            scale = max_side/max(im.size)
            im = im.resize((round(im.width*scale), round(im.height*scale)), Image.LANCZOS)
        expected = np.asarray(im, dtype=np.float32)/255
    np.testing.assert_array_equal(load_rgb(src, max_side), expected)


def test_export_srgb(tmp_path, photo):
    src = tmp_path/'plain.jpg'
    exif = Image.Exif()
    exif[0x8769] = {0xA001: 0xFFFF}
    Image.fromarray(photo).save(src, exif=exif)
    out = save_image(load_rgb(src), tmp_path/'out.jpg', source=src)
    with Image.open(out) as im:
        assert 'srgb' in description(im.info['icc_profile']).lower()
        assert im.getexif().get_ifd(0x8769)[0xA001] == 1


def test_broken_profile_warns(tmp_path, photo, caplog):
    src = tmp_path/'broken.png'
    Image.fromarray(photo).save(src, icc_profile=b'another broken profile')
    np.testing.assert_array_equal(load_rgb(src), photo.astype(np.float32)/255)
    assert 'Invalid ICC profile' in caplog.text
    out = save_image(load_rgb(src), tmp_path/'out.png', source=src)
    with Image.open(out) as im:
        assert 'srgb' in description(im.info['icc_profile']).lower()


def test_cache_version():
    from morph.album import ANALYSIS_VERSION

    assert ANALYSIS_VERSION.startswith('a2-icc-')
