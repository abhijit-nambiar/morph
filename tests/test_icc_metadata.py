import struct
from datetime import datetime

import imagecodecs
import numpy as np
import pytest
import tifffile
from PIL import Image
from PIL.TiffImagePlugin import IFDRational, ImageFileDirectory_v2

from morph import imageio
from morph.album import Album


def integer_tiff_with_exif(path):
    pixels = np.linspace(6554, 52428, 48*72*3).astype(np.uint16).reshape(48, 72, 3)
    tifffile.imwrite(path, pixels, photometric='rgb', metadata=None,
                     extratags=[(0x8768, 'I', 1, 0, True)])
    with tifffile.TiffFile(path) as tif:
        slot = tif.pages[0].tags[0x8768].valueoffset
    data = bytearray(path.read_bytes())
    data[slot-8:slot-6] = struct.pack('<H', 0x8769)
    exif = ImageFileDirectory_v2(prefix=b'II')
    exif[0x9003] = '2024:05:01 10:00:00'
    exif[0x8827] = 200
    exif[0x829A] = IFDRational(1, 125)
    offset = len(data)
    data += exif.tobytes(offset)
    data[slot:slot+4] = struct.pack('<I', offset)
    path.write_bytes(data)
    return pixels.astype(np.float32)/65535


def test_integer_tiff_exif_import_and_export(tmp_path, fake_models):
    folder = tmp_path/'album'
    folder.mkdir()
    source = folder/'16bit.tif'
    pixels = integer_tiff_with_exif(source)
    with tifffile.TiffFile(source) as tif:
        assert tif.pages[0].tags[0x8769].value['DateTimeOriginal'] == '2024:05:01 10:00:00'
    info = imageio.read_info(source)
    assert info.captured == datetime(2024, 5, 1, 10)
    assert info.iso == 200
    assert info.exposure_time == 1/125
    np.testing.assert_allclose(imageio.load_rgb(source), pixels)
    album = Album(folder).analyze()
    assert album.names == ['16bit.tif']
    assert not album.skipped
    out = imageio.save_image(pixels, tmp_path/'export.jpg', source=source)
    with Image.open(out) as im:
        assert im.getexif().get_ifd(0x8769)[0x9003] == '2024:05:01 10:00:00'


@pytest.mark.parametrize('bigtiff', [False, True])
def test_unsupported_float_rgb_tiff_is_skipped(tmp_path, fake_models, bigtiff):
    from PIL import UnidentifiedImageError

    folder = tmp_path/'album'
    folder.mkdir()
    source = folder/'float.tif'
    tifffile.imwrite(source, np.full((48, 72, 3), .3, np.float32), photometric='rgb',
                     bigtiff=bigtiff, metadata=None, description=None,
                     extratags=[(270, 's', 0, 'Café — 東京'.encode('utf-8'), True)])
    Image.new('RGB', (72, 48), (110, 120, 130)).save(folder/'normal.jpg')
    with pytest.raises(UnidentifiedImageError):
        imageio.load_rgb(source)
    album = Album(folder).analyze()
    assert album.names == ['normal.jpg']
    assert [name for name, reason in album.skipped] == ['float.tif']


@pytest.mark.parametrize('suffix', ['.jpg', '.tif'])
@pytest.mark.parametrize('wide', [False, True])
@pytest.mark.parametrize('interop', [{1: 'R98', 2: b'0100'}, {}, None])
def test_export_nested_interop(tmp_path, suffix, wide, interop):
    source = tmp_path/'camera.jpg'
    exif = Image.Exif()
    sub = {0x9003: '2024:05:01 10:00:00', 0xA001: 1}
    if interop is not None:
        sub[0xA005] = interop
    exif[0x8769] = sub
    icc = imagecodecs.cms_profile('adobergb' if wide else 'srgb')
    Image.new('RGB', (72, 48), (110, 120, 130)).save(source, exif=exif, icc_profile=icc)
    out = imageio.save_image(imageio.load_rgb(source), tmp_path/('export'+suffix), source=source)
    with Image.open(out) as im:
        saved = im.getexif()
        sub = saved.get_ifd(0x8769)
        assert sub[0x9003] == '2024:05:01 10:00:00'
        assert sub[0xA001] == (0xFFFF if wide else 1)
        if interop:
            assert 0xA005 in sub
            nested = saved.get_ifd(0xA005)
            assert nested[2] == b'0100'
            if wide:
                assert 1 not in nested
            else:
                assert nested[1] == 'R98'
        else:
            assert 0xA005 not in sub
    if suffix == '.tif':
        with tifffile.TiffFile(out) as tif:
            assert tif.pages[0].dtype == np.uint16


@pytest.mark.parametrize('texts', [
    {270: 'Café — 東京', 315: 'José Núñez', 33432: '© 2024'},
    {270: 'Caption', 315: 'Artist', 33432: 'Copyright 2024'},
    {270: '', 315: '', 33432: ''},
])
def test_print_export_preserves_non_ascii_text_bytes(tmp_path, texts):
    root = ImageFileDirectory_v2(prefix=b'II')
    for tag, text in texts.items():
        root[tag] = text.encode('utf-8')
    root[0x8769] = 0
    sub = ImageFileDirectory_v2(prefix=b'II')
    sub[0x9003] = '2024:05:01 10:00:00'
    sub[0xA001] = 1
    sub[0xA430] = texts[315].encode('utf-8')
    sub.tagtype[0xA430] = 2
    offset = 8 + len(root.tobytes(8))
    root[0x8769] = offset
    metadata = b'Exif\0\0II*\0\x08\0\0\0' + root.tobytes(8) + sub.tobytes(offset)
    source = tmp_path/'camera.jpg'
    Image.new('RGB', (72, 48), (110, 120, 130)).save(source, exif=metadata)
    out = imageio.save_image(imageio.load_rgb(source), tmp_path/'print.tif', source=source)
    with Image.open(out) as im:
        saved = im.getexif()
        for tag, text in texts.items():
            assert saved.get(tag, '').encode('latin-1') == text.encode('utf-8')
        sub = saved.get_ifd(0x8769)
        assert sub[0xA430].encode('latin-1').decode('utf-8') == texts[315]
        assert sub[0x9003] == '2024:05:01 10:00:00'
        assert sub[0xA001] == 1
    with tifffile.TiffFile(out) as tif:
        assert tif.pages[0].dtype == np.uint16
        for tag, text in texts.items():
            if text:
                assert tif.pages[0].tags[tag].value == text
