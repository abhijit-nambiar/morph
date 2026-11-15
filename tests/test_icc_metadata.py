import struct
from datetime import datetime

import numpy as np
import tifffile
from PIL import Image
from PIL.TiffImagePlugin import IFDRational, ImageFileDirectory_v2

from morph import imageio
from morph.album import Album


def float_tiff_with_exif(path):
    pixels = np.linspace(.1, .8, 48*72*3, dtype=np.float32).reshape(48, 72, 3)
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
    return pixels


def test_float_tiff_exif_import_and_export(tmp_path, fake_models):
    folder = tmp_path/'album'
    folder.mkdir()
    source = folder/'float.tif'
    pixels = float_tiff_with_exif(source)
    with tifffile.TiffFile(source) as tif:
        assert tif.pages[0].tags[0x8769].value['DateTimeOriginal'] == '2024:05:01 10:00:00'
    info = imageio.read_info(source)
    assert info.captured == datetime(2024, 5, 1, 10)
    assert info.iso == 200
    assert info.exposure_time == 1/125
    np.testing.assert_allclose(imageio.load_rgb(source), pixels)
    album = Album(folder).analyze()
    assert album.names == ['float.tif']
    assert not album.skipped
    out = imageio.save_image(pixels, tmp_path/'export.jpg', source=source)
    with Image.open(out) as im:
        assert im.getexif().get_ifd(0x8769)[0x9003] == '2024:05:01 10:00:00'
