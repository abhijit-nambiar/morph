#!/usr/bin/env python3
"""Measure neutral JPEG re-encoding error at original resolution."""
import argparse,json,math,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import morph,gamut_renderer

def main():
    p=argparse.ArgumentParser();p.add_argument('groups');p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=4);p.add_argument('--sample',type=int,default=150000);a=p.parse_args();groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));idx=np.linspace(0,len(groups)-1,min(a.count,len(groups)),dtype=int);rows=[]
    with tempfile.TemporaryDirectory(prefix='morph-detail-') as td:
        for i in idx:
            g=groups[int(i)];source=Path(g['representative']);original=morph.load(source);h,w=original.shape[:2];rng=np.random.default_rng(42);flat=original.reshape(-1,3);ids=rng.choice(len(flat),min(a.sample,len(flat)),replace=False);reference=flat[ids].copy();identity=gamut_renderer.render(reference,np.zeros(9));neutral_error=float(np.max(np.abs(identity-reference)))
            path=Path(td)/'neutral.jpg';morph.save_render(original,np.zeros(9),path,renderer=gamut_renderer.render,source=source);del original,flat
            decoded=morph.load(path);samples=decoded.reshape(-1,3)[ids];diff=samples.astype(np.float64)-reference;rmse=float(np.sqrt(np.mean(diff**2)));mae=float(np.mean(abs(diff)));percentile=float(np.quantile(abs(diff),.99));psnr=float(20*math.log10(1/rmse)) if rmse else None
            with Image.open(source) as inp, Image.open(path) as out:
                src_exif=inp.getexif().get_ifd(34665).get(36867);dst_exif=out.getexif().get_ifd(34665).get(36867);info={'same_oriented_dimensions':morph.oriented_size(inp)==out.size,'capture_time_preserved':src_exif==dst_exif,'output_has_icc':bool(out.info.get('icc_profile'))}
            rows.append({'scene':g['id'],'source':str(source),'dimensions':[w,h],'sampled_full_resolution_pixels':len(ids),'pre_encoding_neutral_max_error':neutral_error,'jpeg_roundtrip':{'rmse':rmse,'mean_absolute_error':mae,'absolute_error_99th_percentile':percentile,'psnr_db':psnr},'metadata':info});print(g['id'],'PSNR',round(psnr,2),'dB',flush=True)
    morph.dump({'scope':'Neutral re-encode through the bounded renderer and JPEG quality 95, full-resolution images but random pixel samples only. RGB metrics use decoded sRGB and compare against the original JPEG; they do not measure perceived sharpness, non-neutral edits, or absolute camera RAW quality. JPEG encoding is lossy.','cases':rows},a.out)
if __name__=='__main__':main()
