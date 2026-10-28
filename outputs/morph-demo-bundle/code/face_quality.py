"""Approximate pre-JPEG diagnostics on reduced-resolution face-scene images."""
import numpy as np
import gamut_renderer,face_core


def diagnose(x,weights,boxes,album=None,scene=None,photo=None):
    album=gamut_renderer.vec(album or {});scene=gamut_renderer.vec(scene or {});photo=gamut_renderer.vec(photo or {})
    def stats(im):return {'near_white':float(np.mean(np.any(im>=254/255,axis=-1))), 'near_black':float(np.mean(np.all(im<=1/255,axis=-1)))}
    before=stats(x);source=gamut_renderer.render(x,photo) if np.any(photo) else x
    raw=face_core.model_raw_tile(source,weights,boxes,0,source.shape);out_of_gamut=float(np.mean(np.any((raw<0)|(raw>1),axis=-1)));result=np.clip(raw,0,1)
    if np.any(scene):result=gamut_renderer.render(result,scene)
    if np.any(album):result=gamut_renderer.render(result,album)
    after=stats(result);high=after['near_white']-before['near_white'];low=after['near_black']-before['near_black']
    return {'sample_size':[x.shape[1],x.shape[0]],'before':before,'after':after,'near_white_increase':high,'near_black_increase':low,'near_limit_flag':bool(high>.01 or low>.01),'face_mapping_out_of_gamut_fraction':out_of_gamut}
