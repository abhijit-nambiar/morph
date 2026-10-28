"""Experimental deterministic face-relative color and lighting mapping."""
import numpy as np
import morph,poly_color


def tile_mask(shape,boxes,row_start,full_shape,mode):
    h,w=shape[:2];yy=(np.arange(row_start,row_start+h,dtype=np.float32)+.5)/full_shape[0];xx=(np.arange(w,dtype=np.float32)+.5)/full_shape[1];result=np.zeros((h,w),dtype=np.float32)
    for b in boxes:
        cx=b['x']+b['width']*.5;cy=b['y']+b['height']*.5
        if mode=='model':
            sx=max(b['width']*.8,.025);sy=max(b['height']*.8,.025);m=np.exp(-.5*(((xx[None,:]-cx)/sx)**2+((yy[:,None]-cy)/sy)**2))
        elif mode=='target':
            distance=np.sqrt(((xx[None,:]-cx)/(b['width']*1.2))**2+((yy[:,None]-cy)/(b['height']*1.1))**2)
            m=np.where(distance<=.35,1.,np.where(distance>=1,0.,.5+.5*np.cos(np.pi*(distance-.35)/.65)))
        else:raise ValueError('Unknown mask mode')
        result=np.maximum(result,m)
    return result

def model_raw_tile(x,w,boxes,start,shape):
    m=tile_mask(x.shape,boxes,start,shape,'model');return x+poly_color.features(x)@w[:10]+(m[...,None]*x*(1-x))@w[10:]
def model_tile(x,w,boxes,start,shape):
    return np.clip(model_raw_tile(x,w,boxes,start,shape),0,1)
def render(x,boxes,w):return model_tile(x,w,boxes,0,x.shape)
def target(x,boxes,settings,strength):
    m=tile_mask(x.shape,boxes,0,x.shape,'target');return np.clip(morph.pillow_soft_target(x,'neutral',settings)+strength*m[...,None]*x*(1-x),0,1)
def fit(x,y,boxes,penalty=.0001):
    if x.shape!=y.shape or x.ndim!=3 or x.shape[-1]!=3:raise ValueError('Images must have matching RGB dimensions')
    if not np.isfinite(penalty) or penalty<0:raise ValueError('Invalid regularization')
    m=tile_mask(x.shape,boxes,0,x.shape,'model');features=np.concatenate([poly_color.features(x),m[...,None]*x*(1-x)],axis=-1)
    rng=np.random.default_rng(0);idx=rng.choice(x.shape[0]*x.shape[1],min(x.shape[0]*x.shape[1],30000),replace=False)
    a=features.reshape(-1,13)[idx].astype(np.float64);b=(y-x).reshape(-1,3)[idx].astype(np.float64)
    return np.linalg.lstsq(a.T@a/len(a)+penalty*np.eye(13),a.T@b/len(a),rcond=None)[0]
