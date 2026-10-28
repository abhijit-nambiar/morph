"""Experimental frame-relative smooth local-light and global color mapping."""
import numpy as np
import poly_color

CENTERS=(.15,.5,.85)
SIGMA=.27


def masks(height,width,row_start=0,full_height=None,full_width=None,row_step=1,col_step=1):
    full_height=full_height or height;full_width=full_width or width
    yy=(row_start+np.arange(height,dtype=np.float32)*row_step+.5)/full_height
    xx=(np.arange(width,dtype=np.float32)*col_step+.5)/full_width
    for cy in CENTERS:
        for cx in CENTERS:
            yield np.exp(-.5*(((xx[None,:]-cx)/SIGMA)**2+((yy[:,None]-cy)/SIGMA)**2))


def features(x):
    return np.concatenate([poly_color.features(x),*(m[...,None]*x*(1-x) for m in masks(*x.shape[:2]))],axis=-1)


def identity():return np.zeros((37,3),dtype=np.float64)
def _weights(w):
    w=np.asarray(w,dtype=np.float64)
    if w.shape!=(37,3) or not np.isfinite(w).all():raise ValueError('Invalid grid lighting mapping')
    return w

def serialize(w):return {'kind':'grid-rgb-v1','weights':_weights(w).tolist()}
def deserialize(obj):
    if obj.get('kind')!='grid-rgb-v1':raise ValueError('Unsupported grid lighting mapping')
    return _weights(obj['weights'])


def fit_with_metrics(x,y,regularization=.001):
    if x.shape!=y.shape or x.ndim!=3 or x.shape[-1]!=3:raise ValueError('Images must have matching RGB dimensions')
    if not np.isfinite(regularization) or regularization<0:raise ValueError('regularization must be finite and nonnegative')
    rng=np.random.default_rng(0);idx=rng.choice(x.shape[0]*x.shape[1],min(x.shape[0]*x.shape[1],30000),replace=False)
    a=features(x).reshape(-1,37)[idx].astype(np.float64);b=(y-x).reshape(-1,3)[idx].astype(np.float64)
    w=np.linalg.lstsq(a.T@a/len(a)+regularization*np.eye(37),a.T@b/len(a),rcond=None)[0]
    e=render(x,w)-y
    return w,{'initial_mse':float(np.mean((x-y)**2)),'fit_mse':float(np.mean(e**2)),'rmse':float(np.sqrt(np.mean(e**2))),'metric_scope':'entire fitting-resolution image','regularization':regularization}


def render_raw(x,w,row_start=0,full_shape=None,row_step=1,col_step=1):
    full_shape=full_shape or x.shape
    w=_weights(w);result=x+poly_color.features(x)@w[:10];local=x*(1-x)
    for i,m in enumerate(masks(x.shape[0],x.shape[1],row_start,full_shape[0],full_shape[1],row_step,col_step)):
        result+=m[...,None]*(local@w[10+i*3:13+i*3])
    return result

def out_of_gamut(x,w,pre_transform=None):
    stride=max(1,max(x.shape[:2])//512);sample=x[::stride,::stride]
    if pre_transform is not None:sample=pre_transform(sample)
    raw=render_raw(sample,w,full_shape=x.shape,row_step=stride,col_step=stride)
    return float(np.mean(np.any((raw<0)|(raw>1),axis=-1)))

def render_tile(x,w,row_start,full_shape):return np.clip(render_raw(x,w,row_start,full_shape),0,1)
def render(x,w):return render_tile(x,w,0,x.shape)
render.render_tile=render_tile
