"""Cubic RGB mapping using the existing regularized color-fit convention."""
import numpy as np
import poly_color

NAMES = poly_color.NAMES + ('red3','green3','blue3','red2_green','red2_blue',
                           'green2_red','green2_blue','blue2_red','blue2_green','red_green_blue')


def features(x):
    r,g,b=x[...,0],x[...,1],x[...,2]
    return np.concatenate((poly_color.features(x),np.stack(
        (r**3,g**3,b**3,r*r*g,r*r*b,g*g*r,g*g*b,b*b*r,b*b*g,r*g*b),axis=-1)),axis=-1)


def identity():return np.zeros((len(NAMES),3),dtype=np.float64)


def render_raw(x,weights):
    w=np.asarray(weights,dtype=np.float64)
    if w.shape!=(len(NAMES),3) or not np.isfinite(w).all():
        raise ValueError('Invalid cubic color mapping')
    return x+features(x)@w


def render(x,weights):return np.clip(render_raw(x,weights),0,1)


def out_of_gamut(x,weights):
    stride=max(1,max(x.shape[:2])//512)
    raw=render_raw(x[::stride,::stride],weights)
    return float(np.mean(np.any((raw<0)|(raw>1),axis=-1)))


def fit(x,y,regularization=1e-6,sample_size=30000,seed=0):
    if x.shape!=y.shape or x.shape[-1]!=3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Input and target must have matching finite RGB shapes')
    if not np.isfinite(regularization) or regularization<0:
        raise ValueError('regularization must be finite and nonnegative')
    x=x.reshape(-1,3);y=y.reshape(-1,3)
    idx=np.random.default_rng(seed).choice(len(x),min(len(x),sample_size),replace=False)
    a=features(x[idx]).astype(np.float64);b=(y[idx]-x[idx]).astype(np.float64)
    weights=np.linalg.lstsq(a.T@a/len(a)+regularization*np.eye(len(NAMES)),a.T@b/len(a),rcond=None)[0]
    error=render(x[idx],weights)-y[idx]
    mse=float(np.mean(error*error))
    return weights,{'sampled_pixels':len(idx),'initial_mse':float(np.mean(b*b)),
                    'fit_mse':mse,'rmse':float(np.sqrt(mse)),'regularization':regularization}


def serialize(weights):return {'kind':'cubic-rgb-v1','features':list(NAMES),'weights':np.asarray(weights).tolist()}


def deserialize(obj):
    if obj.get('kind')!='cubic-rgb-v1' or obj.get('features')!=list(NAMES):
        raise ValueError('Unsupported cubic mapping')
    weights=np.asarray(obj['weights'],dtype=np.float64)
    if weights.shape!=(len(NAMES),3) or not np.isfinite(weights).all():
        raise ValueError('Invalid cubic weights')
    return weights
