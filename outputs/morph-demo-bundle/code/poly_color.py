"""Deterministic polynomial color mapping, fitted with regularized least squares."""
import numpy as np

NAMES=('red','green','blue','red2','green2','blue2','red_green','red_blue','green_blue','constant')

def features(x):
    r,g,b=x[...,0],x[...,1],x[...,2]
    return np.stack((r,g,b,r*r,g*g,b*b,r*g,r*b,g*b,np.ones_like(r)),axis=-1)

def identity():return np.zeros((len(NAMES),3),dtype=np.float64)

def render_raw(x,weights):
    w=np.asarray(weights,dtype=np.float64)
    if w.shape!=(len(NAMES),3) or not np.isfinite(w).all():raise ValueError('Invalid polynomial color mapping')
    return x+features(x)@w

def render(x,weights):return np.clip(render_raw(x,weights),0,1)

def out_of_gamut(x,weights):
    stride=max(1,max(x.shape[:2])//512);sample=x[::stride,::stride];raw=render_raw(sample,weights)
    return float(np.mean(np.any((raw<0)|(raw>1),axis=-1)))

def fit(x,y,regularization=.001,sample_size=30000,seed=0):
    if x.shape!=y.shape or x.shape[-1]!=3:raise ValueError('Input and target must have matching RGB shapes')
    if not np.isfinite(regularization) or regularization<0:raise ValueError('regularization must be finite and nonnegative')
    x=x.reshape(-1,3);y=y.reshape(-1,3);rng=np.random.default_rng(seed);idx=rng.choice(len(x),min(len(x),sample_size),replace=False);a=features(x[idx]).astype(np.float64);b=(y[idx]-x[idx]).astype(np.float64);normal=a.T@a/len(a)+regularization*np.eye(len(NAMES));rhs=a.T@b/len(a);weights=np.linalg.lstsq(normal,rhs,rcond=None)[0];error=render(x[idx],weights)-y[idx]
    mse=float(np.mean(error*error));initial=float(np.mean((x[idx]-y[idx])**2))
    return weights,{'sampled_pixels':len(idx),'initial_mse':initial,'fit_mse':mse,'rmse':float(np.sqrt(mse)),'regularization':regularization}

def serialize(w):return {'kind':'polynomial-rgb-v1','features':list(NAMES),'weights':np.asarray(w).tolist()}
def deserialize(obj):
    if obj.get('kind')!='polynomial-rgb-v1' or obj.get('features')!=list(NAMES):raise ValueError('Unsupported polynomial mapping')
    w=np.asarray(obj['weights'],dtype=np.float64)
    if w.shape!=(len(NAMES),3) or not np.isfinite(w).all():raise ValueError('Invalid polynomial mapping weights')
    return w
