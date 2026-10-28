"""Experimental global color mapping plus fixed radial lighting adjustment."""
import numpy as np
import poly_color

def radial_mask(height,width,row_start=0,full_height=None,full_width=None):
    full_height=full_height or height;full_width=full_width or width
    y=(np.arange(row_start,row_start+height,dtype=np.float32)+.5)/full_height*2-1;x=(np.arange(width,dtype=np.float32)+.5)/full_width*2-1
    radius=np.sqrt(y[:,None]**2+x[None,:]**2);return np.clip(1-radius/1.3,0,1)**2

def features(x,mask):
    return np.concatenate([poly_color.features(x),mask[...,None]*x*(1-x)],axis=-1)

def fit(x,y,regularization=.001,seed=0,max_pixels=30000):
    if x.shape!=y.shape or x.ndim!=3 or x.shape[-1]!=3:raise ValueError('Images must have matching RGB dimensions')
    if not np.isfinite(regularization) or regularization<0:raise ValueError('regularization must be finite and nonnegative')
    mask=radial_mask(x.shape[0],x.shape[1]);rng=np.random.default_rng(seed);idx=rng.choice(x.shape[0]*x.shape[1],min(x.shape[0]*x.shape[1],max_pixels),replace=False);a=features(x,mask).reshape(-1,13)[idx].astype(np.float64);original=x.reshape(-1,3)[idx];target=y.reshape(-1,3)[idx];b=target-original
    mat=a.T@a/len(a)+regularization*np.eye(13);weights=np.linalg.lstsq(mat,a.T@b/len(a),rcond=None)[0];return weights

def render(x,weights):
    weights=np.asarray(weights,dtype=np.float64)
    if weights.shape!=(13,3) or not np.isfinite(weights).all():raise ValueError('Invalid spatial mapping')
    return np.clip(x+features(x,radial_mask(x.shape[0],x.shape[1]))@weights,0,1)

def render_tile(x,weights,row_start,full_shape):
    weights=np.asarray(weights,dtype=np.float64)
    if weights.shape!=(13,3) or not np.isfinite(weights).all():raise ValueError('Invalid spatial mapping')
    mask=radial_mask(x.shape[0],x.shape[1],row_start,full_shape[0],full_shape[1]);return np.clip(x+features(x,mask)@weights,0,1)

render.render_tile=render_tile

def identity():return np.zeros((13,3),dtype=np.float64)
def serialize(w):
    weights=np.asarray(w,dtype=np.float64)
    if weights.shape!=(13,3) or not np.isfinite(weights).all():raise ValueError('Invalid spatial mapping')
    return {'kind':'radial-rgb-v1','weights':weights.tolist()}
def deserialize(obj):
    if obj.get('kind')!='radial-rgb-v1':raise ValueError('Unsupported spatial mapping')
    weights=np.asarray(obj['weights'],dtype=np.float64)
    if weights.shape!=(13,3) or not np.isfinite(weights).all():raise ValueError('Invalid spatial mapping')
    return weights

def fit_with_metrics(x,y,regularization=.001):
    w=fit(x,y,regularization=regularization);error=render(x,w)-y
    return w,{'initial_mse':float(np.mean((x-y)**2)),'fit_mse':float(np.mean(error**2)),'rmse':float(np.sqrt(np.mean(error**2))),'metric_scope':'entire fitting-resolution image','regularization':regularization}

def target(x,settings,strength):
    import morph
    y=morph.pillow_soft_target(x,'neutral',settings);mask=radial_mask(x.shape[0],x.shape[1])[...,None]
    return np.clip(y+strength*mask*x*(1-x),0,1)

def target_tile(x,settings,strength,row_start,full_shape):
    import morph
    y=morph.pillow_soft_target(x,'neutral',settings);mask=radial_mask(x.shape[0],x.shape[1],row_start,full_shape[0],full_shape[1])[...,None]
    return np.clip(y+strength*mask*x*(1-x),0,1)

def gaussian_target(x,settings,strength,center=(.5,.5),radius=(.18,.23),row_start=0,full_shape=None):
    """Independent frame-relative off-center Gaussian light plus Pillow tone/color."""
    import morph
    full_shape=full_shape or x.shape
    yy=(np.arange(row_start,row_start+x.shape[0],dtype=np.float32)+.5)/full_shape[0]
    xx=(np.arange(x.shape[1],dtype=np.float32)+.5)/full_shape[1]
    mask=np.exp(-.5*(((xx[None,:]-center[0])/radius[0])**2+((yy[:,None]-center[1])/radius[1])**2))
    return np.clip(morph.pillow_soft_target(x,'neutral',settings)+strength*mask[...,None]*x*(1-x),0,1)
