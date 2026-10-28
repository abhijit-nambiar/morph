"""Experimental color-selective extension of the global Morph renderer."""
import numpy as np
import morph,solver

NAMES=morph.NAMES+['cool_saturation','warm_saturation']
BOUNDS=np.vstack([morph.BOUNDS,[[-.7,.7],[-.7,.7]]])

def vec(d):
    if any(k not in NAMES for k in d):raise ValueError('Unknown controls: '+str(set(d)-set(NAMES)))
    p=np.array([float(d.get(k,0)) for k in NAMES])
    if not np.isfinite(p).all() or np.any(p<BOUNDS[:,0]) or np.any(p>BOUNDS[:,1]):raise ValueError('Recipe outside bounds or nonfinite')
    return p

def render(x,p):
    y=morph.render(x,p[:7]);r,g,b=x[...,0],x[...,1],x[...,2]
    cool=np.clip((b-(r+g)*.5)*4,0,1)[...,None]
    warm=np.clip((r-b)*3,0,1)[...,None]
    lum=np.sum(y*np.array([.2126,.7152,.0722],dtype=np.float32),axis=-1,keepdims=True)
    scale=1+cool*p[7]+warm*p[8]
    return np.clip(lum+(y-lum)*scale,0,1)

def fit(x,y,iterations=45,regularization=0):
    if regularization<0:raise ValueError('regularization must be nonnegative')
    weights=np.array([0.]*7+[regularization,regularization])
    return solver.fit_gauss_newton(x,y,iterations=iterations,renderer=render,bounds=BOUNDS,penalty=weights)
def recipe(p):return {k:float(v) for k,v in zip(NAMES,p)}
