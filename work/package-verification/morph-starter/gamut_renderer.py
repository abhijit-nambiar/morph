"""Experimental bounded tonal renderer with cool/warm selective saturation."""
import numpy as np
import selective,solver
NAMES=selective.NAMES
BOUNDS=selective.BOUNDS
vec=selective.vec
recipe=selective.recipe

def render(x,p):
    e,t,tint,contrast,sh,hi,s,cool,warm=p
    y=np.clip(x,1e-6,1-1e-6); lum=np.sum(x*np.array([.2126,.7152,.0722],dtype=np.float32),axis=-1,keepdims=True)
    logit=np.log(y/(1-y));z=logit*(1+contrast)+(e*np.log(2))+np.array([t,tint,-t],dtype=np.float32)*1.3+2*(sh*(1-lum)**2+hi*lum**2)
    out=1/(1+np.exp(-np.clip(z,-30,30)));out=np.where(x<=0,0,np.where(x>=1,1,out))
    r,g,b=x[...,0],x[...,1],x[...,2];wcool=np.clip((b-(r+g)*.5)*4,0,1)[...,None];wwarm=np.clip((r-b)*3,0,1)[...,None]
    center=np.sum(out*np.array([.2126,.7152,.0722],dtype=np.float32),axis=-1,keepdims=True);diff=out-center
    # Limit saturation increase per pixel to the largest in-gamut factor.
    with np.errstate(divide='ignore',invalid='ignore'):
        max_scale=np.min(np.where(diff>1e-8,(1-center)/np.maximum(diff,1e-8),np.where(diff< -1e-8,center/np.maximum(-diff,1e-8),np.inf)),axis=-1,keepdims=True)
    scale=np.minimum(np.maximum(0,1+s+wcool*cool+wwarm*warm),max_scale)
    return np.clip(center+diff*scale,0,1)

def fit(x,y,iterations=45):return solver.fit_gauss_newton(x,y,iterations=iterations,renderer=render,bounds=BOUNDS)
