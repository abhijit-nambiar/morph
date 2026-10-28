"""Damped numerical least-squares solver for the Morph renderer."""
import math
import numpy as np
import morph

def fit_gauss_newton(original,target,iterations=45,seed=0,renderer=None,bounds=None,penalty=None):
    if iterations<1:raise ValueError('iterations must be positive')
    if original.shape!=target.shape:raise ValueError('Images must have equal shape')
    x=original.reshape(-1,3);y=target.reshape(-1,3);rng=np.random.default_rng(seed);idx=rng.choice(len(x),min(len(x),6000),replace=False);x=x[idx];y=y[idx]
    renderer=renderer if renderer is not None else morph.render
    bounds=bounds if bounds is not None else morph.BOUNDS
    p=np.zeros(len(bounds),dtype=np.float64);lo,hi=bounds.T; damping=.01;calls=0
    weights=np.zeros(len(p)) if penalty is None else np.asarray(penalty,dtype=np.float64)
    if weights.shape!=p.shape or np.any(weights<0) or not np.isfinite(weights).all():raise ValueError('Invalid regularization weights')
    def residual(v):return (renderer(x,v)-y).reshape(-1).astype(np.float64)
    r=residual(p);calls+=1;initial=float(np.mean(r*r));loss=initial
    for _ in range(iterations):
        columns=[]
        for i in range(len(p)):
            eps=1e-3;q=p.copy();q[i]=min(hi[i],p[i]+eps);actual=q[i]-p[i]
            if actual==0:q[i]=max(lo[i],p[i]-eps);actual=q[i]-p[i]
            col=(residual(q)-r)/actual;calls+=1;columns.append(col)
        j=np.stack(columns,axis=1); jt=j.T@j/len(r)+np.diag(weights); grad=j.T@r/len(r)+weights*p;success=False
        for _ in range(6):
            try: delta=np.linalg.solve(jt+damping*np.eye(len(p)),grad)
            except np.linalg.LinAlgError:break
            q=np.clip(p-delta,lo,hi); rr=residual(q);calls+=1; trial=float(np.mean(rr*rr)+np.dot(weights,q*q))
            if trial+1e-12<loss:
                improvement=loss-trial;p=q;r=rr;loss=trial;damping=max(damping/3,1e-8);success=True;break
            damping*=10
        if not success or improvement<1e-10 or np.linalg.norm(delta)<1e-5:break
    fit_mse=float(np.mean(r*r))
    return p,{'initial_mse':initial,'fit_mse':fit_mse,'rmse':math.sqrt(fit_mse),'objective':loss,'evaluations':calls,'solver':'damped_gauss_newton'}
