"""Scalable, isolated 2D nonassociated Coulomb impulse solve.

Only the algebraic contact solve is changed. W, bias, physical impulses, all
rows and the historical velocity residual are retained. No mass change, row
removal, impulse clipping, compliance, cone-associated normal law or relaxed
acceptance tolerance is introduced. This is a candidate, not a production hook.
"""
from time import perf_counter
import numpy as np
from verification.cutting2d.q4_contact_candidate.impulse import ContactFailure, fixed_point_residual


class CoulombAmbiguity(ContactFailure):
    """Two independently verified roots; do not silently choose one later."""


def _map(A,b,p,mu,jacobian=False):
    """Dimensionally scaled cylinder projection; residual and p are m/s."""
    g=b+A@p; z=p-g; n=p[::2]; positive=np.maximum(n,0.)
    f=np.empty_like(p); f[::2]=n-np.maximum(z[::2],0.)
    f[1::2]=p[1::2]-np.clip(z[1::2],-mu*positive,mu*positive)
    if not jacobian:return f
    J=np.eye(len(p));active=z[::2]>0
    normal=np.flatnonzero(active)*2;J[normal]=A[normal]
    if mu:
        for k in range(len(p)//2):
            i=2*k;limit=mu*positive[k]
            if z[i+1]<-limit:J[i+1,i]=mu if n[k]>0 else 0.
            elif z[i+1]>limit:J[i+1,i]=-mu if n[k]>0 else 0.
            else:J[i+1]=A[i+1]
    return f,J


def _modes(A,b,p,mu):
    g=b+A@p;z=p-g;modes=[]
    for k in range(len(p)//2):
        i=2*k
        if z[i]<=0:modes.append('open')
        elif mu==0:modes.append('normal')
        elif z[i+1]>mu*max(p[i],0.):modes.append('minus')
        elif z[i+1]<-mu*max(p[i],0.):modes.append('plus')
        else:modes.append('stick')
    return modes


def _branch_matrix(A,b,modes,mu):
    J=np.eye(len(b));rhs=np.zeros_like(b)
    for k,mode in enumerate(modes):
        i=2*k
        if mode=='open':continue
        J[i]=A[i];rhs[i]=-b[i]
        if mode=='stick':J[i+1]=A[i+1];rhs[i+1]=-b[i+1]
        elif mode=='plus':J[i+1,i]=mu
        elif mode=='minus':J[i+1,i]=-mu
    return J,rhs


def _linear(J,rhs):
    """Invertible diagonal coordinates only. Never a pseudoinverse/filter."""
    rows=np.max(abs(J),axis=1)
    if np.any(rows==0):raise ContactFailure('Singular Coulomb branch: zero equation')
    B=J/rows[:,None];columns=np.max(abs(B),axis=0)
    if np.any(columns==0):raise ContactFailure('Singular Coulomb branch: unconstrained impulse')
    Z=B/columns;singular=np.linalg.svd(Z,compute_uv=False)
    threshold=32*np.finfo(float).eps*len(J)*singular[0]
    if singular[-1]<=threshold:
        raise ContactFailure('Singular or unresolved Coulomb branch; no regularization or impulse gauge is selected')
    answer=np.linalg.solve(Z,rhs/rows)/columns
    # Iterative refinement changes the linear solve only, never W or its rank.
    for _ in range(2):
        remainder=rhs-J@answer
        correction=np.linalg.solve(Z,remainder/rows)/columns
        candidate=answer+correction
        if np.linalg.norm(rhs-J@candidate,np.inf)>=np.linalg.norm(remainder,np.inf):break
        answer=candidate
    return answer,dict(branch_condition=float(singular[0]/singular[-1]),branch_sigma_min=float(singular[-1]),branch_rank_threshold=float(threshold))


def _physical_checks(W,b,lam,mu,tol):
    # Exact impulse sign/cone checks: no force-derived tolerance or clipping.
    if np.any(lam[::2]<0) or np.any(abs(lam[1::2])>mu*lam[::2]):return None
    g=b+W@lam
    if np.min(g[::2],initial=0)<-tol:return None
    closed=lam[::2]>0
    if np.max(abs(g[::2][closed]),initial=0)>tol:return None
    stick=closed & (abs(lam[1::2])<mu*lam[::2])
    if mu and np.max(abs(g[1::2][stick]),initial=0)>tol:return None
    if np.any((g[1::2]>tol)&(lam[1::2]>0)) or np.any((g[1::2]<-tol)&(lam[1::2]<0)):return None
    try:residual=fixed_point_residual(W,b,lam,mu,tol)
    except ContactFailure:return None
    if residual>tol:return None
    return dict(velocity_residual=float(residual),minimum_normal_slack=float(np.min(g[::2],initial=np.inf)),
        impulse_cone_exact=True,all_original_rows_checked=len(b)//2)


def _polish(W,A,b,p,scales,mu,tol):
    modes=_modes(A,b,p,mu);J,rhs=_branch_matrix(A,b,modes,mu)
    value,diag=_linear(J,rhs);lam=value/scales
    # Sliding/open relations are constructed from the branch coordinates,
    # avoiding a second rounded evaluation of the same exact constitutive law.
    for k,mode in enumerate(modes):
        i=2*k
        if mode=='open':lam[i:i+2]=0.
        elif mode=='normal':lam[i+1]=0.
        elif mode=='plus':lam[i+1]=-mu*lam[i]
        elif mode=='minus':lam[i+1]=mu*lam[i]
    checks=_physical_checks(W,b,lam,mu,tol)
    if checks is None:return None
    # Weakly active normal constraints must be included in a degeneracy audit.
    # Otherwise a duplicate row can appear as a nonsingular open/closed branch
    # even though infinitely many impulse splits produce the same velocity.
    g=b+W@lam;audit=list(modes)
    for k,mode in enumerate(audit):
        if mode=='open' and abs(g[2*k])<=tol:
            audit[k]='normal' if mu==0 else ('stick' if abs(g[2*k+1])<=tol else ('plus' if g[2*k+1]>0 else 'minus'))
    auditJ,_=_branch_matrix(A,b,audit,mu);_,auditdiag=_linear(auditJ,np.zeros_like(b))
    return lam,{**diag,**checks,'selected_modes':modes,'weak_activity_audit_condition':auditdiag['branch_condition']}


def _one_seed(W,A,b,seed,scales,mu,tol,max_iterations,deadline):
    p=seed*scales;last_error='No verified branch';trace=[]
    for iteration in range(max_iterations+1):
        if perf_counter()>deadline:raise ContactFailure('Coulomb semismooth solve exceeded bounded wall budget')
        try:
            polished=_polish(W,A,b,p,scales,mu,tol)
            if polished is not None:return polished[0],{**polished[1],'iterations':iteration,'merit_trace':trace}
        except ContactFailure as exc:last_error=str(exc)
        f,J=_map(A,b,p,mu,True);merit=float(f@f);trace.append(float(np.max(abs(f),initial=0)))
        if iteration==max_iterations:break
        try:direction,_=_linear(J,-f)
        except ContactFailure as exc:
            last_error=str(exc);direction=-J.T@f
        slope=float(f@(J@direction))
        if not np.isfinite(slope) or slope>=0:direction=-J.T@f;slope=-float(direction@direction)
        if not np.isfinite(slope) or slope>=0:break
        accepted=False
        for k in range(45):
            step=2.**(-k);trial=p+step*direction;trialf=_map(A,b,trial,mu)
            if np.all(np.isfinite(trialf)) and float(trialf@trialf)<=merit+2e-4*step*slope:
                p=trial;accepted=True;break
        if not accepted:break
    raise ContactFailure('No verified Coulomb root after semismooth iterations: '+last_error)


def resolve_coulomb(W,b,mu,seed,tolerance=1e-11,*,max_iterations=80,cost_seconds=10.,check_multiple=True):
    """Return lambda_Ns, diagnostics. Same positional API as branch fallback.

    The final physical law uses every original row at the original tolerance.
    Multiple deterministic starts detect additional roots but are not a proof
    of global uniqueness for arbitrary coupled Coulomb systems. Singular final
    branches/weak-contact impulse gauges are rejected, not regularized.
    """
    start=perf_counter();W=np.asarray(W,float);b=np.asarray(b,float);seed=np.asarray(seed,float)
    if b.ndim!=1 or len(b)%2 or W.shape!=(len(b),len(b)) or seed.shape!=b.shape:raise ValueError('Paired normal/tangent dimensions required')
    if not all(np.all(np.isfinite(a)) for a in (W,b,seed)) or not np.isfinite(mu) or mu<0:raise ValueError('Finite mobility, load and nonnegative friction required')
    if (not np.isfinite(tolerance) or tolerance<=0 or not np.isfinite(cost_seconds) or
        cost_seconds<=0 or not isinstance(max_iterations,(int,np.integer)) or max_iterations<1):
        raise ValueError('Positive finite tolerance/wall budget and integer iteration budget required')
    if len(b)==0:return seed.copy(),dict(method='semismooth_cylinder_Coulomb2D',velocity_residual=0.,all_original_rows_checked=0)
    scale=max(float(np.max(abs(W))),np.finfo(float).tiny)
    if np.max(abs(W-W.T))/scale>1e-12:raise ContactFailure('Coulomb mobility lost physical symmetry')
    pairscales=[]
    for k in range(len(b)//2):
        pair=W[2*k:2*k+2,2*k:2*k+2]
        if np.linalg.eigvalsh(pair)[0]<=0:raise ContactFailure('Singular point mobility')
        if pair[0,0]<=mu*abs(pair[0,1]):raise ContactFailure('Point mobility outside sufficient Coulomb uniqueness condition')
        pairscales.append(np.linalg.norm(pair,2))
    scales=np.repeat(pairscales,2);A=W/scales[None,:]
    starts=[seed]
    if check_multiple:
        starts.extend([np.zeros_like(b)])
        # Tangentially opposite cone guesses test physically different modes.
        normal=np.maximum(-b[::2],0.)/np.asarray(pairscales)
        for sign in (-1.,1.):
            guess=np.empty_like(b);guess[::2]=normal;guess[1::2]=sign*mu*normal;starts.append(guess)
    roots=[];failures=[];diagnostics=[]
    for guess in starts:
        try:
            lam,diag=_one_seed(W,A,b,guess,scales,mu,tolerance,max_iterations,start+cost_seconds)
            # Compare impulse, not only its possibly identical induced velocity.
            if roots and np.max(abs((lam-roots[0])*scales))>8*tolerance:
                raise CoulombAmbiguity('Multiple verified Coulomb impulse solutions detected')
            roots.append(lam);diagnostics.append(diag)
        except ContactFailure as exc:
            if isinstance(exc,CoulombAmbiguity):raise
            failures.append(str(exc))
    if not roots:raise ContactFailure('Coulomb solve failed for every deterministic start: '+'; '.join(failures))
    best=min(range(len(roots)),key=lambda i:diagnostics[i]['velocity_residual'])
    return roots[best],dict(method='semismooth_cylinder_Coulomb2D',**diagnostics[best],starts_attempted=len(starts),
        roots_verified=len(roots),failed_starts=failures,wall_seconds=perf_counter()-start,
        uniqueness=('Nonsingular returned branch and weak-contact audit; '+
            ('deterministic alternate starts checked; ' if check_multiple else 'one start, alternate roots not searched; ')+
            'global uniqueness not asserted'),
        no_regularization=True,no_constraint_removal=True,acceptance_tolerance_m_s=float(tolerance))
