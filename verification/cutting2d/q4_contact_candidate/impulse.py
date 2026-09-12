"""Isolated Coulomb impulse on actual Q4 boundary traces and physical mass."""
import numpy as np
from itertools import product
from time import perf_counter


class ContactFailure(ValueError):pass


def fixed_point_residual(W,b,lam,mu,tolerance):
    g=b+W@lam;residual=max(0.,float(np.max(-g[::2],initial=0.)))
    for k in range(len(lam)//2):
        ids=slice(2*k,2*k+2);Wkk=W[ids,ids]
        target=block(Wkk,g[ids]-Wkk@lam[ids],mu,tolerance)
        residual=max(residual,float(np.max(abs(target-lam[ids]))*np.linalg.norm(Wkk)))
    return residual


def resolve_active_branches(W,b,mu,seed,tolerance,max_points=6,cost_seconds=10.):
    """Bounded branch solve; every original row must pass the same residual.

    PGS may converge arbitrarily slowly for distinct, nearly redundant traces.
    This changes the algebraic solve, never the physical constraints. A branch
    has normal velocity zero at closed points, zero tangent velocity at sticking
    points, and lambda_t=-mu*sign(v_t)*lambda_n at sliding points. Open points
    have zero impulse. Solve only unknown closed impulses to avoid mixing unit
    identity rows with the inverse-mass scale. No row regularization is used.
    """
    start=perf_counter();allowed=np.flatnonzero((seed[::2]>0)|(b[::2]<0)).tolist()
    if len(allowed)>max_points:raise ContactFailure('Active-branch point count exceeds bounded prototype scope')
    count=len(b)//2;attempts=0;solutions=[]
    choices=('open','normal') if mu==0 else ('open','stick','plus','minus')
    for modes in product(choices,repeat=len(allowed)):
        if perf_counter()-start>cost_seconds:raise ContactFailure('Active-branch cost guard exceeded')
        attempts+=1;columns=[];equations=[]
        for k,mode in zip(allowed,modes):
            if mode=='open':continue
            normal=np.zeros(2*count);normal[2*k]=1.;equations.append(2*k)
            if mode in ('plus','minus'):normal[2*k+1]=-mu if mode=='plus' else mu
            columns.append(normal)
            if mode=='stick':
                tangent=np.zeros(2*count);tangent[2*k+1]=1.;columns.append(tangent);equations.append(2*k+1)
        lam=np.zeros_like(b)
        if columns:
            T=np.stack(columns,axis=1);A=W[equations]@T;rhs=-b[equations]
            z=np.linalg.lstsq(A,rhs,rcond=None)[0]
            if np.max(abs(A@z-rhs),initial=0.)>tolerance:continue
            lam=T@z
        g=b+W@lam
        # Exact signs are required; an inadmissible multiplier is not clipped.
        if np.min(lam[::2],initial=0.)<0:continue
        mode_ok=True
        for k,mode in zip(allowed,modes):
            ln,lt=lam[2*k:2*k+2];gt=g[2*k+1]
            if mode=='stick' and abs(lt)>mu*ln:mode_ok=False
            if mode=='plus' and gt<-tolerance:mode_ok=False
            if mode=='minus' and gt>tolerance:mode_ok=False
        if not mode_ok:continue
        residual=fixed_point_residual(W,b,lam,mu,tolerance)
        if residual<=tolerance:
            solutions.append((residual,lam,modes))
    if not solutions:raise ContactFailure('No globally verified active branch at original velocity tolerance')
    residual,lam,modes=min(solutions,key=lambda item:item[0])
    return lam,{'method':'bounded_active_branches','enumerated_points':allowed,'branches_attempted':attempts,
        'verified_branches':len(solutions),'selected_modes':list(modes),'all_original_rows_checked':count,
        'velocity_residual':residual,'wall_seconds':perf_counter()-start,
        'uniqueness':'not asserted; every returned state satisfies the original full-row contact law'}


def block(W,b,mu,tolerance):
    if b[0]>=-tolerance:return np.zeros(2)
    if np.linalg.eigvalsh(W)[0]<=0:raise ContactFailure('Closing contact has singular point mobility')
    if W[0,0]<=mu*abs(W[0,1]):raise ContactFailure('Point mobility outside sufficient Coulomb uniqueness condition')
    stick=np.linalg.solve(W,-b)
    if stick[0]>=0 and abs(stick[1])<=mu*stick[0]:return stick
    candidates=[]
    for sign in (-1.,1.):
        normal=-b[0]/(W[0,0]-mu*sign*W[0,1]);lam=normal*np.array([1.,-mu*sign]);g=b+W@lam
        if normal>=0 and sign*g[1]>=-tolerance:candidates.append(lam)
    if not candidates:raise ContactFailure('No resolved local Coulomb branch')
    if len(candidates)>1 and np.linalg.norm(candidates[0]-candidates[1])>tolerance/np.linalg.norm(W):raise ContactFailure('Ambiguous local Coulomb branches')
    return candidates[0]


def solve_impulse(metric,q,points,velocity,tool_velocity,dt,mu=0.,fixed=None,velocity_tolerance=1e-11,max_iterations=5000):
    q=np.asarray(q,float);old=np.asarray(velocity,float);u=np.asarray(tool_velocity,float)
    if dt<=0 or mu<0:raise ValueError('dt>0 and mu>=0 required')
    fixed=np.zeros_like(old,dtype=bool) if fixed is None else np.asarray(fixed,bool)
    if np.max(abs(old[fixed]),initial=0)>velocity_tolerance:raise ContactFailure('Input fixed velocities must already be zero')
    spaces=[]
    for d in range(2):
        free=np.flatnonzero(~fixed[:,d])
        H=metric.L.T[:,free];spaces.append(np.linalg.qr(H,mode='reduced')[0])
    def project(y):return np.stack([Q@(Q.T@y[:,d]) for d,Q in enumerate(spaces)],axis=1)
    y0=metric.pack(old);y=y0.copy();B=[];frames=[];gaps=[]
    for p in points:
        n=p['normal'];frame=np.array([n,[-n[1],n[0]]]);base=metric.force(p['coefficients'])
        B.extend(np.outer(base,d).ravel() for d in frame);frames.extend(frame);gaps.extend((p['gap']/dt,0.))
    B=np.asarray(B);frames=np.asarray(frames);gaps=np.asarray(gaps)
    R=np.stack([project(row.reshape(y.shape)).ravel() for row in B],axis=1)
    W=B@R;scale=max(float(np.max(abs(W))),np.finfo(float).tiny)
    if np.max(abs(W-W.T))/scale>1e-12:raise ContactFailure('Physical mobility is not symmetric')
    lam=np.zeros(len(B));bias=B@y0.ravel()-frames@u+gaps
    residual=0.;iterations=0;resolution={'method':'PGS'}
    for it in range(max_iterations):
        iterations=it+1
        for k in range(len(points)):
            ids=slice(2*k,2*k+2);Wkk=W[ids,ids];free_bias=B[ids]@y.ravel()-frames[ids]@u+gaps[ids]-Wkk@lam[ids]
            new=block(Wkk,free_bias,mu,velocity_tolerance);y+=(R[:,ids]@(new-lam[ids])).reshape(y.shape);lam[ids]=new
        residual=0.
        for k in range(len(points)):
            ids=slice(2*k,2*k+2);Wkk=W[ids,ids];g=B[ids]@y.ravel()-frames[ids]@u+gaps[ids]
            target=block(Wkk,g-Wkk@lam[ids],mu,velocity_tolerance)
            residual=max(residual,float(np.max(abs(target-lam[ids]))*np.linalg.norm(Wkk)),max(-g[0],0.))
        if residual<=velocity_tolerance:break
    else:
        stalled=residual
        lam,resolution=resolve_active_branches(W,bias,mu,lam,velocity_tolerance)
        resolution['PGS_stalled_velocity_residual']=stalled
        y=y0+(R@lam).reshape(y.shape)
        residual=fixed_point_residual(W,bias,lam,mu,velocity_tolerance)
        if residual>velocity_tolerance:raise ContactFailure('Active-branch reconstruction failed original residual')
    v=metric.unpack(y);dy=y-y0;nodal=np.zeros_like(v);heat=0.;gap_work=0.;friction_roundoff=0.;events=[]
    for k,p in enumerate(points):
        ln,lt=lam[2*k:2*k+2];frame=frames[2*k:2*k+2];impulse=frame.T@np.array([ln,lt]);relative=frame@(p['coefficients']@v-u)
        if ln>0 and p['feature'].startswith('posterior_vertex'):raise ContactFailure('Active nonsmooth posterior vertex excluded')
        raw_heat=-lt*relative[1];physical_heat=0. if abs(relative[1])<=velocity_tolerance else raw_heat
        if physical_heat<-velocity_tolerance*max(abs(lt),1e-300):raise ContactFailure('Resolved negative Coulomb dissipation')
        heat+=physical_heat;friction_roundoff+=physical_heat-raw_heat;gap_work+=ln*p['gap']/dt
        nodal+=p['coefficients'][:,None]*impulse
        if ln>0:events.append({'event_id':f'point-{p["point_id"]}','point_index':k,'point_id':p['point_id'],
            **{key:p[key] for key in ('edge_id','edge','edge_parameter','owner_cell','local_edge','trace_owners')},
            'parameter':p['parameter'],'position':p['position'],'feature':p['feature'],'normal':p['normal'],
            'lambda':np.array([ln,lt]),'relative_velocity':relative,'gap':p['gap'],'friction_heat_J':float(physical_heat)})
    reaction=metric.M@(v-old)-nodal;impulse=nodal.sum(0);work=float(impulse@u)
    K=lambda a:float(.5*np.sum(a*a))
    identity=K(y)-K(y0)-work+heat+gap_work+K(dy)-friction_roundoff
    angular=lambda f:float(np.sum(q[:,0]*f[:,1]-q[:,1]*f[:,0]))
    full_slack=B@y.ravel()-frames@u+gaps
    return v,{'y_before':y0,'y_after':y,'B':B,'response':R,'mobility':W,'bias':bias,'lambda':lam,
        'kinetic_before':K(y0),'kinetic_after':K(y),'tool_impulse':impulse,'tool_angular_impulse':angular(nodal),
        'support_impulse':reaction.sum(0),'support_angular_impulse':angular(reaction),
        'free_support_reaction_norm':float(np.linalg.norm(reaction[~fixed])),'tool_work':work,
        'friction_heat':float(heat),'gap_work':float(gap_work),'contact_projection_loss':K(dy),
        'friction_roundoff_defect':float(friction_roundoff),'energy_identity':float(identity),
        'minimum_final_linear_slack':float(full_slack[::2].min()),'iterations':iterations,'velocity_residual':residual,'resolution':resolution,
        'momentum_identity':np.sum(metric.M@(v-old),axis=0)-impulse-reaction.sum(0),
        'angular_identity':angular(metric.M@(v-old))-angular(nodal)-angular(reaction),
        'nodal_contact_impulse':nodal,'support_reaction':reaction,'events':events,
        'scope':'Isolated impulse on frozen geometry; all real point rows retained. No position update or finite-step contact acceptance.'}
