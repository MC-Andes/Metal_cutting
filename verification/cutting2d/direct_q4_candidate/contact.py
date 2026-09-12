"""Coulomb boundary impulses in the complete consistent-mass Q4 space.

All geometric constraints are checked. The active-set screening only avoids
mass solves for inactive rows; it does not truncate the physical mobility.
"""
import numpy as np
from scipy import sparse
from verification.cutting2d.q4_contact_candidate.impulse import (
    ContactFailure, block, fixed_point_residual, resolve_active_branches)


def solve_contact(model, q, points, velocity, tool_velocity, dt, fixed, *,
                  mu=.35, velocity_tolerance=1e-11, max_iterations=1000):
    q=np.asarray(q,float); old=np.asarray(velocity,float); u=np.asarray(tool_velocity,float)
    if dt<=0 or not np.isfinite(dt) or mu<0 or not np.isfinite(mu):
        raise ValueError('Positive finite dt and nonnegative mu required')
    if old.shape!=q.shape or fixed.shape!=q.shape or np.any(old[fixed]!=0):
        raise ContactFailure('Input velocity must satisfy the exact stationary material support')
    n=len(points); frames=np.empty((2*n,2)); gaps=np.zeros(2*n)
    rows=[];cols=[];data=[]
    for k,p in enumerate(points):
        normal=np.asarray(p['normal']); frames[2*k:2*k+2]=[normal,[-normal[1],normal[0]]]
        if abs(np.linalg.norm(normal)-1)>1e-12:raise ContactFailure('Contact normal is not normalized')
        a=np.asarray(p['coefficients']); ids=np.flatnonzero(a)
        if a.shape!=(len(q),) or abs(a.sum()-1)>1e-12 or np.min(a)<0:
            raise ContactFailure('Contact requires a convex real material edge trace')
        for direction in range(2):
            for node in ids:
                for component in range(2):
                    rows.append(2*k+direction);cols.append(2*node+component)
                    data.append(a[node]*frames[2*k+direction,component])
        gaps[2*k]=p['gap']/dt
    B=sparse.csr_matrix((data,(rows,cols)),shape=(2*n,old.size))
    bias=np.asarray(B@old.ravel())-frames@u+gaps
    active=set(np.flatnonzero(bias[::2]<-velocity_tolerance).tolist())
    lam=np.zeros(2*n); v=old.copy(); iterations=0; passes=0; residual=0.; resolution={'method':'empty'}
    responses={}
    while active:
        passes+=1; point_ids=np.array(sorted(active)); ids=(2*point_ids[:,None]+[0,1]).ravel()
        for i in ids:
            if int(i) not in responses:
                responses[int(i)]=model.solve(B.getrow(i).toarray().reshape(q.shape),fixed).ravel()
        R=np.stack([responses[int(i)] for i in ids],axis=1)
        Ba=B[ids]; W=np.asarray(Ba@R); ba=bias[ids]
        scale=max(float(np.max(abs(W),initial=0)),np.finfo(float).tiny)
        if np.max(abs(W-W.T),initial=0)/scale>1e-12:
            raise ContactFailure('Consistent physical contact mobility lost symmetry')
        impulse=lam[ids].copy(); slack=ba+W@impulse; resolution={'method':'PGS'}
        for iteration in range(max_iterations):
            iterations+=1
            for k in range(len(point_ids)):
                pair=slice(2*k,2*k+2); diagonal=W[pair,pair]
                new=block(diagonal,slack[pair]-diagonal@impulse[pair],mu,velocity_tolerance)
                difference=new-impulse[pair];slack+=W[:,pair]@difference;impulse[pair]=new
            residual=fixed_point_residual(W,ba,impulse,mu,velocity_tolerance)
            if residual<=velocity_tolerance:break
        else:
            impulse,resolution=resolve_active_branches(W,ba,mu,impulse,velocity_tolerance)
        lam[ids]=impulse;v=old+(R@impulse).reshape(q.shape)
        actual=np.asarray(B@v.ravel())-frames@u+gaps
        residual=fixed_point_residual(W,actual[ids]-W@impulse,impulse,mu,velocity_tolerance)
        if residual>velocity_tolerance:raise ContactFailure('Reconstructed physical contact velocity fails tolerance')
        newly=set(np.flatnonzero(actual[::2]<-velocity_tolerance).tolist())-active
        if not newly:break
        active.update(newly)
        if passes>n:raise ContactFailure('Finite contact active-set screening failed')
    slack=np.asarray(B@v.ravel())-frames@u+gaps
    if np.min(slack[::2],initial=np.inf)<-velocity_tolerance:
        raise ContactFailure('An inactive physical contact constraint is closing')
    nodal=np.asarray(B.T@lam).reshape(q.shape)
    reaction=model.multiply(v-old)-nodal
    reaction_error=float(np.linalg.norm(reaction[~fixed]))
    if reaction_error>1e-11*max(float(np.linalg.norm(nodal)),np.finfo(float).tiny):
        raise ContactFailure('Contact support reaction has nonzero free virtual work')
    events=[];heat=0.;gap_work=0.;roundoff=0.
    for k,p in enumerate(points):
        ln,lt=lam[2*k:2*k+2]
        if ln<=0:continue
        if p['feature'].startswith('posterior_vertex'):
            raise ContactFailure('Active nonsmooth posterior vertex is unsupported')
        relative=frames[2*k:2*k+2]@(p['coefficients']@v-u)
        raw=-lt*relative[1];physical=0. if abs(relative[1])<=velocity_tolerance else raw
        if physical < -velocity_tolerance*max(abs(lt),np.finfo(float).tiny):
            raise ContactFailure('Resolved Coulomb heat is negative')
        heat+=physical;roundoff+=physical-raw;gap_work+=ln*p['gap']/dt
        events.append(dict(point_id=p['point_id'],owner_cell=p['owner_cell'],edge=p['edge'],
            position=p['position'],normal=p['normal'],feature=p['feature'],
            impulse=frames[2*k:2*k+2].T@np.array([ln,lt]),lambda_nt=np.array([ln,lt]),
            relative_velocity=relative,gap=p['gap'],friction_heat_J=float(physical)))
    impulse=nodal.sum(0); work=float(impulse@u);delta=v-old
    angular=lambda f:float(np.sum(q[:,0]*f[:,1]-q[:,1]*f[:,0]))
    identity=model.kinetic(v)-model.kinetic(old)-work+heat+gap_work+model.kinetic(delta)-roundoff
    return v,dict(tool_impulse=impulse,tool_angular_impulse=angular(nodal),
        support_impulse=reaction.sum(0),support_angular_impulse=angular(reaction),
        nodal_contact_impulse=nodal,support_reaction=reaction,tool_work=work,
        friction_heat=float(heat),gap_work=float(gap_work),contact_projection_loss=model.kinetic(delta),
        friction_roundoff_defect=float(roundoff),energy_identity=float(identity),
        momentum_identity=model.momentum(delta)-impulse-reaction.sum(0),
        angular_identity=angular(model.multiply(delta))-angular(nodal)-angular(reaction),
        velocity_residual=float(residual),minimum_final_linear_slack=float(np.min(slack[::2],initial=np.inf)),
        free_reaction_norm=reaction_error,iterations=iterations,active_set_passes=passes,
        constraints=n,mass_response_rows=len(responses),resolution=resolution,events=events)
