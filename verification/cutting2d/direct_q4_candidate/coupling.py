"""Physical source scatter with an absolute floating-point error bound."""
import numpy as np


def scatter_heat(diffusion_heat,events,fraction,total_friction_heat):
    diffusion=np.asarray(diffusion_heat,float)
    if diffusion.ndim!=1 or len(diffusion)%4 or not np.isfinite(diffusion).all():
        raise ValueError('Finite heat at four material GP per cell required')
    if not 0<=fraction<=1 or not np.isfinite(total_friction_heat) or total_friction_heat<0:
        raise ValueError('Admissible physical friction partition required')
    heat=diffusion.copy();distributed=0.
    for event in events:
        owner=event['owner_cell']
        if isinstance(owner,(bool,np.bool_)) or not isinstance(owner,(int,np.integer)):
            raise ValueError('Material owner must be an integer cell index')
        cell=int(owner);amount=fraction*event['friction_heat_J']
        if cell<0 or 4*cell+4>len(heat) or not np.isfinite(amount) or amount<0:
            raise ValueError('Invalid material owner or physical friction heat')
        heat[4*cell:4*cell+4]+=amount/4.;distributed+=amount
    piece=fraction*total_friction_heat
    # Positive friction summation has no cancellation. This checks the owner
    # scatter separately from the sign-changing conservative conduction field.
    eps=np.finfo(float).eps;n=len(heat)+4*len(events)+8
    gamma=n*eps/(1-n*eps)
    friction_error=distributed-piece
    if abs(friction_error)>gamma*max(abs(distributed)+abs(piece),np.finfo(float).tiny):
        raise ValueError('Boundary friction events and physical heat ledger differ')
    expected=float(diffusion.sum())+piece;error=float(heat.sum())-expected
    magnitude=float(np.sum(abs(diffusion))+np.sum(abs(heat)))+abs(piece)
    bound=gamma*magnitude
    if not np.isfinite(heat).all() or abs(error)>bound:
        raise ValueError('Joint material heat scatter exceeds its floating-point conservation bound')
    return heat,dict(source_scatter_error_J=error,source_scatter_roundoff_bound_J=bound,
        source_scatter_L1_scale_J=magnitude,friction_scatter_error_J=friction_error,
        physical_piece_friction_heat_J=piece,conduction_net_J=float(diffusion.sum()))
