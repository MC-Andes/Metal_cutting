"""Process-independent identity of coordinates used by a rank certificate.

Current positions and numerical values of the lift are deliberately not
hashed: the certified cache bounds the actual change of Z separately. The
token identifies the persistent material rows and the deterministic grid
coordinate ordering, so a changed layout triggers a fresh certificate.
"""
import hashlib
import numpy as np


def _array(hasher,label,value):
    a=np.ascontiguousarray(value)
    hasher.update(label.encode()+b'\0'+a.dtype.str.encode()+b'\0')
    hasher.update(str(a.shape).encode()+b'\0'+a.tobytes())


def material_identity(domain):
    hasher=hashlib.sha256(b'MPM_persistent_Q4_four_GP_reference_v1')
    for name in ('X','cells','ids','N','G_ref','mass','volume0','reference_points'):
        _array(hasher,name,getattr(domain,name))
    return hasher.hexdigest()


def coordinate_identity(transfer,fixed,supported,trace,active,*,material_token=None):
    fixed=np.asarray(fixed);active=np.asarray(active)
    if fixed.dtype!=bool or fixed.shape!=(transfer.domain.nodes,2) or active.dtype!=bool or active.ndim!=1:
        raise ValueError('Explicit Boolean support and active-coordinate ordering required')
    token=material_identity(transfer.domain) if material_token is None else material_token
    if not isinstance(token,str) or len(token)!=64 or any(c not in '0123456789abcdef' for c in token):
        raise ValueError('Stable SHA256 material reference identity required')
    hasher=hashlib.sha256(b'MPM_preD_local_Householder_postD_coordinate_order_v1')
    hasher.update(token.encode());hasher.update(float(transfer.stencil.h).hex().encode())
    hasher.update(b'supported' if supported else b'free')
    for name,value in (('origin',transfer.stencil.origin),('lattice',transfer.stencil.lattice_nodes),
                       ('fixed',fixed),('active_reduced_columns',active)):
        _array(hasher,name,value)
    if supported:
        if trace is None:raise ValueError('Supported coordinate identity requires the exact bottom trace')
        for name,value in (('bottom_x_addresses',trace.xs),('bottom_y_addresses',trace.ys),('bottom_material_positions',trace.q)):
            _array(hasher,name,value)
    return hasher.hexdigest()
