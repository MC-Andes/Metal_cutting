"""Candidate MPM grid basis interpolated over shared convected Q4 domains.

R[a,i]=B2_i(q_corner[a]); S[p,i]=sum_a N_Q4[p,a] R[a,i].
Unknown velocities and momentum equations belong to the reset background
grid. Four persistent material quadrature states per domain provide physical
mass/history. This module builds transfers only; it is not a cutting solver
and does not claim to reproduce the published one-state CPDI2 algorithm.
"""
import numpy as np
from scipy import sparse
from cutting2d.transfer import build_stencil


class DomainGridTransfer:
    def __init__(self,material_domain,q,grid_h,origin=None):
        self.domain=material_domain;self.q=material_domain._nodal(q,'Domain corner positions').copy()
        corners=material_domain.corner_jacobians(self.q)
        if not np.isfinite(corners).all() or np.any(corners<=0):
            raise ValueError('Convected material domain has a nonpositive corner Jacobian')
        self.stencil=build_stencil(self.q,grid_h,origin)
        n=material_domain.nodes;P=material_domain.point_count;self.nodes=len(self.stencil.nodes)
        rows=np.repeat(np.arange(n),9)
        self.R=sparse.coo_matrix((self.stencil.weights.ravel(),(rows,self.stencil.indices.ravel())),shape=(n,self.nodes)).tocsr()
        self.N=sparse.coo_matrix((material_domain.N.ravel(),(np.repeat(np.arange(P),4),material_domain.ids.ravel())),shape=(P,n)).tocsr()
        self.S=(self.N@self.R).tocsr()
        self.H=(sparse.diags(np.sqrt(material_domain.mass))@self.S).tocsr()
        self.mass_matrix=(self.H.T@self.H).tocsr()
        self.grid_mass=np.asarray(self.S.T@material_domain.mass).ravel()
        self.active=self.grid_mass>0  # Structural zero only; never a capacity cutoff.
        self.material_points=np.asarray(self.N@self.q)
        unity=float(np.max(abs(np.asarray(self.S.sum(1)).ravel()-1.)))
        affine=float(np.max(abs(self.S@self.stencil.nodes-self.material_points)))
        scale=max(float(np.max(abs(self.q))),float(grid_h))
        if unity>128*np.finfo(float).eps or affine>128*np.finfo(float).eps*scale:
            raise ValueError('Interpolated grid partition or affine completeness failed')
        total=float(self.mass_matrix.sum());mass=float(material_domain.mass.sum())
        if abs(total-mass)>128*np.finfo(float).eps*mass:
            raise ValueError('Grid consistent mass differs from physical material quadrature')
        self.diagnostics=dict(operator='mpm_shared_domain_b2_four_gp_v1',grid_nodes=self.nodes,
            active_nodes=int(self.active.sum()),domain_corners=n,material_points=P,
            partition_error=unity,affine_coordinate_error_m=affine,mass_kg=mass,
            mass_identity_error_kg=total-mass,mass_nnz=self.mass_matrix.nnz,energy_map_nnz=self.H.nnz,
            no_mass_lumping=True,no_mass_scaling=True,no_history_on_background_grid=True,
            stage='Frozen pre-advection grid basis; rebuild from accepted corner positions each step.')

    def _grid(self,v):
        v=np.asarray(v,float)
        if v.shape!=(self.nodes,2) or not np.isfinite(v).all():raise ValueError('Finite background (nodes,2) field required')
        return v

    def corner_velocity(self,grid_velocity):return np.asarray(self.R@self._grid(grid_velocity))
    def particle_velocity(self,grid_velocity):return np.asarray(self.S@self._grid(grid_velocity))

    def particle_to_grid_momentum(self,material_corner_velocity):
        v=self.domain._nodal(material_corner_velocity,'Carried material corner velocity')
        return np.asarray(self.S.T@(self.domain.mass[:,None]*(self.N@v)))

    def internal_force(self,first_piola):
        return np.asarray(self.R.T@self.domain.force_from_piola(first_piola))

    def deformation_rate_reference(self,grid_velocity):
        v=self.corner_velocity(grid_velocity)
        return np.einsum('pai,paj->pij',v[self.domain.ids],self.domain.G_ref)

    def kinetic_grid(self,grid_velocity):
        v=self._grid(grid_velocity);y=self.H@v
        return .5*float(np.sum(y*y))

    def boundary_rows(self,material_nodes):
        nodes=np.asarray(material_nodes)
        if nodes.dtype.kind not in 'iu' or nodes.ndim!=1 or np.any(nodes<0) or np.any(nodes>=self.domain.nodes):
            raise ValueError('Integer material boundary node indices required')
        return self.R[nodes].tocsr()
