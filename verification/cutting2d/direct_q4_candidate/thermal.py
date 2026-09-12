"""Local material DGQ1/SIPG heat diffusion with explicit final-state FCT SSPRK2.

Four temperatures per conforming Q4 cell, no global projection and no JC calls.
The high spatial form and its trace penalty follow the archived SIPG candidate;
the explicit time/FCT combination is new and requires its own verification.
No temperature clipping, source remapping, artificial heat or implicit solve.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from .thermal_native import load_native

G = 1/np.sqrt(3.)
GP_SIGNS = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
CORNERS = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
NONCONSTANT = np.array([[1.,1.,1.],[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]])/2


def basis(points, *, gauss=False):
    """Q1 cardinal basis on physical-geometry corners or the four Gauss nodes."""
    points = np.asarray(points, float)
    signs = GP_SIGNS/G if gauss else CORNERS
    a = 1+points[..., None, :]*signs
    values = a[..., 0]*a[..., 1]/4
    derivatives = np.stack((signs[:,0]*a[...,1], signs[:,1]*a[...,0]), axis=-1)/4
    return values, derivatives


N_GP, DN_GP = basis(GP_SIGNS*G)
_, DPHI_GP = basis(GP_SIGNS*G, gauss=True)
_, DN_CORNERS = basis(CORNERS)
FACE_POINTS = np.array([[[t,-1.] for t in (-G,G)], [[1.,t] for t in (-G,G)],
                        [[-t,1.] for t in (-G,G)], [[-1.,-t] for t in (-G,G)]])
N_FACE, DN_FACE = basis(FACE_POINTS)
PHI_FACE, DPHI_FACE = basis(FACE_POINTS, gauss=True)
CORNER_RECONSTRUCTION, _ = basis(CORNERS, gauss=True)


def _readonly(a):
    a = np.array(a, copy=True)
    a.setflags(write=False)
    return a


def _positive_scalar(value, name, allow_zero=False):
    value = float(value)
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if allow_zero else "positive"}')
    return value


def _divergence(i, j, flux, size):
    return np.bincount(i, weights=flux, minlength=size)-np.bincount(j, weights=flux, minlength=size)


@dataclass(frozen=True)
class ThermalStep:
    temperature: np.ndarray
    heat_J: np.ndarray
    diagnostics: dict


class DirectQ4Thermal:
    """Frozen-geometry cache around a local explicit thermal step.

    Corner order: (-,-),(+,-),(+,+),(-,+), counterclockwise. GP order per cell:
    (-g,-g),(-g,+g),(+g,-g),(+g,+g). Temperatures have flat shape (4*cells,).
    ``update_geometry(q)`` validates and assembles on the caller's accepted q.
    ``step`` does not mutate q, T or persistent material state. Sources are
    separate. Reconstructing this object and updating q restores the operator.
    """

    def __init__(self, reference_nodes, cells, width, rho, cp, k, *,
                 periodic_pairs=(), mass_gp=None, reference_volume_gp=None):
        self.X = _readonly(np.asarray(reference_nodes, float))
        ids = np.asarray(cells)
        if (self.X.ndim != 2 or self.X.shape[1] != 2 or len(self.X) < 4 or
                not np.isfinite(self.X).all() or ids.ndim != 2 or ids.shape[1] != 4 or
                not len(ids) or not np.issubdtype(ids.dtype, np.integer) or
                np.any(ids < 0) or np.any(ids >= len(self.X))):
            raise ValueError('Finite 2D reference nodes and valid integer Q4 cells required')
        if np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
            raise ValueError('Each Q4 needs four distinct corner nodes')
        self.cells = _readonly(ids.astype(int))
        self.width = _positive_scalar(width, 'width')
        self.rho = _positive_scalar(rho, 'rho')
        self.cp = _positive_scalar(cp, 'cp')
        self.k = _positive_scalar(k, 'k', allow_zero=True)
        self.point_count = 4*len(self.cells)
        q0, J0, corners0 = self._geometry(self.X)
        geometry_volume = (np.linalg.det(J0)*self.width).ravel()
        self.volume0 = self._physical_array(reference_volume_gp, geometry_volume, 'reference_volume_gp')
        self.mass = self._physical_array(mass_gp, self.rho*self.volume0, 'mass_gp')
        self.capacity = _readonly(self.mass*self.cp)
        self.ids = _readonly(np.repeat(self.cells, 4, axis=0))
        self.N = _readonly(np.tile(N_GP, (len(self.cells), 1)))
        self.reference_points = _readonly(np.einsum('pa,eai->epi',N_GP,q0).reshape(-1,2))
        self._make_faces(periodic_pairs)
        self._q = None
        self._native_stage = None
        self._native_stage_bindings = None
        self._native_stage_components = None
        self.update_geometry(self.X)

    @staticmethod
    def _physical_array(supplied, expected, name):
        result = expected if supplied is None else np.asarray(supplied, float)
        if (result.shape != expected.shape or not np.isfinite(result).all() or
                np.any(result <= 0) or not np.allclose(result, expected, rtol=128*np.finfo(float).eps, atol=0.)):
            raise ValueError(f'{name} must equal the physical reference quadrature without renormalization')
        return _readonly(result)

    def _geometry(self, q):
        q = np.asarray(q, float)
        if q.shape != self.X.shape or not np.isfinite(q).all():
            raise ValueError('Finite q with reference nodal shape required')
        corners = q[self.cells]
        centered = corners-corners[:, :1]
        J = np.einsum('eai,paj->epij', centered, DN_GP)
        corner_J = np.einsum('eai,paj->epij', centered, DN_CORNERS)
        det = np.linalg.det(corner_J)
        if (not np.isfinite(J).all() or not np.isfinite(det).all() or
                np.any(det <= 0) or np.any(np.linalg.det(J) <= 0)):
            raise ValueError('Q4 must have positive Jacobians at all corners and Gauss points')
        return corners, J, det

    def _make_faces(self, periodic_pairs):
        edges = {}
        for e, ids in enumerate(self.cells):
            for f in range(4):
                a,b = int(ids[f]),int(ids[(f+1)%4])
                edges.setdefault(tuple(sorted((a,b))), []).append((e,f,a,b))
        if any(len(v)>2 for v in edges.values()):
            raise ValueError('Nonmanifold Q4 thermal edge')
        pairs=[];exterior=set()
        for entries in edges.values():
            if len(entries)==2:
                a,b=entries
                if (a[2],a[3]) != (b[3],b[2]):
                    raise ValueError('Shared Q4 edge must have opposite cell orientations')
                pairs.append((a[:2],b[:2],False))
            else:
                exterior.add(entries[0][:2])
        for left,right in periodic_pairs:
            left,right=tuple(left),tuple(right)
            if left==right or left not in exterior or right not in exterior:
                raise ValueError('Periodic faces must be distinct exterior edges used once')
            exterior.remove(left);exterior.remove(right)
            pairs.append((left,right,True))
        self.pairs=tuple(pairs);self.exterior=tuple(sorted(exterior))
        self._el=np.array([p[0][0] for p in pairs],int)
        self._fl=np.array([p[0][1] for p in pairs],int)
        self._er=np.array([p[1][0] for p in pairs],int)
        self._fr=np.array([p[1][1] for p in pairs],int)
        self._periodic=np.array([p[2] for p in pairs],bool)
        E=len(self.cells)
        adjacency=sparse.coo_matrix((np.ones(len(pairs)),(self._el,self._er)),shape=(E,E)).tocsr()
        self.component_count,cell_components=connected_components(adjacency,directed=False)
        self.components=_readonly(np.repeat(cell_components,4))
        self._component_first=np.array([np.flatnonzero(self.components==c)[0] for c in range(self.component_count)])
        local_ids=np.arange(self.point_count).reshape(-1,4)
        self._local_rows=np.repeat(local_ids,4,axis=1).ravel()
        self._local_cols=np.tile(local_ids,(1,4)).ravel()
        self._face_ids=np.concatenate((local_ids[self._el],local_ids[self._er]),axis=1)
        self._face_rows=np.repeat(self._face_ids,8,axis=1).ravel()
        self._face_cols=np.tile(self._face_ids,(1,8)).ravel()

    def corner_temperatures(self, temperature):
        T=np.asarray(temperature,float)
        if T.shape!=(self.point_count,) or not np.isfinite(T).all():
            raise ValueError('Finite flat GP temperature vector required')
        return T.reshape(-1,4)@CORNER_RECONSTRUCTION.T

    def update_geometry(self, q):
        """Validate first, then replace the disposable cache atomically."""
        start=time.perf_counter()
        q=np.asarray(q,float)
        if self._q is not None and np.array_equal(q,self._q):
            return self
        corners,J,corner_det=self._geometry(q)
        E=len(self.cells);count=self.point_count
        centered=corners-corners[:,:1]
        base_diagnostics=dict(operator='direct_q4_dg_fct_ssprk2_v1',points=count,cells=E,
            conducting_components=self.component_count,interior_or_periodic_faces=len(self.pairs),
            periodic_faces=int(self._periodic.sum()),neumann_faces=len(self.exterior),
            minimum_corner_jacobian_m2=float(corner_det.min()),
            no_global_projection=True,no_implicit_solve=True,capacity='fixed physical m_gp cp',
            numerical_diffusion='SIPG penalty and limited graph correction; no net heat source',
            gp_bounds_do_not_bound_polynomial=True)
        if self.k==0:
            self.KH=sparse.csr_matrix((count,count));self.KL=self.KH.copy()
            self.D=self.KH.copy();self.stable_dt_s=float('inf')
            self.maximum_forward_euler_dt_s=float('inf')
            self.edge_i=np.empty(0,int);self.edge_j=np.empty(0,int)
            self._low_edge=np.empty(0);self._correction_edge=np.empty(0)
            self._q=q.copy();self.diagnostics={**base_diagnostics,'k0_exact':True,'assembly_seconds':time.perf_counter()-start}
            return self
        invJ=np.linalg.inv(J)
        gradients=np.einsum('pai,epij->epaj',DPHI_GP,invJ)
        volumes=np.linalg.det(J)*self.width
        A=np.einsum('ep,epai,epbi->eab',volumes,gradients,gradients)
        reduced=np.einsum('ai,eab,bj->eij',NONCONSTANT,A,NONCONSTANT)
        chol=np.linalg.cholesky(reduced)
        invchol=np.linalg.inv(chol)
        face_J=np.einsum('eai,fpaj->efpij',centered,DN_FACE)
        if np.any(np.linalg.det(face_J)<=0):
            raise ValueError('Nonpositive face Jacobian')
        face_grad=np.einsum('fpai,efpij->efpaj',DPHI_FACE,np.linalg.inv(face_J))
        edge=np.roll(corners,-1,axis=1)-corners
        length=np.linalg.norm(edge,axis=2)
        if np.any(length<=0) or not np.isfinite(length).all():
            raise ValueError('Zero/nonfinite Q4 edge')
        normals=np.stack((edge[:,:,1],-edge[:,:,0]),axis=-1)/length[:,:,None]
        face_weight=np.broadcast_to(length[:,:,None]*self.width/2,(E,4,2))
        ng=np.einsum('efpai,efi->efpa',face_grad,normals)
        B=np.einsum('efp,efpa,efpb->efab',face_weight,ng,ng)
        Bred=np.einsum('ai,efab,bj->efij',NONCONSTANT,B,NONCONSTANT)
        standard=np.einsum('eia,efab,ejb->efij',invchol,Bred,invchol)
        standard=.5*(standard+standard.swapaxes(-1,-2))
        trace=2*np.linalg.eigvalsh(standard)[:,:,-1]
        margin=np.linalg.eigvalsh(trace[:,:,None,None]*reduced[:,None]-Bred)[:,:,0]
        margin_scale=np.linalg.norm(trace[:,:,None,None]*reduced[:,None],axis=(-2,-1))
        if (not np.isfinite(trace).all() or np.any(trace<=0) or
                np.any(margin < -1e-12*margin_scale)):
            raise ValueError('Discrete local trace coercivity bound failed')
        el,fl,er,fr=self._el,self._fl,self._er,self._fr
        n=normals[el,fl];nr=normals[er,fr]
        w=face_weight[el,fl];wr=face_weight[er,fr][:,::-1]
        if len(el):
            if (np.max(np.linalg.norm(n+nr,axis=1))>1e-10 or
                    np.any(np.max(abs(w-wr),axis=1)>1e-10*np.maximum(w.max(axis=1),wr.max(axis=1)))):
                raise ValueError('Paired thermal faces do not have opposite normals/equal lengths')
            # Ordinary pairs share the same material node IDs exactly. Periodic
            # pairs must represent one translation of both edge endpoints.
            for index in np.flatnonzero(self._periodic):
                left=corners[el[index],[(fl[index]+a)%4 for a in (0,1)]]
                right=corners[er[index],[(fr[index]+a)%4 for a in (1,0)]]
                shift=left-right
                scale=max(float(length[el[index],fl[index]]),float(np.max(np.ptp(q,axis=0))),np.finfo(float).tiny)
                if np.max(abs(shift-shift[0]))>1e-10*scale:
                    raise ValueError('Periodic face endpoints do not agree under a translation')
        jump=np.concatenate((PHI_FACE[fl],-PHI_FACE[fr][:,::-1]),axis=2)
        gl=face_grad[el,fl];gr=face_grad[er,fr][:,::-1]
        flux=.5*self.k*np.concatenate((np.einsum('fpai,fi->fpa',gl,n),np.einsum('fpai,fi->fpa',gr,n)),axis=2)
        penalty=4*self.k*(trace[el,fl]+trace[er,fr])
        cross=np.einsum('fpa,fp,fpb->fab',jump,w,flux)
        face_matrix=-cross-cross.swapaxes(1,2)+np.einsum('fpa,fp,fpb->fab',jump,penalty[:,None]*w,jump)
        rows=np.r_[self._local_rows,self._face_rows]
        cols=np.r_[self._local_cols,self._face_cols]
        values=np.r_[(self.k*A).ravel(),face_matrix.ravel()]
        raw=sparse.coo_matrix((values,(rows,cols)),shape=(count,count)).tocsr()
        norm=float(sparse.linalg.norm(raw));scale=max(norm,np.finfo(float).tiny)
        asym=float(sparse.linalg.norm(raw-raw.T))/scale
        null=float(np.linalg.norm(raw@np.ones(count)))/scale
        if not np.isfinite(raw.data).all() or max(asym,null)>1e-12:
            raise ValueError('Local SIPG symmetry/constant residual failed')
        high=.5*(raw+raw.T)
        original_diagonal=high.diagonal().copy()
        high.setdiag(0.);high.eliminate_zeros()
        high.setdiag(-np.asarray(high.sum(axis=1)).ravel())
        diagonal_change=float(np.linalg.norm(high.diagonal()-original_diagonal))/scale
        edges=sparse.triu(high,k=1).tocoo()
        edge_i,edge_j=edges.row,edges.col
        correction=-np.maximum(edges.data,0.)
        low_edge=edges.data+correction
        # Both endpoint sums are positive; this is the graph degree, not a flux.
        low_diagonal=(np.bincount(edge_i,weights=-low_edge,minlength=count)+
                      np.bincount(edge_j,weights=-low_edge,minlength=count))
        Doff=sparse.coo_matrix((np.r_[correction,correction],
                                (np.r_[edge_i,edge_j],np.r_[edge_j,edge_i])),shape=(count,count)).tocsr()
        D=Doff-sparse.diags(np.asarray(Doff.sum(axis=1)).ravel())
        low=sparse.coo_matrix((np.r_[low_edge,low_edge],
                               (np.r_[edge_i,edge_j],np.r_[edge_j,edge_i])),shape=(count,count)).tocsr()+sparse.diags(low_diagonal)
        low.eliminate_zeros()
        if np.any(low_diagonal<=0):
            raise ValueError('Conducting GP has no resolved graph degree')
        maximum_dt=float(np.min(self.capacity/low_diagonal))
        if not np.isfinite(maximum_dt) or maximum_dt<=0:
            raise ValueError('Thermal CFL is not representable')
        self.KH=high.tocsr();self.KL=low.tocsr();self.D=D.tocsr()
        self.edge_i=_readonly(edge_i);self.edge_j=_readonly(edge_j)
        self._low_edge=_readonly(low_edge);self._correction_edge=_readonly(correction)
        self.maximum_forward_euler_dt_s=maximum_dt
        self.stable_dt_s=.9*maximum_dt
        self._q=q.copy()
        self.diagnostics={**base_diagnostics, 'relative_asymmetry':asym,'relative_constant_residual':null,
            'diagonal_roundoff_change_relative':diagonal_change,
            'trace_minimum_margin':float(margin.min()),
            'trace_constants_minmax':[float(trace.min()),float(trace.max())],
            'penalty_minmax':[float(penalty.min()) if len(penalty) else 0.,float(penalty.max()) if len(penalty) else 0.],
            'matrix_nnz':int(high.nnz),'flux_edges':len(edge_i),
            'maximum_forward_euler_dt_s':maximum_dt,'stable_dt_s':self.stable_dt_s,
            'explicit_cfl_safety_factor':.9,'assembly_seconds':time.perf_counter()-start}
        return self

    def _ensure_native_stage(self):
        """Disposable private copies for the current successfully assembled geometry.

        Publish the native object and all bindings only after construction passes.
        The original coefficients are read-only; intentional in-place mutation of
        private geometry/matrices remains outside the conductor's cache contract.
        """
        names=('edge_i','edge_j','_low_edge','_correction_edge','capacity','components','KH','_q')
        bindings={name:getattr(self,name) for name in names}
        previous=self._native_stage_bindings
        if (previous is not None and self._native_stage_components==self.component_count
                and all(previous[name] is value for name,value in bindings.items())):
            return
        stage=load_native().FrozenStage(self.edge_i,self.edge_j,self._low_edge,
            self._correction_edge,self.capacity,self.components,self.component_count)
        self._native_stage=stage
        self._native_stage_bindings=bindings
        self._native_stage_components=self.component_count

    def _stage(self,u,dt):
        self._ensure_native_stage()
        data=self._native_stage.compute(u,dt)
        C=self.capacity
        low_heat=data['low_heat'];raw_heat=data['raw_heat'];low=data['low']
        lower=data['lower'];upper=data['upper'];raw_flux=data['raw_flux'];alpha=data['alpha']
        limited=data['limited'];correction_heat=data['correction_heat'];heat=data['heat'];new=data['new']
        bound_scale=max(float(np.max(abs(u))),1.)
        tolerance=128*np.finfo(float).eps*bound_scale
        low_defect=max(float(np.max(lower-low)),float(np.max(low-upper)),0.)
        if low_defect>tolerance:
            raise ValueError('Low explicit graph stage violated component bounds')
        if not np.isfinite(alpha).all() or np.any(alpha<0) or np.any(alpha>1):
            raise ValueError('Invalid pairwise FCT coefficient')
        defect=max(float(np.max(lower-new)),float(np.max(new-upper)),0.)
        if not np.isfinite(new).all() or defect>tolerance:
            raise ValueError('Final explicit FCT stage violated component bounds; no clipping')
        scale=float(np.sum(abs(raw_flux)))
        withheld=float(np.sum(abs(raw_flux-limited)))
        return new,heat,dict(minimum_alpha=float(alpha.min()) if len(alpha) else 1.,
            withheld_flux_fraction=withheld/scale if scale else 0.,
            limited_edge_fraction=float(np.mean((alpha<1)&(raw_flux!=0))) if len(alpha) else 0.,
            low_roundoff_bound_defect_K=low_defect,final_roundoff_bound_defect_K=defect,
            full_recovery_residual_J=float(np.linalg.norm(low_heat+raw_heat+dt*(self.KH@u))),
            heat_sum_J=float(heat.sum()),
            high_difference_rms_K=float(np.sqrt(np.mean(((raw_heat-correction_heat)/C)**2))))

    def step(self, temperature, dt):
        T=np.asarray(temperature,float)
        if T.shape!=(self.point_count,) or not np.isfinite(T).all() or np.any(T<=0):
            raise ValueError('One finite positive absolute temperature per GP required')
        dt=_positive_scalar(dt,'dt')
        if self.k>0 and dt>self.stable_dt_s*(1+16*np.finfo(float).eps):
            raise ValueError(f'Thermal explicit CFL exceeded: dt={dt:g}, limit={self.stable_dt_s:g}')
        corner_before=self.corner_temperatures(T)
        offsets=T[self._component_first][self.components]
        u=T-offsets
        if self.k==0 or np.all(u==0):
            return ThermalStep(_readonly(T),_readonly(np.zeros_like(T)),dict(
                operator='direct_q4_dg_fct_ssprk2_v1',k0_exact=self.k==0,
                heat_sum_J=0.,stored_heat_change_J=0.,variance_change_J_K=0.,
                minimum_temperature_K=float(T.min()),maximum_temperature_K=float(T.max()),
                minimum_reconstructed_corner_K=float(corner_before.min()),
                maximum_reconstructed_corner_K=float(corner_before.max()),
                reconstructed_corners_positive=bool(np.all(corner_before>0)),
                maximum_withheld_flux_fraction=0.,stages=[]))
        u1,heat1,diag1=self._stage(u,dt)
        _,heat2,diag2=self._stage(u1,dt)
        heat=.5*(heat1+heat2)
        new=T+heat/self.capacity
        if not np.isfinite(new).all() or np.any(new<=0):
            raise ValueError('Thermal SSPRK2 produced nonpositive temperature; no clipping')
        stored=self.capacity*(new-T)
        component_heat=np.bincount(self.components,weights=heat,minlength=self.component_count)
        mean=np.bincount(self.components,weights=self.capacity*u,minlength=self.component_count)/np.bincount(self.components,weights=self.capacity,minlength=self.component_count)
        centered=u-mean[self.components];centered_new=centered+heat/self.capacity
        before=float(self.capacity@(centered*centered));after=float(self.capacity@(centered_new*centered_new))
        if after>before+1e-11*max(before,np.finfo(float).tiny):
            raise ValueError('Thermal FCT SSPRK2 variance increased beyond roundoff')
        corners=self.corner_temperatures(new)
        diag=dict(operator='direct_q4_dg_fct_ssprk2_v1',dt_s=dt,dt_over_maximum_forward_euler=dt/self.maximum_forward_euler_dt_s,
            heat_sum_J=float(heat.sum()),stored_heat_change_J=float(stored.sum()),
            component_heat_sum_J=component_heat.tolist(),
            maximum_stored_heat_residual_J=float(np.max(abs(stored-heat))),
            variance_before_J_K=before,variance_after_J_K=after,variance_change_J_K=after-before,
            minimum_temperature_K=float(new.min()),maximum_temperature_K=float(new.max()),
            minimum_reconstructed_corner_before_K=float(corner_before.min()),
            minimum_reconstructed_corner_K=float(corners.min()),maximum_reconstructed_corner_K=float(corners.max()),
            reconstructed_corners_positive=bool(np.all(corners>0)),
            maximum_withheld_flux_fraction=max(diag1['withheld_flux_fraction'],diag2['withheld_flux_fraction']),
            stages=[diag1,diag2])
        return ThermalStep(_readonly(new),_readonly(heat),diag)
