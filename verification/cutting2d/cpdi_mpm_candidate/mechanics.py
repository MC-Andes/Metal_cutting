"""Momentum equations in the reset MPM background space.

Q4 operators here supply material quadrature, kinematics, and work-conjugate
loads only. The solved coordinates are those of the background basis R,
restricted by the exact material trace. Dense SVD is a bounded reference
backend; it is not advertised as the scalable final cutting implementation.
"""
import numpy as np
from scipy import sparse
from .transfer import DomainGridTransfer
from .support import BottomTrace


class DenseRange:
    def __init__(self,Z,maximum_columns=512):
        if Z.shape[1]>maximum_columns:raise ValueError('Dense MPM reference size guard exceeded')
        self.Z=Z;U,s,Vt=np.linalg.svd(Z.toarray(),full_matrices=False)
        cutoff=max(Z.shape)*np.finfo(float).eps*s[0] if len(s) else 0.
        keep=s>cutoff;self.U=U[:,keep];self.s=s[keep];self.V=Vt[keep].T
        if np.any(keep) and self.s[-1]<=32*cutoff:
            raise ValueError('MPM energy map has unresolved retained rank near roundoff')
        if np.any(~keep) and np.max(s[~keep])>=cutoff/32:
            raise ValueError('MPM energy map has no clear numerical nullspace separation')
        self.diagnostics=dict(backend='bounded_dense_unsquared_SVD',coordinates=Z.shape[1],rank=int(keep.sum()),
            sigma_max=float(s[0]) if len(s) else 0.,sigma_min_retained=float(self.s[-1]) if len(self.s) else None,
            cutoff=float(cutoff),coordinate_change='invertible column norm scaling; physical mass unchanged')

    def project_coefficients(self,y):
        return self.V@((self.U.T@y)/self.s)

    def dual_coefficients(self,b):
        return self.V@((self.V.T@b)/(self.s*self.s))


class GridMechanics:
    def __init__(self,domain,q,grid_h,fixed,*,origin=None,bottom_trace=None,backend='dense',rank_caches=None):
        self.domain=domain;self.transfer=DomainGridTransfer(domain,q,grid_h,origin)
        if rank_caches is None:rank_caches={}
        if not isinstance(rank_caches,dict):raise ValueError('MPM rank caches must be a dictionary')
        self.nodes=domain.nodes;self.q=self.transfer.q
        fixed=np.asarray(fixed)
        if fixed.dtype!=bool or fixed.shape!=(self.nodes,2):raise ValueError('Boolean material support mask required')
        self.fixed=fixed.copy();bottom=np.flatnonzero(domain.X[:,1]==0.);self.bottom=bottom
        if np.any(fixed[np.setdiff1d(np.arange(self.nodes),bottom)]):raise ValueError('Only the physical bottom support is implemented')
        if any(np.any(fixed[bottom,a]) and not np.all(fixed[bottom,a]) for a in range(2)):
            raise ValueError('Each supported bottom component must cover the entire trace')
        self.trace=bottom_trace or (BottomTrace(q[bottom],grid_h,origin) if np.any(fixed) else None)
        raw_norm=np.sqrt(np.asarray(self.transfer.H.multiply(self.transfer.H).sum(0)).ravel())
        pre_scale=np.ones(self.transfer.nodes);pre_scale[raw_norm>0.]=1/raw_norm[raw_norm>0.]
        free=(sparse.diags(pre_scale,format='csr'),(self.transfer.R@sparse.diags(pre_scale)).tocsr(),{})
        restricted=self.trace.reduction(self.transfer,bottom,pre_scale) if self.trace is not None else free
        self.components=[];cache={}
        for a in range(2):
            supported=bool(np.any(fixed[:,a]))
            if supported in cache:self.components.append(cache[supported]);continue
            L,B,trace_record=restricted if supported else free
            H=(sparse.diags(np.sqrt(domain.mass))@self.transfer.N@B).tocsr()
            norms=np.sqrt(np.asarray(H.multiply(H).sum(0)).ravel());active=norms>0.
            # Only exact zero columns are removed. D changes coordinates,
            # never quadrature mass or the kinetic bilinear form.
            D=1/norms[active];W=(B[:,active]@sparse.diags(D)).tocsr()
            Z=(H[:,active]@sparse.diags(D)).tocsr()
            raw=(self.transfer.R@L[:,active]@sparse.diags(D)).tocsr()
            delta=(raw-W).tocsr()
            column_error=np.sqrt(np.asarray(delta.multiply(delta).sum(0)).ravel())
            column_size=np.sqrt(np.asarray(W.multiply(W).sum(0)).ravel())
            relative=float(np.max(column_error/column_size,initial=0.))
            if relative>1e-11:raise ValueError('Symbolic support cancellation is amplified by grid coordinate scaling')
            trace_record=dict(trace_record,symbolic_zero_scaled_column_relative=relative)
            if backend=='dense':factor=DenseRange(Z)
            elif backend=='sparse_qr':
                from verification.cutting2d.mpm_sparse_energy_candidate.qr import SparseQRProjector
                factor=SparseQRProjector(Z,audit='small_svd',rtol=1e-11)
            elif backend in ('certified_gram','certified_range','certified_range_fast','certified_range_robust'):
                if backend=='certified_gram':
                    from verification.cutting2d.mpm_sparse_energy_candidate.accelerated import CertifiedGramCache as Cache
                elif backend=='certified_range':
                    from verification.cutting2d.mpm_sparse_energy_candidate.reduced import CertifiedRangeGramCache as Cache
                elif backend=='certified_range_fast':
                    from verification.cutting2d.mpm_sparse_energy_candidate.fast_range import CertifiedFastRangeGramCache as Cache
                else:
                    from verification.cutting2d.mpm_sparse_energy_candidate.robust import CertifiedRobustRangeGramCache as Cache
                from .cache_identity import material_identity,coordinate_identity
                key='supported' if supported else 'free'
                if key not in rank_caches:rank_caches[key]=Cache(rtol=1e-11)
                if not isinstance(rank_caches[key],Cache):raise ValueError('Invalid MPM rank cache')
                # The reference quadrature is immutable; geometry changes are
                # bounded against the actual current Z inside the certificate.
                if not hasattr(domain,'_mpm_material_identity'):
                    domain._mpm_material_identity=material_identity(domain)
                token=coordinate_identity(self.transfer,fixed,supported,self.trace,active,
                    material_token=domain._mpm_material_identity)
                factor=rank_caches[key].factor(Z,token)
            else:raise ValueError('Unknown MPM momentum backend')
            data=dict(W=W,Z=Z,L=L[:,active],D=D,factor=factor,trace=trace_record)
            cache[supported]=data;self.components.append(data)
        self.diagnostics=dict(formulation='MPM_shared_domain_B2_four_persistent_GP_v1',
            background_spacing_m=float(grid_h),transfer=self.transfer.diagnostics,
            components=[dict(x['factor'].diagnostics,trace=x['trace']) for x in self.components],
            maximum_equation_residual=0.,maximum_kinematic_residual=0.)

    def _checked(self,a,c):
        data=self.components[a];v=np.asarray(data['W']@c).ravel()
        actual=np.sqrt(self.domain.mass)*(self.transfer.N@v);expected=data['Z']@c
        scale=max(float(np.linalg.norm(actual)),float(np.linalg.norm(expected)),np.finfo(float).tiny)
        relative=float(np.linalg.norm(actual-expected)/scale)
        if not np.isfinite(v).all() or relative>1e-11:raise ValueError('MPM grid and material kinetic fields disagree')
        if np.any(v[self.fixed[:,a]]!=0.):raise ValueError('MPM symbolic material support is not exact')
        self.diagnostics['maximum_kinematic_residual']=max(relative,self.diagnostics['maximum_kinematic_residual'])
        return v

    def project_velocity(self,v):
        v=self.domain._nodal(v,'Carried material velocity');out=np.empty_like(v)
        for a,data in enumerate(self.components):
            y=np.sqrt(self.domain.mass)*(self.transfer.N@v[:,a])
            c=data['factor'].project_coefficients(y);out[:,a]=self._checked(a,c)
            residual=data['Z'].T@(data['Z']@c-y)
            scale=max(float(np.linalg.norm(data['Z'].T@y)),np.finfo(float).tiny)
            relative=float(np.linalg.norm(residual)/scale)
            if relative>1e-11:raise ValueError('MPM particle-to-grid momentum equation failed')
        loss=self.kinetic(v)-self.kinetic(out);delta=self.kinetic(v-out)
        scale=max(self.kinetic(v),self.kinetic(out),np.finfo(float).tiny)
        if abs(loss-delta)>1e-11*scale:raise ValueError('MPM reset projection violates the physical energy identity')
        return out,dict(reset_kinetic_loss_J=loss,reset_orthogonal_loss_J=delta,
            reset_energy_identity_J=loss-delta,reset_heat_added_J=0.)

    def solve(self,force,fixed=None):
        if fixed is not None and not np.array_equal(fixed,self.fixed):raise ValueError('MPM solve support differs from assembled trace')
        force=self.domain._nodal(force,'Material force or impulse');out=np.empty_like(force)
        for a,data in enumerate(self.components):
            b=np.asarray(data['W'].T@force[:,a]).ravel();c=data['factor'].dual_coefficients(b)
            out[:,a]=self._checked(a,c)
            residual=data['Z'].T@(data['Z']@c)-b
            relative=float(np.linalg.norm(residual)/max(float(np.linalg.norm(b)),np.finfo(float).tiny))
            if relative>1e-11:raise ValueError('MPM physical background momentum equation failed')
            self.diagnostics['maximum_equation_residual']=max(relative,self.diagnostics['maximum_equation_residual'])
        return out

    def virtual_work_residual(self,reaction,reference):
        reaction=self.domain._nodal(reaction,'Constraint reaction');reference=self.domain._nodal(reference,'Applied load')
        values=[]
        for a,data in enumerate(self.components):
            numerator=np.linalg.norm(data['W'].T@reaction[:,a])
            # A reference component can be exactly zero; use both physical
            # components for the scale without introducing a force floor.
            denominator=max(np.linalg.norm(data['W'].T@reference),np.finfo(float).tiny)
            values.append(float(numerator/denominator))
        return max(values)

    def multiply(self,v):return self.domain.multiply(v)
    def reaction_envelope(self,new,old,load):
        return np.asarray(abs(self.domain.M)@(abs(new)+abs(old)))+abs(load)
    def kinetic(self,v):return self.domain.kinetic(v)
    def momentum(self,v):return self.domain.momentum(v)
    def angular_momentum(self,q,v):return self.domain.angular_momentum(q,v)
