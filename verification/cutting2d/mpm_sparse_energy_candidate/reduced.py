"""Current independent-column Gram with COMPLETE certified range checks.

This is a separate candidate from full-rank-only accelerated.py. A QR pivot
set is only a proposal: a full left-inverse/dependency certificate must first
accept it. Reuse bounds the SAME retained columns by Weyl and reconstructs
EVERY dependent column of the current map with outward product error bounds.
No dependent physical equation is dropped: all RHS are checked against Z.T Z.
"""
import copy
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from cutting2d.energy import ProjectorFailure
from .qr import SparseQRProjector
from .rank_certificate import certify,up,down,norm_upper,product_and_error,spectral_scale_bounds,EPS,ETA
from .accelerated import _valid_identity,_payload_digest,_proof_bits,_restore_csc

SNAPSHOT_SCHEMA='mpm-certified-range-gram-cache-snapshot-v1'


def _current_range_proof(Z,retained,reference_Z,certificate,lu,*,block_size):
    """Existence bounds: computed T can be arbitrary; no solve accuracy premise."""
    started=time.perf_counter();A=Z[:,retained].tocsc();Aref=reference_Z[:,retained]
    delta=(A-Aref).tocoo();roundoff=(abs(A)+abs(Aref)).tocoo()
    change=up(norm_upper(delta.data)+EPS*norm_upper(roundoff.data))
    minimum=down(certificate['sigma_r_lower']-change)
    dependent=np.setdiff1d(np.arange(Z.shape[1]),retained);remainder2=0.;peak=0
    for first in range(0,len(dependent),block_size):
        ids=dependent[first:first+block_size];target=Z[:,ids].toarray()
        T=lu.solve(np.asarray(A.T@target))
        product,error=product_and_error(A,T)
        residual=target-product
        bound=np.nextafter(abs(residual)+error+EPS*(abs(target)+abs(product))+ETA,np.inf)
        n=norm_upper(bound);remainder2=up(remainder2+up(n*n))
        peak=max(peak,target.nbytes+T.nbytes+product.nbytes+error.nbytes+bound.nbytes)
    remainder=up(np.sqrt(remainder2)) if remainder2 else 0.
    low,high=spectral_scale_bounds(Z);cutlow=down(max(Z.shape)*EPS*low);cuthigh=up(max(Z.shape)*EPS*high)
    separation=certificate['separation_factor'];retained_pass=minimum>separation*cuthigh
    discarded_pass=not len(dependent) or remainder<cutlow/separation
    return dict(certified=bool(retained_pass and discarded_pass),rank=len(retained),columns=Z.shape[1],rows=Z.shape[0],
        full_column_rank=len(retained)==Z.shape[1],
        method='retained_columns_Weyl_and_COMPLETE_current_dependency_reconstruction_v1',
        retained_reference_sigma_min_lower=certificate['sigma_r_lower'],retained_change_F_upper=change,
        sigma_r_lower=minimum,sigma_r_plus_one_upper=remainder,sigma_max_lower=low,sigma_max_upper=high,
        cutoff_lower=cutlow,cutoff_upper=cuthigh,separation_factor=separation,
        retained_separated=bool(retained_pass),discarded_separated=bool(discarded_pass),
        visited_dependent_columns=len(dependent),block_size=block_size,peak_explicit_block_bytes=peak,
        elapsed_s=time.perf_counter()-started,
        scope='Same semantic coordinates and retained pivot IDs; Weyl certifies A only, all other Z columns rechecked; frozen binary64 map')


class CertifiedRangeGramProjector:
    """Fixed linear reduced inverse, with physical guards on the full map."""
    def __init__(self,Z,retained,*,rtol,certificate_diagnostics=None):
        self.Z=sparse.csc_matrix(Z,copy=True);self.retained_columns=np.array(retained,dtype=np.int64,copy=True)
        self.retained_columns.setflags(write=False);self.rank=len(retained);self.rtol=rtol
        self.A=self.Z[:,self.retained_columns].tocsc();self.frobenius=float(np.linalg.norm(self.Z.data))
        started=time.perf_counter();self.G=(self.A.T@self.A).tocsc()
        try:self.lu=splu(self.G,permc_spec='COLAMD')
        except RuntimeError as error:raise ProjectorFailure('Current retained-column Gram factorization failed') from error
        self.diagnostics=dict(backend='current_sparse_retained_Gram_LU_with_complete_range_certificate_v1',
            rank=self.rank,coordinates=self.Z.shape[1],dependent_coordinates=self.Z.shape[1]-self.rank,
            retained_columns=self.retained_columns.tolist(),current_gram_factorization_seconds=time.perf_counter()-started,
            Gram_nnz=self.G.nnz,LU_nnz=self.lu.L.nnz+self.lu.U.nnz,rank_certificate=certificate_diagnostics,
            no_mass_scaling=True,no_regularization=True,reused_Gram=False,
            coordinate_change='Exact supplied Z; independent pivot gauge, dependent coefficients zero, ALL physical equations checked')

    def _input(self,value,rows):
        value=np.asarray(value,float)
        if value.ndim not in (1,2) or value.shape[0]!=rows or not np.isfinite(value).all():raise ValueError('Finite map-compatible RHS required')
        return (value[:,None] if value.ndim==1 else value),value.ndim==1

    def _guard(self,record):
        values=np.array(list(record.values()),float)
        if not np.isfinite(values).all() or np.max(values,initial=0.)>self.rtol:
            raise ProjectorFailure('Current full physical range-Gram checks failed: '+str(record))

    def project(self,y,*,return_coefficients=False):
        y,vector=self._input(y,self.Z.shape[0]);largest=float(np.max(abs(y),initial=0.))
        c=np.zeros((self.Z.shape[1],y.shape[1]))
        if largest==0.:p=np.zeros_like(y);normal=energy=0.
        else:
            c[self.retained_columns]=self.lu.solve(self.A.T@y);p=self.Z@c
            yn=y/largest;pn=p/largest;r=yn-pn;scale=float(np.linalg.norm(yn))
            normal=float(np.linalg.norm(self.Z.T@r))/(self.frobenius*scale)
            energy=abs(float(np.sum(yn*yn)-np.sum(pn*pn)-np.sum(r*r)))/scale**2
        record=dict(normal_relative=normal,energy_identity_relative=energy);self._guard(record)
        if not np.isfinite(c).all() or not np.isfinite(p).all():raise ProjectorFailure('Nonfinite range-Gram projection')
        if vector:c=c[:,0];p=p[:,0]
        return (p,c,record) if return_coefficients else (p,record)

    def solve_gram(self,b,*,return_response=False):
        b,vector=self._input(b,self.Z.shape[1]);largest=float(np.max(abs(b),initial=0.));c=np.zeros_like(b)
        if largest==0.:p=np.zeros((self.Z.shape[0],b.shape[1]));residual=energy=0.
        else:
            c[self.retained_columns]=self.lu.solve(b[self.retained_columns]);p=self.Z@c;bn=b/largest
            residual=float(np.linalg.norm(self.Z.T@(p/largest)-bn))/float(np.linalg.norm(bn))
            plargest=float(np.max(abs(p),initial=0.))
            if plargest==0.:energy=0.
            else:
                pn=p/plargest;energy=abs(float(np.sum((b/plargest)*(c/plargest))-np.sum(pn*pn)))/float(np.sum(pn*pn))
        record=dict(full_Gram_residual_relative=residual,dual_energy_identity_relative=energy);self._guard(record)
        if not np.isfinite(c).all() or not np.isfinite(p).all():raise ProjectorFailure('Nonfinite range-Gram response')
        if vector:c=c[:,0];p=p[:,0]
        return (c,p,record) if return_response else (c,record)

    def project_coefficients(self,y):return self.project(y,return_coefficients=True)[1]
    def dual_coefficients(self,b):return self.solve_gram(b)[0]


class CertifiedRangeGramCache:
    def __init__(self,*,rtol=1e-11,certificate_block_size=32):
        if not 0<rtol<=1e-11:raise ValueError('Physical residual tolerance may not exceed 1e-11')
        if type(certificate_block_size) is not int or certificate_block_size<1:raise ValueError('Positive integer block size required')
        self.rtol=rtol;self.block_size=certificate_block_size;self._reference=None;self._identity=None
        self.certifications=0;self.reuses=0;self.last_check=None

    def factor(self,Z,coordinate_identity):
        try:hash(coordinate_identity)
        except TypeError as error:raise ValueError('Complete immutable coordinate identity required') from error
        if coordinate_identity is None:raise ValueError('Coordinate identity is mandatory')
        Z=sparse.csc_matrix(Z,dtype=np.float64,copy=True);Z.sum_duplicates();Z.sort_indices()
        if not np.isfinite(Z.data).all():raise ValueError('Finite current energy map required')
        started=time.perf_counter();reference=self._reference;attempt=None;result=None
        reason='initial_reference' if reference is None else 'coordinate_identity_changed'
        if reference is not None and coordinate_identity==self._identity and Z.shape==reference['Z'].shape:
            reason='retained_Weyl_or_dependencies_failed'
            try:
                result=CertifiedRangeGramProjector(Z,reference['retained'],rtol=self.rtol)
                attempt=_current_range_proof(Z,reference['retained'],reference['Z'],reference['certificate'],result.lu,block_size=self.block_size)
            except ProjectorFailure as error:attempt=dict(certified=False,failure=str(error))
            if attempt['certified']:reason='certified_retained_Weyl_and_dependencies_reuse'
        if reason!='certified_retained_Weyl_and_dependencies_reuse':
            qr=SparseQRProjector(Z,audit='probes_only',rtol=self.rtol);certificate=certify(qr,block_size=self.block_size)
            if not certificate['certified'] or certificate['rank']<1:
                self.last_check=dict(reason=reason,certificate=certificate,reuse_attempt=attempt)
                raise ProjectorFailure('Range Gram requires a complete separated-rank certificate: '+str(self.last_check))
            retained=qr.retained_columns.copy();retained.setflags(write=False)
            reference=dict(Z=Z.copy(),retained=retained,certificate=copy.deepcopy(certificate))
            proof=dict(qr_factorization_seconds=qr.diagnostics['native_factorization_seconds'],certificate=certificate)
            result=CertifiedRangeGramProjector(Z,retained,rtol=self.rtol,certificate_diagnostics=proof)
        else:
            proof=dict(reference_certificate=copy.deepcopy(reference['certificate']),current_range=attempt)
            result.diagnostics['rank_certificate']=proof
        self._reference=reference;self._identity=coordinate_identity
        if reason=='certified_retained_Weyl_and_dependencies_reuse':self.reuses+=1
        else:self.certifications+=1
        self.last_check=dict(reason=reason,certifications=self.certifications,reuses=self.reuses,rank_certificate=proof,
            reuse_attempt=attempt,elapsed_s=time.perf_counter()-started)
        result.diagnostics['rank_cache']=copy.deepcopy(self.last_check)
        return result

    def snapshot(self):
        reference=None
        if self._reference is not None:
            if not _valid_identity(self._identity):raise ValueError('Snapshots require persistent lowercase SHA256 identity')
            Z=self._reference['Z']
            reference=dict(Z_csc=dict(shape=list(Z.shape),data=Z.data.tolist(),indices=Z.indices.tolist(),indptr=Z.indptr.tolist(),
                index_dtype=Z.indices.dtype.str,indptr_dtype=Z.indptr.dtype.str),
                retained_columns=self._reference['retained'].tolist(),certificate=copy.deepcopy(self._reference['certificate']))
        payload=dict(schema=SNAPSHOT_SCHEMA,rtol=float(self.rtol),certificate_block_size=self.block_size,
            identity_sha256=self._identity,reference=reference,certifications=self.certifications,reuses=self.reuses,
            last_check=copy.deepcopy(self.last_check))
        payload['payload_sha256']=_payload_digest(payload);return payload

    @classmethod
    def restore(cls,payload):
        keys={'schema','rtol','certificate_block_size','identity_sha256','reference','certifications','reuses','last_check','payload_sha256'}
        if not isinstance(payload,dict) or set(payload)!=keys or payload.get('schema')!=SNAPSHOT_SCHEMA:
            raise ProjectorFailure('Malformed certified range-Gram cache snapshot schema')
        payload=copy.deepcopy(payload)
        try:digest=_payload_digest(payload)
        except (TypeError,ValueError,OverflowError) as error:raise ProjectorFailure('Nonportable range cache snapshot data') from error
        if not _valid_identity(payload['payload_sha256']) or digest!=payload['payload_sha256']:
            raise ProjectorFailure('Certified range-Gram snapshot checksum mismatch')
        if (type(payload['rtol']) is not float or not 0<payload['rtol']<=1e-11 or
            type(payload['certificate_block_size']) is not int or payload['certificate_block_size']<1 or
            any(type(payload[key]) is not int or payload[key]<0 for key in ('certifications','reuses')) or
            payload['last_check'] is not None and not isinstance(payload['last_check'],dict)):
            raise ProjectorFailure('Malformed certified range-Gram cache options/counters')
        result=cls(rtol=payload['rtol'],certificate_block_size=payload['certificate_block_size']);reference=payload['reference']
        if reference is None:
            if payload['identity_sha256'] is not None or payload['certifications'] or payload['reuses']:
                raise ProjectorFailure('Empty range cache has identity or successful counters')
        else:
            if (not isinstance(reference,dict) or set(reference)!={'Z_csc','certificate','retained_columns'} or
                not _valid_identity(payload['identity_sha256']) or payload['certifications']<1 or
                not isinstance(reference['certificate'],dict)):
                raise ProjectorFailure('Malformed cached range reference/identity')
            retained=reference['retained_columns']
            if not isinstance(retained,list) or any(type(i) is not int for i in retained):raise ProjectorFailure('Invalid retained column IDs')
            Z=_restore_csc(reference['Z_csc']);qr=SparseQRProjector(Z,audit='probes_only',rtol=result.rtol)
            regenerated=certify(qr,block_size=result.block_size)
            if (not regenerated['certified'] or regenerated['rank']<1 or qr.retained_columns.tolist()!=retained or
                _proof_bits(regenerated)!=_proof_bits(reference['certificate'])):
                raise ProjectorFailure('Regenerated complete range certificate or pivots differ from snapshot')
            ids=np.asarray(retained,dtype=np.int64);ids.setflags(write=False)
            result._reference=dict(Z=Z,retained=ids,certificate=reference['certificate']);result._identity=payload['identity_sha256']
        result.certifications=payload['certifications'];result.reuses=payload['reuses'];result.last_check=payload['last_check']
        return result
