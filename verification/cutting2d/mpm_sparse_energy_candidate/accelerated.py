"""Certified FULL-rank reference plus current sparse physical Gram factor.

Only the rank certificate is reused. Each factor owns current Z and a newly
factored Z.T Z. The caller must supply an immutable identity covering the
persistent material object/topology/mass/GP ordering, lattice node addresses,
grid spacing/origin, support mask, bottom xs/ys, active reduced-column IDs and
the deterministic local Householder-coordinate convention. A shape or
coordinate count alone is never an acceptable identity.

No threshold is retried, no mass is changed, and a deficient/uncertified
reference is rejected. These objects do not mutate any material state.
"""
import copy
import hashlib
import json
import re
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from cutting2d.energy import ProjectorFailure
from .qr import SparseQRProjector
from .rank_certificate import certify,FullColumnReference

SNAPSHOT_SCHEMA='mpm-certified-gram-cache-snapshot-v1'


def _payload_digest(payload):
    body={key:value for key,value in payload.items() if key!='payload_sha256'}
    return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _proof_bits(value):
    """Compare all non-timing certificate fields, preserving scalar types/bits."""
    if isinstance(value,dict):return {key:_proof_bits(item) for key,item in value.items() if key!='elapsed_s'}
    if isinstance(value,list):return [_proof_bits(item) for item in value]
    if isinstance(value,float):return ['binary64',value.hex()]
    if isinstance(value,bool):return ['bool',value]
    if isinstance(value,int):return ['int',value]
    return value


def _valid_identity(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def _restore_csc(payload):
    if not isinstance(payload,dict) or set(payload)!=set(('shape','data','indices','indptr','index_dtype','indptr_dtype')):
        raise ProjectorFailure('Malformed cached CSC schema')
    shape=payload['shape']
    if not isinstance(shape,list) or len(shape)!=2 or any(type(v) is not int or v<1 for v in shape) or shape[0]<shape[1]:
        raise ProjectorFailure('Malformed cached CSC shape')
    indices=payload['indices'];indptr=payload['indptr'];data=payload['data']
    if (not all(isinstance(v,list) for v in (indices,indptr,data)) or
        any(type(v) is not int for v in indices+indptr) or any(type(v) not in (int,float) for v in data)):
        raise ProjectorFailure('Malformed cached CSC scalar types')
    if (len(indices)!=len(data) or len(indptr)!=shape[1]+1 or indptr[0]!=0 or indptr[-1]!=len(data) or
        any(v<0 or v>=shape[0] for v in indices) or any(v<0 or v>len(data) for v in indptr) or
        any(a>b for a,b in zip(indptr,indptr[1:]))):raise ProjectorFailure('Malformed cached CSC structure')
    if payload['index_dtype'] not in ('<i4','<i8') or payload['indptr_dtype'] not in ('<i4','<i8'):
        raise ProjectorFailure('Unsupported cached CSC integer encoding')
    for first,last in zip(indptr,indptr[1:]):
        if any(a>=b for a,b in zip(indices[first:last],indices[first+1:last])):
            raise ProjectorFailure('Cached CSC must have strictly sorted unique row indices')
    values=np.asarray(data,dtype=np.float64)
    if not np.isfinite(values).all():raise ProjectorFailure('Nonfinite cached CSC value')
    row=np.asarray(indices,dtype=payload['index_dtype']);ptr=np.asarray(indptr,dtype=payload['indptr_dtype'])
    # scipy may choose a smaller index dtype in its constructor; restore the
    # serialized canonical buffers explicitly for lossless snapshots.
    result=sparse.csc_matrix((values,row,ptr),shape=tuple(shape));result.indices=row;result.indptr=ptr
    return result


class CertifiedGramCache:
    def __init__(self,*,rtol=1e-11,certificate_block_size=32):
        if not 0<rtol<=1e-11:raise ValueError('Physical residual tolerance may not exceed 1e-11')
        self.rtol=rtol;self.block_size=certificate_block_size;self._reference=None;self._identity=None
        self.certifications=0;self.reuses=0;self.last_check=None

    def factor(self,Z,coordinate_identity):
        # The root solver constructs the complete semantic token. Requiring
        # hashability excludes mutable ndarray/list identities.
        try:hash(coordinate_identity)
        except TypeError as error:raise ValueError('Complete immutable coordinate identity required') from error
        if coordinate_identity is None:raise ValueError('Coordinate identity is mandatory')
        Z=sparse.csc_matrix(Z,dtype=np.float64,copy=True);Z.sum_duplicates();Z.sort_indices()
        started=time.perf_counter();reference=self._reference;reuse=None
        reason='initial_reference' if reference is None else 'coordinate_identity_changed'
        if reference is not None and coordinate_identity==self._identity:
            reuse=reference.check(Z,row_ids=range(Z.shape[0]),column_ids=range(Z.shape[1]))
            reason='Weyl_margin_expired'
            if reuse['certified']:reason='certified_Weyl_reuse'
        if reason!='certified_Weyl_reuse':
            qr=SparseQRProjector(Z,audit='probes_only',rtol=self.rtol)
            certificate=certify(qr,block_size=self.block_size)
            self.last_check=dict(reason=reason,certificate=certificate,reuse_attempt=reuse)
            if not certificate['certified'] or not certificate['full_column_rank']:
                raise ProjectorFailure('Accelerated Gram requires certified full column rank: '+str(self.last_check))
            reference=FullColumnReference(Z,certificate,row_ids=range(Z.shape[0]),column_ids=range(Z.shape[1]))
            certificate_source=dict(qr_factorization_seconds=qr.diagnostics['native_factorization_seconds'],certificate=certificate)
            del qr
        else:certificate_source=dict(reference_certificate=reference.certificate,Weyl=reuse)
        # Failed factorization never replaces a valid existing rank reference.
        result=CertifiedGramProjector(Z,rtol=self.rtol,certificate_diagnostics=certificate_source)
        self._reference=reference;self._identity=coordinate_identity
        if reason=='certified_Weyl_reuse':self.reuses+=1
        else:self.certifications+=1
        self.last_check=dict(reason=reason,certifications=self.certifications,reuses=self.reuses,
            rank_certificate=certificate_source,elapsed_s=time.perf_counter()-started)
        result.diagnostics['rank_cache']=self.last_check.copy()
        return result

    def snapshot(self):
        """Portable verification-cache data; never serializes current Gram/LU.

        A persistent SHA256 semantic identity is required for nonempty caches.
        The checksum detects edits/corruption; it is not an authentication key.
        """
        reference=None
        if self._reference is not None:
            if not _valid_identity(self._identity):
                raise ValueError('Snapshots require a persistent lowercase SHA256 coordinate identity')
            Z=self._reference.Z
            reference=dict(Z_csc=dict(shape=list(Z.shape),data=Z.data.tolist(),indices=Z.indices.tolist(),indptr=Z.indptr.tolist(),
                index_dtype=Z.indices.dtype.str,indptr_dtype=Z.indptr.dtype.str),certificate=copy.deepcopy(self._reference.certificate))
        payload=dict(schema=SNAPSHOT_SCHEMA,rtol=float(self.rtol),certificate_block_size=int(self.block_size),
            identity_sha256=self._identity,reference=reference,certifications=self.certifications,reuses=self.reuses,
            last_check=copy.deepcopy(self.last_check))
        payload['payload_sha256']=_payload_digest(payload)
        return payload

    @classmethod
    def restore(cls,payload):
        """Recompute the complete reference proof before restoring diagnostics.

        Every mathematical certificate field must match bit for bit; only its
        elapsed_s is excluded. Original timing records/counters are then kept,
        so verification work itself does not change the checkpointed cache.
        """
        keys={'schema','rtol','certificate_block_size','identity_sha256','reference','certifications','reuses','last_check','payload_sha256'}
        if not isinstance(payload,dict) or set(payload)!=keys or payload.get('schema')!=SNAPSHOT_SCHEMA:
            raise ProjectorFailure('Malformed certified Gram cache snapshot schema')
        payload=copy.deepcopy(payload)
        try:digest=_payload_digest(payload)
        except (TypeError,ValueError,OverflowError) as error:raise ProjectorFailure('Nonportable cache snapshot data') from error
        if not _valid_identity(payload['payload_sha256']) or digest!=payload['payload_sha256']:
            raise ProjectorFailure('Certified Gram cache snapshot checksum mismatch')
        if (type(payload['rtol']) is not float or not 0<payload['rtol']<=1e-11 or
            type(payload['certificate_block_size']) is not int or payload['certificate_block_size']<1 or
            any(type(payload[key]) is not int or payload[key]<0 for key in ('certifications','reuses')) or
            payload['last_check'] is not None and not isinstance(payload['last_check'],dict)):
            raise ProjectorFailure('Malformed certified Gram cache options/counters')
        result=cls(rtol=payload['rtol'],certificate_block_size=payload['certificate_block_size'])
        reference=payload['reference']
        if reference is None:
            if payload['identity_sha256'] is not None or payload['certifications'] or payload['reuses']:
                raise ProjectorFailure('Empty cache snapshot has an identity or successful counters')
        else:
            if (not isinstance(reference,dict) or set(reference)!={'Z_csc','certificate'} or
                not _valid_identity(payload['identity_sha256']) or payload['certifications']<1 or
                not isinstance(reference['certificate'],dict)):
                raise ProjectorFailure('Malformed cached reference/identity')
            Z=_restore_csc(reference['Z_csc'])
            qr=SparseQRProjector(Z,audit='probes_only',rtol=result.rtol)
            regenerated=certify(qr,block_size=result.block_size)
            if (not regenerated['certified'] or not regenerated['full_column_rank'] or
                _proof_bits(regenerated)!=_proof_bits(reference['certificate'])):
                raise ProjectorFailure('Regenerated complete rank certificate differs from snapshot mathematics')
            result._reference=FullColumnReference(Z,reference['certificate'],row_ids=range(Z.shape[0]),column_ids=range(Z.shape[1]))
            result._identity=payload['identity_sha256']
        result.certifications=payload['certifications'];result.reuses=payload['reuses'];result.last_check=payload['last_check']
        return result


class CertifiedGramProjector:
    def __init__(self,Z,*,rtol,certificate_diagnostics):
        self.Z=sparse.csc_matrix(Z,copy=True);self.rank=self.Z.shape[1];self.rtol=rtol
        if not np.isfinite(self.Z.data).all():raise ValueError('Finite current energy map required')
        self.retained_columns=np.arange(self.rank,dtype=np.int64);self.retained_columns.setflags(write=False)
        self.frobenius=float(np.linalg.norm(self.Z.data));started=time.perf_counter()
        self.G=(self.Z.T@self.Z).tocsc()
        try:self.lu=splu(self.G,permc_spec='COLAMD')
        except RuntimeError as error:raise ProjectorFailure('Certified current Gram factorization failed') from error
        self.diagnostics=dict(backend='current_sparse_Gram_LU_with_independent_full_rank_certificate_v1',
            rank=self.rank,coordinates=self.rank,current_gram_factorization_seconds=time.perf_counter()-started,
            Gram_nnz=self.G.nnz,LU_nnz=self.lu.L.nnz+self.lu.U.nnz,
            rank_certificate=certificate_diagnostics,no_mass_scaling=True,no_regularization=True,
            reused_Gram=False,coordinate_change='Exact caller supplied Z; no hidden rescaling')

    def _input(self,value,rows):
        value=np.asarray(value,float)
        if value.ndim not in (1,2) or value.shape[0]!=rows or not np.isfinite(value).all():raise ValueError('Finite map-compatible RHS required')
        return (value[:,None] if value.ndim==1 else value),value.ndim==1

    def _guard(self,record):
        values=np.array(list(record.values()),float)
        if not np.isfinite(values).all() or np.max(values,initial=0.)>self.rtol:
            raise ProjectorFailure('Current physical Gram checks failed: '+str(record))

    def project(self,y,*,return_coefficients=False):
        y,vector=self._input(y,self.Z.shape[0]);largest=float(np.max(abs(y),initial=0.))
        if largest==0.:
            c=np.zeros((self.rank,y.shape[1]));p=np.zeros_like(y);normal=energy=0.
        else:
            c=self.lu.solve(self.Z.T@y);p=self.Z@c
            yn=y/largest;pn=p/largest;r=yn-pn;scale=float(np.linalg.norm(yn))
            normal=float(np.linalg.norm(self.Z.T@r))/(self.frobenius*scale)
            energy=abs(float(np.sum(yn*yn)-np.sum(pn*pn)-np.sum(r*r)))/scale**2
        record=dict(normal_relative=normal,energy_identity_relative=energy)
        self._guard(record)
        if not np.isfinite(c).all() or not np.isfinite(p).all():raise ProjectorFailure('Nonfinite Gram projection')
        if vector:c=c[:,0];p=p[:,0]
        return (p,c,record) if return_coefficients else (p,record)

    def solve_gram(self,b,*,return_response=False):
        b,vector=self._input(b,self.rank);largest=float(np.max(abs(b),initial=0.))
        if largest==0.:
            c=np.zeros_like(b);p=np.zeros((self.Z.shape[0],b.shape[1]));residual=energy=0.
        else:
            c=self.lu.solve(b);p=self.Z@c;bn=b/largest
            residual=float(np.linalg.norm(self.Z.T@(p/largest)-bn))/float(np.linalg.norm(bn))
            plargest=float(np.max(abs(p),initial=0.))
            if plargest==0.:energy=0.
            else:
                pn=p/plargest;energy=abs(float(np.sum((b/plargest)*(c/plargest))-np.sum(pn*pn)))/float(np.sum(pn*pn))
        record=dict(full_Gram_residual_relative=residual,dual_energy_identity_relative=energy)
        self._guard(record)
        if not np.isfinite(c).all() or not np.isfinite(p).all():raise ProjectorFailure('Nonfinite Gram response')
        if vector:c=c[:,0];p=p[:,0]
        return (c,p,record) if return_response else (c,record)

    def project_coefficients(self,y):return self.project(y,return_coefficients=True)[1]
    def dual_coefficients(self,b):return self.solve_gram(b)[0]
