"""Separate fast certified range cache with explicit complete-proof fallbacks.

Weyl/dependency reuse and the physical projector remain those of reduced.py.
Only obtaining a new reference changes: complete Gram proof of a proposed
basis, then fixed QR proposal and complete unsquared proof if necessary.
Neither LU/QR success nor sampled residuals ever certify the selected range.
"""
import copy
import time
import numpy as np
from scipy import sparse
from cutting2d.energy import ProjectorFailure
from .accelerated import _valid_identity,_payload_digest,_restore_csc
from .reduced import _current_range_proof,CertifiedRangeGramProjector
from .gram_certificate import certify_gram
from .rank_certificate import certify
from .qr import SparseQRProjector

SNAPSHOT_SCHEMA='mpm-certified-fast-range-gram-cache-snapshot-v1'
GRAM_METHOD='complete_Gram_inverse_residual_WITH_Gram_formation_error_v1'
UNSQUARED_METHOD='complete_left_inverse_and_dependent_reconstruction_with_gamma_bounds_v1'


def _proof_bits(value):
    """Keep mathematical types/bits, excluding only the two named timer fields."""
    if isinstance(value,dict):return {key:_proof_bits(item) for key,item in value.items() if key not in ('elapsed_s','timings')}
    if isinstance(value,list):return [_proof_bits(item) for item in value]
    if isinstance(value,float):return ['binary64',value.hex()]
    if isinstance(value,bool):return ['bool',value]
    if isinstance(value,int):return ['int',value]
    return value


class CertifiedFastRangeGramCache:
    def __init__(self,*,rtol=1e-11,certificate_block_size=32):
        if not 0<rtol<=1e-11:raise ValueError('Physical residual tolerance may not exceed 1e-11')
        if type(certificate_block_size) is not int or certificate_block_size<1:raise ValueError('Positive integer block size required')
        self.rtol=rtol;self.block_size=certificate_block_size;self._reference=None;self._identity=None
        self.certifications=0;self.reuses=0;self.last_check=None

    def _new_reference(self,Z,old_retained):
        attempts=[];proposals=[]
        if old_retained is not None:proposals.append(('current_retained_columns',old_retained))
        all_columns=np.arange(Z.shape[1],dtype=np.int64)
        if old_retained is None or not np.array_equal(old_retained,all_columns):proposals.append(('all_current_columns',all_columns))
        for label,retained in proposals:
            try:certificate=certify_gram(Z,retained,block_size=self.block_size)
            except ProjectorFailure as error:
                attempts.append(dict(proposal=label,certified=False,failure=str(error)));continue
            attempts.append(dict(proposal=label,certificate=certificate))
            if certificate['certified']:return retained,certificate,attempts
        # QR proposes a fixed basis; its rank and necessary probes are not a
        # substitute for either complete proof below. No pivot retry occurs.
        qr=SparseQRProjector(Z,audit='probes_only',rtol=self.rtol);retained=qr.retained_columns
        try:certificate=certify_gram(Z,retained,block_size=self.block_size)
        except ProjectorFailure as error:
            certificate=None;attempts.append(dict(proposal='fixed_QR_columns_Gram',certified=False,failure=str(error)))
        if certificate is not None:
            attempts.append(dict(proposal='fixed_QR_columns_Gram',certificate=certificate))
            if certificate['certified']:return retained,certificate,attempts
        certificate=certify(qr,block_size=self.block_size)
        attempts.append(dict(proposal='fixed_QR_columns_unsquared_complete',certificate=certificate))
        if not certificate['certified'] or certificate['rank']<1:
            raise ProjectorFailure('Fast range cache cannot certify complete separated rank: '+str(attempts))
        return retained,certificate,attempts

    def factor(self,Z,coordinate_identity):
        try:hash(coordinate_identity)
        except TypeError as error:raise ValueError('Complete immutable coordinate identity required') from error
        if coordinate_identity is None:raise ValueError('Coordinate identity is mandatory')
        Z=sparse.csc_matrix(Z,dtype=np.float64,copy=True);Z.sum_duplicates();Z.sort_indices()
        if not np.isfinite(Z.data).all():raise ValueError('Finite current energy map required')
        started=time.perf_counter();reference=self._reference;attempt=None;result=None;old_retained=None
        reason='initial_reference' if reference is None else 'coordinate_identity_changed'
        if reference is not None and coordinate_identity==self._identity and Z.shape==reference['Z'].shape:
            old_retained=reference['retained'];reason='retained_Weyl_or_dependencies_failed'
            try:
                result=CertifiedRangeGramProjector(Z,old_retained,rtol=self.rtol)
                attempt=_current_range_proof(Z,old_retained,reference['Z'],reference['certificate'],result.lu,block_size=self.block_size)
            except ProjectorFailure as error:attempt=dict(certified=False,failure=str(error))
            if attempt['certified']:reason='certified_retained_Weyl_and_dependencies_reuse'
        if reason!='certified_retained_Weyl_and_dependencies_reuse':
            try:retained,certificate,attempts=self._new_reference(Z,old_retained)
            except ProjectorFailure as error:
                self.last_check=dict(reason=reason,reuse_attempt=attempt,failure=str(error));raise
            retained=np.array(retained,dtype=np.int64,copy=True);retained.setflags(write=False)
            reference=dict(Z=Z.copy(),retained=retained,certificate=copy.deepcopy(certificate))
            proof=dict(certificate=certificate,reference_method=certificate['method'],complete_proof_attempts=attempts)
            # The proof may succeed while the squared physical solve fails.
            # Preserve that failure; do not weaken the equation guards.
            result=CertifiedRangeGramProjector(Z,retained,rtol=self.rtol,certificate_diagnostics=proof)
        else:
            proof=dict(reference_certificate=copy.deepcopy(reference['certificate']),current_range=attempt)
            result.diagnostics['rank_certificate']=proof
        self._reference=reference;self._identity=coordinate_identity
        if reason=='certified_retained_Weyl_and_dependencies_reuse':self.reuses+=1
        else:self.certifications+=1
        self.last_check=dict(reason=reason,certifications=self.certifications,reuses=self.reuses,
            rank_certificate=proof,reuse_attempt=attempt,elapsed_s=time.perf_counter()-started)
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
            raise ProjectorFailure('Malformed fast range-Gram cache snapshot schema')
        payload=copy.deepcopy(payload)
        try:digest=_payload_digest(payload)
        except (TypeError,ValueError,OverflowError) as error:raise ProjectorFailure('Nonportable fast range cache snapshot data') from error
        if not _valid_identity(payload['payload_sha256']) or digest!=payload['payload_sha256']:
            raise ProjectorFailure('Fast range-Gram snapshot checksum mismatch')
        if (type(payload['rtol']) is not float or not 0<payload['rtol']<=1e-11 or
            type(payload['certificate_block_size']) is not int or payload['certificate_block_size']<1 or
            any(type(payload[key]) is not int or payload[key]<0 for key in ('certifications','reuses')) or
            payload['last_check'] is not None and not isinstance(payload['last_check'],dict)):
            raise ProjectorFailure('Malformed fast range-Gram cache options/counters')
        result=cls(rtol=payload['rtol'],certificate_block_size=payload['certificate_block_size']);reference=payload['reference']
        if reference is None:
            if payload['identity_sha256'] is not None or payload['certifications'] or payload['reuses']:
                raise ProjectorFailure('Empty fast range cache has identity or successful counters')
        else:
            if (not isinstance(reference,dict) or set(reference)!={'Z_csc','certificate','retained_columns'} or
                not _valid_identity(payload['identity_sha256']) or payload['certifications']<1 or
                not isinstance(reference['certificate'],dict)):
                raise ProjectorFailure('Malformed cached fast range reference/identity')
            retained=reference['retained_columns'];certificate=reference['certificate']
            if not isinstance(retained,list) or any(type(i) is not int for i in retained):raise ProjectorFailure('Invalid retained column IDs')
            Z=_restore_csc(reference['Z_csc']);method=certificate.get('method')
            if method==GRAM_METHOD:
                try:regenerated=certify_gram(Z,np.asarray(retained,dtype=np.int64),block_size=result.block_size)
                except (ValueError,ProjectorFailure) as error:raise ProjectorFailure('Cannot regenerate complete Gram certificate') from error
            elif method==UNSQUARED_METHOD:
                qr=SparseQRProjector(Z,audit='probes_only',rtol=result.rtol)
                if qr.retained_columns.tolist()!=retained:raise ProjectorFailure('Restored unsquared QR pivot IDs differ')
                regenerated=certify(qr,block_size=result.block_size)
            else:raise ProjectorFailure('Unknown complete reference certificate method')
            if not regenerated['certified'] or regenerated['rank']<1 or _proof_bits(regenerated)!=_proof_bits(certificate):
                raise ProjectorFailure('Regenerated complete fast range certificate differs from snapshot mathematics')
            ids=np.asarray(retained,dtype=np.int64);ids.setflags(write=False)
            result._reference=dict(Z=Z,retained=ids,certificate=certificate);result._identity=payload['identity_sha256']
        result.certifications=payload['certifications'];result.reuses=payload['reuses'];result.last_check=payload['last_check']
        return result
