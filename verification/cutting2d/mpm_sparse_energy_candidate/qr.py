"""Fixed sparse Householder range, explicit pivot policy and separate audits."""
from functools import lru_cache
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys
import sysconfig
import tarfile
import time
import numpy as np
from scipy import sparse
from cutting2d.energy import ProjectorFailure

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
POLICY='eigen_default_explicit_20_m_plus_n_eps_max_column_norm_v1'


@lru_cache(maxsize=1)
def load_native():
    import pybind11
    source=HERE/'native_sparseqr.cpp';vendor=ROOT/'constitutive/finite_jc/vendor/eigen-3.4.0.tar.gz'
    vendor_hash=hashlib.sha256(vendor.read_bytes()).hexdigest()
    assert vendor_hash==json.loads((vendor.parent/'manifest.json').read_text())['sha256']
    identity=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),eigen_archive_sha256=vendor_hash,
        compiler=subprocess.check_output(['c++','--version'],text=True),python=sys.version,pybind11=pybind11.__version__,policy=POLICY)
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    build=ROOT/'.build/mpm_sparseqr'/digest;build.mkdir(parents=True,exist_ok=True)
    binary=build/('mpm_sparseqr_native'+sysconfig.get_config_var('EXT_SUFFIX'))
    if not binary.exists():
        with tarfile.open(vendor) as package:package.extractall(build,filter='data')
        command=['c++','-std=c++17','-O2','-shared','-fPIC','-fvisibility=hidden','-DEIGEN_MPL2_ONLY','-undefined','dynamic_lookup',
            '-I'+str(build/'eigen-3.4.0'),'-I'+pybind11.get_include(),'-I'+sysconfig.get_paths()['include'],str(source),'-o',str(binary)]
        result=subprocess.run(command,capture_output=True,text=True,timeout=120);(build/'build.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('SparseQR compilation failed; see '+str(build/'build.log'))
        identity.update(command=command,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
        (build/'manifest.json').write_text(json.dumps(identity,indent=2)+'\n')
    spec=importlib.util.spec_from_file_location('mpm_sparseqr_native',binary);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.build_manifest=str(build/'manifest.json');return module


class SparseQRProjector:
    """Z is the exact supplied map, including any explicit caller scaling.

    No Gram is formed. Small mode verifies the FIXED QR range against a dense
    SVD of this same Z. Large mode records probes only, never calls them a rank
    certificate. A failed audit does not modify the pivot threshold or retry
    with a different rank. Grid coefficients have a fixed numerical-nullspace
    gauge determined by COLAMD and the QR pivots, not a mass/DOF regularizer.
    """
    def __init__(self,Z,*,audit='small_svd',rtol=1e-10,max_dense_entries=2_000_000,ordering='COLAMD'):
        start=time.perf_counter();self.Z=sparse.csc_matrix(Z,copy=True);self.Z.sum_duplicates();self.Z.sort_indices()
        if not np.isfinite(self.Z.data).all() or not 0<rtol<1:raise ValueError('Finite supplied map and physical audit tolerance required')
        self.rtol=rtol;self.frobenius=float(np.linalg.norm(self.Z.data))
        column_norms=np.sqrt(np.asarray(self.Z.multiply(self.Z).sum(0)).ravel())
        # Eigen3.4 SparseQR.h:413-418. Fixed before any result; no threshold
        # adaptation after a failed solve, rank comparison or contact guard.
        maximum=float(np.max(column_norms,initial=0.)) or 1.
        threshold=20*sum(self.Z.shape)*maximum*np.finfo(float).eps
        module=load_native();factor_start=time.perf_counter()
        if ordering not in ('COLAMD','Natural'):raise ValueError('Explicit fixed COLAMD or Natural ordering required')
        constructor=module.SparseQRRange if ordering=='COLAMD' else module.SparseQRNaturalRange
        self.native=constructor(self.Z,threshold);factor_seconds=time.perf_counter()-factor_start;self.rank=self.native.rank
        self.retained_columns=np.array(self.native.permutation()[:self.rank],dtype=np.int64);self.retained_columns.setflags(write=False)
        self.diagnostics=dict(backend='Eigen3.4_SparseQR_'+ordering,shape=list(self.Z.shape),rank=self.rank,
            pivot_threshold=threshold,pivot_policy=POLICY,R_nonzeros=self.native.r_nonzeros,
            pivot_diagonal=self.native.diagonal().tolist(),permutation=self.native.permutation().tolist(),
            no_gram=True,no_regularization=True,no_threshold_retry=True,numerical_rank_equivalence_to_svd=False,
            build_manifest=load_native().build_manifest)
        self.diagnostics['native_factorization_seconds']=factor_seconds
        if audit=='small_svd':
            if np.prod(self.Z.shape)>max_dense_entries:raise ProjectorFailure('Requested small SVD audit exceeds explicit evidence budget')
            dense=self.Z.toarray();U,s,V=np.linalg.svd(dense,full_matrices=False)
            cutoff=max(self.Z.shape)*np.finfo(float).eps*s[0];rank=int(np.count_nonzero(s>cutoff));U=U[:,:rank]
            Q=self.native.basis_action(np.eye(self.rank))
            orth=float(np.linalg.norm(Q.T@Q-np.eye(self.rank)))
            off=float(np.linalg.norm(Q-U@(U.T@Q)))
            reconstruction=float(np.linalg.norm(dense-Q@(Q.T@dense))/max(np.linalg.norm(dense),np.finfo(float).tiny))
            ambiguous=bool(np.any((s>=cutoff/32)&(s<=32*cutoff)))
            self.diagnostics['audit']=dict(mode=audit,svd_rank=rank,svd_cutoff=cutoff,svd_values=s.tolist(),
                ambiguous_svd_rank=ambiguous,svd_separation_factor=32,orthogonality_frobenius=orth,off_svd_range_frobenius=off,reconstruction_relative=reconstruction,
                scope='Observed floating-point equivalence at this frozen Z; not interval proof or future-geometry certificate')
            if rank!=self.rank or ambiguous or max(orth,off,reconstruction)>rtol:
                raise ProjectorFailure('Fixed sparse QR rank/subspace differs from the small SVD audit: '+str(self.diagnostics['audit']))
            self.diagnostics['numerical_rank_equivalence_to_svd']=True
        elif audit=='probes_only':
            rng=np.random.default_rng(29031);c=rng.normal(size=(self.Z.shape[1],2));y=self.Z@c
            p,_=self.native.project(y);repeat,_=self.native.project(p)
            self.diagnostics['audit']=dict(mode=audit,range_probe_relative=float(np.linalg.norm(y-p)/np.linalg.norm(y)),
                idempotence_probe_relative=float(np.linalg.norm(repeat-p)/np.linalg.norm(p)),
                scope='Necessary probes only; no global numerical-rank certificate')
            if max(self.diagnostics['audit']['range_probe_relative'],self.diagnostics['audit']['idempotence_probe_relative'])>rtol:
                raise ProjectorFailure('SparseQR large-map necessary probes failed')
        else:raise ValueError('Explicit small_svd or probes_only audit required')
        self.diagnostics['construction_seconds']=time.perf_counter()-start

    def _input(self,value,rows):
        value=np.asarray(value,float)
        if value.ndim not in (1,2) or value.shape[0]!=rows or not np.isfinite(value).all():raise ValueError('Finite map-compatible vector/matrix required')
        return value[:,None] if value.ndim==1 else value,value.ndim==1

    def project(self,y,*,return_coefficients=False):
        y,vector=self._input(y,self.Z.shape[0]);largest=float(np.max(abs(y),initial=0.))
        if largest==0.:
            p=np.zeros_like(y);c=np.zeros((self.Z.shape[1],y.shape[1]));reconstruction=energy=orthogonality=0.
        else:
            p,c=self.native.project(y)
            # Normalize before products/norms: squaring a tiny nonzero RHS
            # must not turn its physical residual or energy identity into NaN.
            yn=y/largest;pn=p/largest;cn=c/largest;rn=yn-pn;scale=float(np.linalg.norm(yn))
            reconstruction=float(np.linalg.norm(self.Z@cn-pn))/scale
            energy=float(abs(np.sum(yn*yn)-np.sum(pn*pn)-np.sum(rn*rn)))/scale**2
            orthogonality=float(np.linalg.norm(self.Z.T@rn))/(self.frobenius*scale)
        record=dict(map_reconstruction_relative=reconstruction,energy_identity_relative=energy,normal_relative=orthogonality,
            rank=self.rank,rank_audit=self.diagnostics['audit']['mode'])
        if not np.isfinite(p).all() or not np.isfinite(c).all() or not np.isfinite([reconstruction,energy,orthogonality]).all() or max(reconstruction,energy,orthogonality)>self.rtol:
            raise ProjectorFailure('Sparse QR projection checks failed: '+str(record))
        if vector:p=p[:,0];c=c[:,0]
        return (p,c,record) if return_coefficients else (p,record)

    def solve_gram(self,b,*,return_response=False):
        b,vector=self._input(b,self.Z.shape[1]);largest=float(np.max(abs(b),initial=0.))
        if largest==0.:
            c=np.zeros_like(b);p=np.zeros((self.Z.shape[0],b.shape[1]));compatibility=reconstruction=energy=0.
        else:
            c,p=self.native.solve_gram(b);bn=b/largest
            compatibility=float(np.linalg.norm(self.Z.T@(p/largest)-bn))/float(np.linalg.norm(bn))
            plargest=float(np.max(abs(p),initial=0.))
            if plargest==0.:
                reconstruction=0.;energy=0.
            else:
                pn=p/plargest;cn=c/plargest;scale=float(np.linalg.norm(pn))
                reconstruction=float(np.linalg.norm(self.Z@cn-pn))/scale
                energy=float(abs(np.sum((b/plargest)*cn)-np.sum(pn*pn)))/scale**2
        record=dict(full_gram_residual_relative=compatibility,map_reconstruction_relative=reconstruction,
            dual_energy_identity_relative=energy,rank=self.rank,rank_audit=self.diagnostics['audit']['mode'],
            coefficient_gauge='dependent pivot coordinates zero; physical nullspace equivalence checked through full map')
        if not np.isfinite(c).all() or not np.isfinite(p).all() or not np.isfinite([compatibility,reconstruction,energy]).all() or max(compatibility,reconstruction,energy)>self.rtol:
            raise ProjectorFailure('Sparse QR compatible Gram checks failed: '+str(record))
        if vector:c=c[:,0];p=p[:,0]
        return (c,p,record) if return_response else (c,record)

    def project_coefficients(self,y):
        """Fixed-gauge coefficients c with Z c = projection(y)."""
        return self.project(y,return_coefficients=True)[1]

    def dual_coefficients(self,b):
        """Fixed-gauge c solving Z.T Z c = compatible b."""
        return self.solve_gram(b)[0]
