"""Native finite-JC batches with versioned stable Hencky arithmetic."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
import copy
import hashlib
import importlib.util
import json
import os
import platform
import shlex
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import numpy as np
from cutting2d.material import ENERGY_FIELDS, MaterialFailure, kernel_material

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SCHEMA='direct-q4-native-jc-batch-v1'


@lru_cache(maxsize=1)
def load_native():
    import pybind11
    source=HERE/'native/finite_jc_batch.cpp'
    vendor=ROOT/'constitutive/finite_jc/vendor/eigen-3.4.0.tar.gz'
    vendor_manifest=json.loads((vendor.parent/'manifest.json').read_text())
    vendor_hash=hashlib.sha256(vendor.read_bytes()).hexdigest()
    if vendor_hash!=vendor_manifest['sha256']:raise RuntimeError('Eigen vendor hash mismatch')
    compiler=shlex.split(os.environ.get('CXX','c++'))
    flags=['-std=c++17','-O2','-shared','-fPIC','-fvisibility=hidden','-DEIGEN_MPL2_ONLY']
    if sys.platform=='darwin':flags+=['-undefined','dynamic_lookup']
    identity={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'historical_core_sha256':hashlib.sha256((ROOT/'constitutive/finite_jc/finite_jc_core.cpp').read_bytes()).hexdigest(),
              'eigen_sha256':vendor_hash,'pybind11':pybind11.__version__,'python':sys.version,
              'platform':platform.platform(),'compiler':subprocess.check_output(compiler+['--version'],text=True).splitlines()[0],
              'flags':flags}
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    build=ROOT/'.build/direct_q4'/digest;build.mkdir(parents=True,exist_ok=True)
    binary=build/('direct_q4_finite_jc'+sysconfig.get_config_var('EXT_SUFFIX'))
    if not binary.exists():
        with tempfile.TemporaryDirectory(dir=build) as directory:
            temporary=Path(directory)
            with tarfile.open(vendor) as archive:archive.extractall(temporary,filter='data')
            candidate=temporary/binary.name
            command=compiler+flags+['-I'+str(temporary/'eigen-3.4.0'),'-I'+pybind11.get_include(),
                         '-I'+sysconfig.get_paths()['include'],str(source),'-o',str(candidate)]
            result=subprocess.run(command,text=True,capture_output=True,timeout=120)
            (build/'build.log').write_text(result.stdout+result.stderr)
            if result.returncode:raise RuntimeError('Native finite JC build failed: '+str(build/'build.log'))
            os.replace(candidate,binary)
            identity.update(command=command,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
            (build/'manifest.json').write_text(json.dumps(identity,indent=2)+'\n')
    spec=importlib.util.spec_from_file_location('direct_q4_finite_jc',binary)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.build_manifest=str(build/'manifest.json')
    return module


@dataclass(frozen=True)
class Response:
    """Read-only views own their native buffer even after later trials/commits."""
    fields: dict
    _batch: object
    _generation: int

    def __getattr__(self,name):
        if name=='temperature':name='T'
        try:return self.fields[name]
        except KeyError:raise AttributeError(name) from None

    @property
    def energy_densities(self):return MappingProxyType({k:self.fields[k] for k in ENERGY_FIELDS})

    def energies_J(self,volume0):
        volume=np.asarray(volume0,float)
        if volume.shape!=self.T.shape or np.any(volume<=0) or not np.isfinite(volume).all():
            raise ValueError('Positive finite reference volume per point required')
        return {k:self.fields[k]*volume for k in ENERGY_FIELDS}

    @property
    def first_piola(self):
        return np.linalg.solve(self.F,np.swapaxes(self.tau,1,2)).swapaxes(1,2)


class ConstitutiveBatch:
    """One material, immutable committed native histories, atomic trial batch.

    F is total 3D deformation, dt is seconds, external heat is J, and volume0
    is the fixed positive reference volume. No buffer may be concurrently
    modified by the caller while trial is running. One batch cannot run two
    simultaneous calls; separate batches can run with the GIL released.
    """
    def __init__(self,material,N,T=None,*,options=None,threads=1):
        if not isinstance(N,(int,np.integer)) or N<=0:raise ValueError('Positive integer point count required')
        properties=kernel_material(material)
        temperature=np.broadcast_to(np.asarray(properties['REFERENCE_TEMPERATURE'] if T is None else T,float),(N,))
        if np.any(temperature<=0) or np.any(temperature>=properties['MELD_TEMPERATURE']) or not np.isfinite(temperature).all():
            raise ValueError('Initial temperature must satisfy 0 < T < Tm')
        kernel=load_native()
        cache={float(t):kernel.initialize_state(properties,temperature=float(t)) for t in np.unique(temperature)}
        self._setup(properties,[cache[float(t)] for t in temperature],options,threads)

    def _setup(self,properties,states,options,threads):
        if not isinstance(threads,int) or not 1<=threads<=32:raise ValueError('threads must be an integer in [1,32]')
        self.material=MappingProxyType(copy.deepcopy(properties));self.options=MappingProxyType(copy.deepcopy(options or {}))
        self.threads=threads;self.N=len(states);self._generation=0;self._pending=None
        self.native=load_native().NativeBatch(dict(self.material),states,dict(self.options))
        self.state=Response(MappingProxyType(self.native.fields()),self,0)

    def _trial(self,F,dt,heat_J,volume0,fd_step):
        self.reject()
        deformation=np.ascontiguousarray(F,dtype=np.float64)
        volume=np.broadcast_to(np.asarray(volume0,float),(self.N,))
        heat=np.broadcast_to(np.asarray(heat_J,float),(self.N,))
        if deformation.shape!=(self.N,3,3) or not np.isfinite(deformation).all():raise ValueError('Finite F with shape (N,3,3) required')
        if np.any(volume<=0) or not np.isfinite(volume).all() or not np.isfinite(heat).all():raise ValueError('Positive reference volumes and finite heat required')
        if not np.isfinite(dt) or dt<=0:raise ValueError('Positive finite dt required')
        density=np.ascontiguousarray(heat/volume)
        base_T=self.state.T+density/(self.material['DENSITY']*self.material['SPECIFIC_HEAT'])
        invalid=(base_T<=0)|(base_T>=self.material['MELD_TEMPERATURE'])
        if invalid.any():
            raise MaterialFailure(int(np.flatnonzero(invalid)[0]),'solid_temperature_limit','External heat leaves 0 < T < Tm')
        if fd_step is None:result=self.native.trial(deformation,float(dt),density,self.threads)
        else:result=self.native.trial_with_tangent(deformation,float(dt),density,self.threads,float(fd_step))
        if not result['success']:raise MaterialFailure(result['index'],result['code'],result['message'])
        self._generation+=1
        response=Response(MappingProxyType(self.native.fields(True)),self,self._generation)
        self._pending=response
        return response

    def trial(self,F,dt,heat_J,volume0):
        return self._trial(F,dt,heat_J,volume0,None)

    def trial_with_tangent(self,F,dt,heat_J,volume0,*,fd_step=2e-6):
        """Nominal response plus central dP_ij/dF_kl for in-plane entries.

        Each perturbation starts from the same committed history and uses
        delta F_kl=fd_step*max(1,abs(F_kl)); Fzz=1 and transverse shears=0.
        .tangent has shape (N,2,2,2,2). .tangent_branch_changes (N,4,2)
        marks a perturbed return whose elastic/plastic branch differs from
        the nominal one. This is a numerical Jacobian, not an analytic
        consistent tangent or a certificate at nonsmooth branch crossings.
        """
        return self._trial(F,dt,heat_J,volume0,fd_step)

    def commit(self,response=None):
        chosen=self._pending if response is None else response
        if chosen is None or chosen is not self._pending:raise ValueError('Commit requires this batch\'s latest successful trial')
        self.native.commit();self.state=chosen;self._pending=None
        return self.state

    def reject(self):
        self.native.reject();self._pending=None
        return self.state

    def diagnostics(self,*,pending=False):return self.native.diagnostics(pending)

    def snapshot(self):
        return {'schema':SCHEMA,'kernel_version':load_native().__version__,
                'material':dict(self.material),'options':dict(self.options),'states':self.native.states(),
                'threads':self.threads}

    @classmethod
    def restore(cls,snapshot,*,threads=None):
        if snapshot.get('schema')!=SCHEMA or snapshot.get('kernel_version')!=load_native().__version__:
            raise ValueError('Incompatible native finite JC snapshot')
        states=snapshot['states']
        if not isinstance(states,list) or not states:raise ValueError('Nonempty serialized states required')
        result=cls.__new__(cls)
        result._setup(kernel_material(snapshot['material']),states,snapshot.get('options'),snapshot.get('threads',1) if threads is None else threads)
        return result
