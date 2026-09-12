"""Transactional SI adapter to the existing, unmodified finite Johnson–Cook core.

Energy histories are densities per *initial* volume. External heat enters in J;
``trial`` divides it by the fixed reference particle volumes exactly once.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import sysconfig
import tarfile
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_FIELDS = {
    'young_modulus_pa': 'YOUNG_MODULUS', 'poisson_ratio': 'POISSON_RATIO',
    'density_kg_m3': 'DENSITY', 'specific_heat_j_kg_k': 'SPECIFIC_HEAT',
    'jc_a_pa': 'JC_PARAMETER_A', 'jc_b_pa': 'JC_PARAMETER_B',
    'jc_c': 'JC_PARAMETER_C', 'jc_n': 'JC_PARAMETER_n', 'jc_m': 'JC_PARAMETER_m',
    'reference_strain_rate_s_1': 'REFERENCE_STRAIN_RATE',
    'reference_temperature_k': 'REFERENCE_TEMPERATURE',
    'melt_temperature_k': 'MELD_TEMPERATURE',
    'taylor_quinney': 'TAYLOR_QUINNEY_COEFFICIENT',
}
ENERGY_FIELDS = ('elastic_energy', 'initial_elastic_energy', 'plastic_work',
                 'plastic_heat', 'external_heat', 'stored_energy',
                 'thermal_energy', 'mechanical_work')


class MaterialFailure(RuntimeError):
    """A failed particle trial. The entire batch remains uncommitted."""
    def __init__(self, index, code, message, diagnostics=None):
        self.index, self.code = index, code
        self.diagnostics = diagnostics or {}
        super().__init__(f'Particle {index}: {code}: {message}')


def kernel_material(material):
    """Accept a flat SI property dictionary, sourced card, or core dictionary."""
    if 'properties' in material:
        material = material['properties']
    elif 'material' in material:
        material = material['material']
    if 'YOUNG_MODULUS' in material:
        return {key: float(material[key]) for key in _FIELDS.values()}
    return {target: float(material[source]) for source, target in _FIELDS.items()}


@lru_cache(maxsize=1)
def load_kernel():
    """Build the unchanged source for this Python ABI and load its hashed binary.

    Requires a C++17 compiler, Python development headers, and pybind11 2.13.6.
    Eigen is taken from the archived, hash-checked repository dependency.
    No network access, compiler fast-math, or Kratos library is involved.
    """
    import pybind11
    source = ROOT / 'constitutive/finite_jc/finite_jc_core.cpp'
    vendor = source.parent / 'vendor/eigen-3.4.0.tar.gz'
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    flags = ['-std=c++17', '-O2', '-shared', '-fPIC', '-fvisibility=hidden', '-DEIGEN_MPL2_ONLY']
    if sys.platform == 'darwin':
        flags += ['-undefined', 'dynamic_lookup']
    identity = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    eigen_sha256=hashlib.sha256(vendor.read_bytes()).hexdigest(),
                    pybind11=pybind11.__version__, python=sys.version,
                    platform=platform.platform(), compiler=subprocess.check_output(
                        compiler + ['--version'], text=True).splitlines()[0], flags=flags)
    vendor_manifest = json.loads((vendor.parent / 'manifest.json').read_text())
    if identity['eigen_sha256'] != vendor_manifest['sha256']:
        raise RuntimeError('Vendored Eigen archive hash differs from provenance manifest')
    identity['vendor_manifest'] = vendor_manifest
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    build = ROOT / '.build/cutting2d' / digest
    build.mkdir(parents=True, exist_ok=True)
    binary = build / ('finite_jc_core' + sysconfig.get_config_var('EXT_SUFFIX'))
    if not binary.exists():
        with tempfile.TemporaryDirectory(prefix='build-', dir=build) as temporary:
            temporary = Path(temporary)
            with tarfile.open(vendor) as archive:
                archive.extractall(temporary, filter='data')
            eigen = temporary / 'eigen-3.4.0'
            candidate = temporary / binary.name
            command = compiler + flags + [f'-I{eigen}', f'-I{pybind11.get_include()}',
                        f'-I{sysconfig.get_paths()["include"]}', str(source), '-o', str(candidate)]
            result = subprocess.run(command, text=True, capture_output=True)
            (build / 'build.log').write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(f'Finite JC compilation failed: {build / "build.log"}')
            os.replace(candidate, binary)
            identity['command'] = command
            identity['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
            (build / 'manifest.json').write_text(json.dumps(identity, indent=2) + '\n')
    spec = importlib.util.spec_from_file_location('finite_jc_core', binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.build_manifest = str(build / 'manifest.json')
    return module


def _vector(value, n, name, *, positive=False):
    result = np.broadcast_to(np.asarray(value, dtype=np.float64), (n,)).copy()
    if not np.isfinite(result).all() or (positive and np.any(result <= 0)):
        raise ValueError(f'{name} must be finite' + (' and positive' if positive else ''))
    return result


@dataclass(frozen=True)
class MaterialState:
    """Detached, serializable batch. Tensor components are physical 3D tensors.

    Treat arrays as read-only; use trial/commit/rollback to replace this state.
    ``thermal_energy`` is relative to the initial temperature. Total internal
    energy is elastic + stored + thermal, not elastic + plastic + thermal.
    """
    material: dict
    raw_states: tuple
    diagnostics: tuple
    F: np.ndarray
    Fp: np.ndarray
    sigma: np.ndarray
    tau: np.ndarray
    J: np.ndarray
    alpha: np.ndarray
    alpha_dot: np.ndarray
    T: np.ndarray
    energy_densities: dict

    @property
    def temperature(self):
        return self.T

    @property
    def q_cauchy(self):
        return np.array([s['q_cauchy'] for s in self.raw_states])

    def energies_J(self, reference_volume):
        volume = _vector(reference_volume, len(self.T), 'reference_volume', positive=True)
        return {name: density * volume for name, density in self.energy_densities.items()}

    def serialize(self):
        return {'schema': 'cutting2d-material-v1', 'material': copy.deepcopy(self.material),
                'kernel_version': load_kernel().__version__,
                'states': copy.deepcopy(list(self.raw_states))}


def _pack(material, states, diagnostics=()):
    states = tuple(states)
    def values(name):
        return np.asarray([s[name] for s in states], dtype=np.float64)
    return MaterialState(copy.deepcopy(material), states, tuple(diagnostics),
                         *(values(name) for name in ('F', 'Fp', 'sigma', 'tau', 'J',
                                                     'alpha', 'alpha_dot', 'temperature')),
                         {name: values(name) for name in ENERGY_FIELDS})


def initialize(material, N, T=None):
    if not isinstance(N, (int, np.integer)) or N <= 0:
        raise ValueError('N must be a positive particle count')
    properties = kernel_material(material)
    temperatures = _vector(properties['REFERENCE_TEMPERATURE'] if T is None else T,
                           N, 'T', positive=True)
    if np.any(temperatures >= properties['MELD_TEMPERATURE']):
        raise MaterialFailure(int(np.argmax(temperatures)), 'solid_temperature_limit',
                              'Temperature reaches melting; the solid law is inapplicable')
    kernel = load_kernel()
    states = [kernel.initialize_state(properties, temperature=float(t)) for t in temperatures]
    return _pack(properties, states)


def trial(F, prior, dt, external_heat_J, reference_volume, *, options=None, optimize_unchanged=True):
    """Evaluate all particles without mutating ``prior``; fail the batch atomically.

    F is total deformation (N,3,3), dt seconds, heat J, volume initial m³. The
    local implicit return couples beta plastic heat with thermal softening.
    A final or externally heated T >= Tm is rejected, never clipped.
    """
    n = len(prior.T)
    deformation = np.asarray(F, dtype=np.float64)
    if deformation.shape != (n, 3, 3) or not np.isfinite(deformation).all():
        raise ValueError('F must have finite shape (N,3,3)')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    volume = _vector(reference_volume, n, 'reference_volume', positive=True)
    heat_density = _vector(external_heat_J, n, 'external_heat_J') / volume
    properties = prior.material
    base_T = prior.T + heat_density / (properties['DENSITY'] * properties['SPECIFIC_HEAT'])
    invalid = (base_T <= 0) | (base_T >= properties['MELD_TEMPERATURE'])
    if invalid.any():
        index = int(np.flatnonzero(invalid)[0])
        raise MaterialFailure(index, 'solid_temperature_limit', 'External heat leaves 0 < T < Tm')
    kernel = load_kernel()
    states, diagnostics = [], []
    unchanged = (np.all(deformation == prior.F, axis=(1, 2)) & (heat_density == 0.)
                 & (prior.alpha == 0.) & (prior.alpha_dot == 0.))
    if not optimize_unchanged or options:
        unchanged[:] = False
    for index in range(n):
        if unchanged[index]:
            # A virgin elastic state at exactly the same F and T is a fixed
            # point. Plastic states must still relax at the new strain rate.
            # Histories are treated as read-only; no constituent is modified.
            states.append(prior.raw_states[index])
            diagnostics.append({'code': 'ok', 'branch': 'elastic_unchanged',
                                'plastic_increment': 0., 'plastic_heat_increment_j_m3_reference': 0.})
            continue
        result = kernel.evaluate(deformation[index].tolist(), prior.raw_states[index],
                                 properties, float(dt), float(heat_density[index]), options or {})
        diagnostic = result['diagnostics']
        if not result['success']:
            raise MaterialFailure(index, diagnostic['code'], diagnostic.get('message', ''), diagnostic)
        state = result['state']
        if state['temperature'] >= properties['MELD_TEMPERATURE']:
            raise MaterialFailure(index, 'solid_temperature_limit',
                                  'Coupled plastic heating reaches melting', diagnostic)
        states.append(state)
        diagnostics.append(diagnostic)
    return _pack(properties, states, diagnostics)


def commit(candidate):
    """Detach a successful candidate; the caller owns global acceptance."""
    return _pack(candidate.material, copy.deepcopy(candidate.raw_states), copy.deepcopy(candidate.diagnostics))


def rollback(prior):
    """Detach the last committed state; there is no pending heat to replay."""
    return commit(prior)


def restore(serialized):
    if serialized.get('schema') != 'cutting2d-material-v1':
        raise ValueError('Unsupported material state schema')
    kernel = load_kernel()
    if serialized.get('kernel_version') != kernel.__version__:
        raise ValueError('Kernel version mismatch')
    properties = kernel_material(serialized['material'])
    # Session construction validates complete histories without advancing time.
    states = [kernel.Session(properties, state).state() for state in serialized['states']]
    if not states or any(not 0 < s['temperature'] < properties['MELD_TEMPERATURE'] for s in states):
        raise ValueError('Restored state outside the solid temperature range')
    return _pack(properties, states)


if __name__ == '__main__':
    print(load_kernel().build_manifest)
