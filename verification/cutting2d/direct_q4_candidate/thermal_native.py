"""Read-only thermal-stage native loader. Building is a separate explicit action.

All executable paths are relative to this project root and sealed by hashes.
No compiler query, compilation, package install, search fallback or file writes
occur here. A frozen runner must additionally pin its complete Python runtime.
"""
from pathlib import Path
from functools import lru_cache
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
import sysconfig

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FAMILY='thermal_fct'
MODULE='_native_fct_stage'
SCHEMA='thermal-fct-native-build-v1'
SELECTOR=Path('.build')/FAMILY/'selected.json'
SOURCES=('verification/cutting2d/direct_q4_candidate/thermal.py',
         'verification/cutting2d/direct_q4_candidate/thermal_native.py',
         'verification/cutting2d/direct_q4_candidate/thermal_native_build.py',
         'verification/cutting2d/direct_q4_candidate/native/thermal_stage.cpp')
POLICY=dict(arithmetic='IEEE754_binary64_original_order',stage='original_SSPRK2_FCT_stage_v1',
            cxx_standard='c++17',optimization='O3',fast_math=False,fp_contract='off',native_threads=1)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def runtime_abi():
    return dict(python=sys.version,implementation=sys.implementation.name,
        cache_tag=sys.implementation.cache_tag,SOABI=sysconfig.get_config_var('SOABI'),
        extension_suffix=sysconfig.get_config_var('EXT_SUFFIX'),platform=sys.platform,
        machine=platform.machine(),byteorder=sys.byteorder,
        numpy=importlib.metadata.version('numpy'),pybind11=importlib.metadata.version('pybind11'))


def relative_file(name):
    """No absolute, escaping or symlinked executable/dependency paths."""
    rel=Path(name)
    if rel.is_absolute() or not rel.parts or '..' in rel.parts:
        raise ValueError('Thermal native path must remain project-relative')
    target=ROOT/rel
    if target.is_symlink() or target.resolve()!=target.absolute() or not target.is_file():
        raise ValueError('Thermal native file absent or path redirected: '+str(rel))
    if not target.resolve().is_relative_to(ROOT):raise ValueError('Thermal native path escaped project')
    return target


def verify_native(files=None):
    """Verify without importing the extension; optional frozen source map is exact."""
    selector=relative_file(SELECTOR)
    selected=json.loads(selector.read_text())
    if selected.get('schema')!='thermal-fct-native-selection-v1':raise ValueError('Unknown thermal native selection')
    identity_hash=selected.get('identity_sha256')
    if not isinstance(identity_hash,str) or len(identity_hash)!=64 or any(c not in '0123456789abcdef' for c in identity_hash):
        raise ValueError('Invalid thermal native identity key')
    manifest_relative=Path('.build')/FAMILY/identity_hash/'manifest.json'
    if selected.get('manifest')!=str(manifest_relative):raise ValueError('Thermal native selection path differs')
    manifest=relative_file(manifest_relative)
    if sha(manifest)!=selected.get('manifest_sha256'):raise ValueError('Thermal native manifest changed')
    data=json.loads(manifest.read_text());identity=data.get('identity',{})
    if data.get('schema')!=SCHEMA or digest(identity)!=identity_hash or data.get('identity_sha256')!=identity_hash:
        raise ValueError('Thermal native build identity differs')
    if identity.get('policy')!=POLICY or identity.get('module')!=MODULE:
        raise ValueError('Thermal native arithmetic policy differs')
    if identity.get('runtime_abi')!=runtime_abi():raise ValueError('Thermal native runtime/ABI differs')
    source_hashes=identity.get('sources')
    if not isinstance(source_hashes,dict) or set(source_hashes)!=set(SOURCES):
        raise ValueError('Thermal native source identity is incomplete')
    sources={name:relative_file(name) for name in SOURCES}
    if any(sha(path)!=source_hashes[name] for name,path in sources.items()):
        raise ValueError('Thermal native source changed; build explicitly before freezing')
    expected_binary=manifest_relative.parent/(MODULE+sysconfig.get_config_var('EXT_SUFFIX'))
    if data.get('binary')!=str(expected_binary):raise ValueError('Thermal native binary path differs')
    binary=relative_file(expected_binary)
    if sha(binary)!=data.get('binary_sha256'):raise ValueError('Thermal native binary changed')
    paths={str(SELECTOR):selector,str(manifest_relative):manifest,str(expected_binary):binary,**sources}
    observed={name:sha(path) for name,path in paths.items()}
    if files is not None and any(files.get(name)!=value for name,value in observed.items()):
        raise ValueError('Thermal native executable missing/changed in frozen source identity')
    return dict(family=FAMILY,identity_sha256=identity_hash,binary=str(binary),
        binary_relative=str(expected_binary),binary_sha256=data['binary_sha256'],
        manifest=str(manifest),selection=str(selector),source_files=observed,
        rebuild_required=False,runtime_abi=identity['runtime_abi'],policy=POLICY.copy())


def runtime_files():
    """Absolute verified runtime artifacts for archiving/preflight; never build."""
    proof=verify_native()
    return [Path(proof[name]) for name in ('selection','manifest','binary')]


@lru_cache(maxsize=1)
def _load_binary(path,binary_hash,identity_hash):
    previous=sys.modules.get(MODULE)
    if previous is not None:
        if (Path(getattr(previous,'__file__','')).resolve()!=Path(path) or
                getattr(previous,'_thermal_identity_sha256',None)!=identity_hash or
                getattr(previous,'_thermal_binary_sha256',None)!=binary_hash):
            raise ImportError('Another thermal-stage binary is already loaded; use a fresh process')
        return previous
    spec=importlib.util.spec_from_file_location(MODULE,path)
    if spec is None or spec.loader is None:raise ImportError('No loader for verified thermal native binary')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if not hasattr(module,'FrozenStage'):raise ImportError('Thermal stage API is incomplete')
    module._thermal_identity_sha256=identity_hash;module._thermal_binary_sha256=binary_hash
    sys.modules[MODULE]=module
    return module


def load_native():
    """Load only the verified selected artifact; never build or silently fallback."""
    proof=verify_native()
    module=_load_binary(proof['binary'],proof['binary_sha256'],proof['identity_sha256'])
    module.build_manifest=proof['manifest'];module.build_selection=proof['selection']
    module.build_source_files=proof['source_files'].copy()
    return module


# Runner archive hooks can query the already loaded binary without triggering a build.
load_native.cache_info=_load_binary.cache_info
