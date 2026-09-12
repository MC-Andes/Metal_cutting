"""Freeze executable MPM/native sources; run fresh t=0 in a verified copy.

The Python environment is pinned and rechecked, not copied or made portable.
No checkpoint conversion, ROOT override, native rebuild or project fallback.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,sys
import importlib.abc,importlib.util,importlib.machinery,importlib.metadata
import platform,shlex,subprocess,sysconfig,ast

SCHEMA='mpm-frozen-source-run-v2'
PROJECTS=('verification','cutting2d','scripts')
BACKENDS=('dense','sparse_qr','certified_gram','certified_range','certified_range_fast','certified_range_robust')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def threads_env():
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'

def runtime_identity():
    """Pin shared Python/stdlib and complete installed numerical packages."""
    files={};packages={}
    for name in ('numpy','scipy','pybind11'):
        dist=importlib.metadata.distribution(name);packages[name]=dist.version
        for entry in dist.files or ():
            p=Path(dist.locate_file(entry)).resolve()
            if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts:files[str(p)]=sha(p)
    stdlib=Path(sysconfig.get_paths()['stdlib']).resolve()
    for directory,dirs,names in os.walk(stdlib):
        dirs[:]=[d for d in dirs if d not in ('site-packages','__pycache__')]
        for name in names:
            p=Path(directory)/name
            if p.suffix in ('.py','.so','.pyd','.dylib','.zip'):files[str(p.resolve())]=sha(p)
    executable=Path(sys.executable).resolve();files[str(executable)]=sha(executable)
    for p in (Path(sys.base_prefix)/'lib').glob('libpython*'):
        if p.is_file():files[str(p.resolve())]=sha(p)
    return dict(python=sys.version,platform=platform.platform(),executable=str(executable),packages=packages,files=files,
        scope='Pinned shared interpreter/stdlib and numpy/scipy/pybind11 files; OS libraries/hardware are not copied or certified')

class FrozenImports(importlib.abc.MetaPathFinder):
    def __init__(self,root,files):self.root=root;self.files=files
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] not in PROJECTS:return None
        relative=Path(*fullname.split('.'));directory=self.root/relative
        for candidate,package in ((directory/'__init__.py',True),(directory.with_suffix('.py'),False)):
            if candidate.is_file():
                rel=str(candidate.relative_to(self.root))
                if rel not in self.files or sha(candidate)!=self.files[rel]:raise ImportError('Changed/unarchived project import: '+fullname)
                return importlib.util.spec_from_file_location(fullname,candidate,submodule_search_locations=[str(directory)] if package else None)
        if directory.is_dir():
            spec=importlib.machinery.ModuleSpec(fullname,loader=None,is_package=True);spec.submodule_search_locations=[str(directory)];return spec
        raise ImportError('Project import absent from frozen runtime: '+fullname)

def native_preflight(root,files,backend):
    """Predict original loaders' cache choices before they can create a build."""
    import pybind11
    vendor=root/'constitutive/finite_jc/vendor/eigen-3.4.0.tar.gz';vh=sha(vendor)
    vm=json.loads((vendor.parent/'manifest.json').read_text());compiler=shlex.split(os.environ.get('CXX','c++'))
    version=subprocess.check_output(compiler+['--version'],text=True).splitlines()[0]
    flags=['-std=c++17','-O2','-shared','-fPIC','-fvisibility=hidden','-DEIGEN_MPL2_ONLY']
    if sys.platform=='darwin':flags+=['-undefined','dynamic_lookup']
    families=[('cutting2d','finite_jc_core'),('direct_q4','direct_q4_finite_jc')]
    if backend!='dense':families.append(('mpm_sparseqr','mpm_sparseqr_native'))
    report=[]
    for family,module in families:
        if family=='mpm_sparseqr':
            code=root/'verification/cutting2d/mpm_sparse_energy_candidate/qr.py'
            policy=next(ast.literal_eval(n.value) for n in ast.parse(code.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='POLICY' for t in n.targets))
            identity=dict(source_sha256=sha(code.with_name('native_sparseqr.cpp')),eigen_archive_sha256=vh,
                compiler=subprocess.check_output(['c++','--version'],text=True),python=sys.version,pybind11=pybind11.__version__,policy=policy)
        else:
            source=root/('constitutive/finite_jc/finite_jc_core.cpp' if family=='cutting2d' else 'verification/cutting2d/direct_q4_candidate/native/finite_jc_batch.cpp')
            identity=dict(source_sha256=sha(source),eigen_sha256=vh,pybind11=pybind11.__version__,python=sys.version,
                platform=platform.platform(),compiler=version,flags=flags)
            if family=='cutting2d':identity['vendor_manifest']=vm
            else:identity['historical_core_sha256']=sha(root/'constitutive/finite_jc/finite_jc_core.cpp')
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
        directory=root/'.build'/family/key;manifest=directory/'manifest.json';binary=directory/(module+sysconfig.get_config_var('EXT_SUFFIX'))
        for p in (manifest,binary):
            if not p.is_file() or files.get(str(p.relative_to(root)))!=sha(p):raise ValueError('Native cache mismatch; recompilation forbidden: '+family)
        data=json.loads(manifest.read_text())
        if any(data.get(k)!=v for k,v in identity.items()) or data['binary_sha256']!=sha(binary):raise ValueError('Native manifest differs: '+family)
        report.append(dict(family=family,binary=str(binary),sha256=sha(binary),cache=key,rebuild_required=False))
    # This read-only loader derives its root from the archived __file__. Never
    # leave this temporary module in the source inventory of the preparer.
    thermal_path=root/'verification/cutting2d/direct_q4_candidate/thermal_native.py'
    if files.get(str(thermal_path.relative_to(root)))!=sha(thermal_path):
        raise ValueError('Thermal loader source differs before preflight')
    spec=importlib.util.spec_from_file_location('_frozen_thermal_preflight',thermal_path)
    thermal=importlib.util.module_from_spec(spec)
    saved=sys.modules.get(spec.name);sys.modules[spec.name]=thermal
    try:
        spec.loader.exec_module(thermal)
        checked=thermal.verify_native(files=files)
        report.append(dict(family='thermal_fct',verification=checked,rebuild_required=False))
    finally:
        if saved is None:sys.modules.pop(spec.name,None)
        else:sys.modules[spec.name]=saved
    return report

def prepare(folder,backend,case_path=None):
    if backend not in BACKENDS:raise ValueError('Unsupported frozen backend')
    threads_env();sys.dont_write_bytecode=True
    if any(n.split('.')[0] in PROJECTS for n in sys.modules):raise RuntimeError('Prepare requires a fresh process before project imports')
    project=Path(__file__).resolve().parents[4];sys.path.insert(0,str(project))
    from verification.cutting2d.cpdi_mpm_candidate import run,implicit
    from verification.cutting2d.direct_q4_candidate import geometry,constitutive
    from cutting2d import material
    from verification.cutting2d.direct_q4_candidate import thermal_native,thermal_native_build
    thermal_native_build.build_native();thermal_native.load_native()
    material.load_kernel();constitutive.load_native()
    if backend!='dense':
        from verification.cutting2d.mpm_sparse_energy_candidate.qr import load_native
        load_native()
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=False)
    identity,changed=run.archive(folder,backend=backend)
    if changed:raise RuntimeError('Executable sources changed during freeze')
    root=folder/'source_archive';vendor=Path('constitutive/finite_jc/vendor/eigen-3.4.0.tar.gz')
    vh=sha(run.ROOT/vendor);expected=json.loads((run.ROOT/vendor.parent/'manifest.json').read_text())['sha256']
    if vh!=expected:raise ValueError('Vendored dependency hash mismatch')
    target=root/vendor;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(run.ROOT/vendor,target)
    files=identity['files'];native=native_preflight(root,files,backend);environment=runtime_identity()
    # Detect source mutation across the entire freeze, not just the copy loop.
    if any(sha(run.ROOT/name)!=h or sha(root/name)!=h for name,h in files.items()):raise RuntimeError('Executable source changed while preparing runtime')
    case=None
    if case_path:
        path=Path(case_path).resolve();value=json.loads(path.read_text());(folder/'case.json').write_bytes(path.read_bytes())
        case=dict(path=str(folder/'case.json'),sha256=sha(path),source=str(path),physical_case_digest=digest(value))
    spec=dict(schema=SCHEMA,source_root=str(root),source_identity=identity,supplemental_files={str(vendor):vh},backend=backend,
        launcher_relative=str(Path(__file__).resolve().relative_to(run.ROOT)),launcher_sha256=sha(__file__),environment=environment,
        native_preflight=native,case=case,purpose='Fresh t=0 trajectory; no checkpoint migration')
    spec['metadata_sha256']=digest(spec);write(folder/'runtime.json',spec)
    for p in root.rglob('*'):
        if p.is_file():p.chmod(0o444)
    for p in sorted([p for p in root.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):p.chmod(0o555)
    root.chmod(0o555)
    if case:Path(case['path']).chmod(0o444)
    print(json.dumps(dict(prepared=str(folder),sources=len(files),backend=backend,source_sha256=identity['source_sha256'],simulation_steps=0),indent=2))

def verified_spec(folder):
    if not sys.flags.isolated:raise RuntimeError('Execute with Python -I to exclude mutable project search paths')
    folder=Path(folder).resolve();spec=json.loads((folder/'runtime.json').read_text());unsigned={k:v for k,v in spec.items() if k!='metadata_sha256'}
    if spec.get('schema')!=SCHEMA or spec.get('metadata_sha256')!=digest(unsigned):raise ValueError('Frozen runtime metadata changed or schema differs')
    if sha(__file__)!=spec['launcher_sha256']:raise ValueError('Frozen launcher changed')
    root=Path(spec['source_root']).resolve()
    if root!=folder/'source_archive':raise ValueError('Frozen source ROOT mismatch')
    identity=spec['source_identity'];files=identity['files']
    if identity['source_sha256']!=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest():raise ValueError('Frozen source identity checksum differs')
    for name,h in {**files,**spec['supplemental_files']}.items():
        rel=Path(name);path=root/rel
        if rel.is_absolute() or '..' in rel.parts or not path.resolve().is_relative_to(root) or not path.is_file() or sha(path)!=h:raise ValueError('Frozen runtime source/dependency changed: '+name)
    return folder,spec,root

def execute(folder,destination,case_path,grid_h,material_h,dt,end,wall,threads,startup_dt,check_only=False):
    threads_env();sys.dont_write_bytecode=True
    folder,spec,root=verified_spec(folder)
    frozen=root/spec['launcher_relative']
    if Path(__file__).resolve()!=frozen:
        # Include the actually executing launcher in the same source identity
        # produced during prepare. No module/file identity is forged.
        args=[sys.executable,'-I',str(frozen),'check' if check_only else 'run',str(folder),
            '--grid-h',str(grid_h),'--material-h',str(material_h),'--dt',str(dt),'--end',str(end),
            '--wall',str(wall),'--threads',str(threads),'--startup-dt',str(startup_dt)]
        if destination:args+=['--destination',str(Path(destination).resolve())]
        if case_path:args+=['--case',str(Path(case_path).resolve())]
        os.execv(sys.executable,args)
    if any(n.split('.')[0] in PROJECTS for n in sys.modules):raise RuntimeError('Project modules loaded before frozen import guard')
    if runtime_identity()!=spec['environment']:raise ValueError('Frozen numerical runtime files or versions changed')
    sys.meta_path.insert(0,FrozenImports(root,spec['source_identity']['files']));sys.path.insert(0,str(root))
    native=native_preflight(root,spec['source_identity']['files'],spec['backend'])
    from verification.cutting2d.cpdi_mpm_candidate import run,implicit
    from verification.cutting2d.direct_q4_candidate import geometry,constitutive
    from cutting2d import material
    from verification.cutting2d.direct_q4_candidate import thermal_native
    if run.ROOT!=root or constitutive.ROOT!=root or material.ROOT!=root:raise RuntimeError('Loaded solver escaped frozen source ROOT')
    material.load_kernel();constitutive.load_native();thermal_native.load_native()
    if spec['backend']!='dense':
        from verification.cutting2d.mpm_sparse_energy_candidate.qr import load_native
        load_native()
    observed={str(p.relative_to(root)):sha(p) for p in run._source_files(spec['backend'])}
    if observed!=spec['source_identity']['files']:raise ValueError('Loaded executable source set differs from frozen identity')
    chosen=Path(case_path or (spec['case'] or {}).get('path','')).resolve()
    if not chosen.is_file():raise ValueError('Supply --case or prepare with a frozen case')
    case=json.loads(chosen.read_text());case_sha=sha(chosen)
    if spec['case'] and (case_sha!=spec['case']['sha256'] or digest(case)!=spec['case']['physical_case_digest']):raise ValueError('Physical case differs from frozen case')
    invocation=run._invocation(grid_h,material_h,None,spec['backend'],dt,end,wall,threads,'cached','hybrid','uniform',None,5e-9,startup_dt)
    report=dict(accepted=False,preflight_passed=True,check_only=check_only,source_root=str(root),source_identity_exact=True,
        source_sha256=spec['source_identity']['source_sha256'],case=str(chosen),case_sha256=case_sha,invocation=invocation,native=native,
        environment_digest=digest(spec['environment']),no_ROOT_override=True,no_recompile=True,fresh_t0=True)
    if check_only:write(folder/'check.json',report);print(json.dumps(report,indent=2));return report
    if not destination:raise ValueError('A new --destination is required')
    destination=Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(folder):raise ValueError('Destination must be new and outside frozen runtime')
    try:
        result=run.run(destination,case,grid_h=grid_h,material_h=material_h,backend=spec['backend'],dt_max=dt,t_end=end,
            wall_seconds=wall,threads=threads,integrator='hybrid',startup_time=5e-9,startup_dt=startup_dt)
    finally:
        if destination.exists():write(destination/'launcher_preflight.json',report)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','check','run']);p.add_argument('runtime')
    p.add_argument('--backend',choices=BACKENDS,default='certified_range');p.add_argument('--destination');p.add_argument('--case')
    p.add_argument('--grid-h',type=float,default=100e-6);p.add_argument('--material-h',type=float,default=25e-6)
    p.add_argument('--dt',type=float,default=500e-9);p.add_argument('--end',type=float,default=1.2e-3)
    p.add_argument('--wall',type=float,default=1800.);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--startup-dt',type=float,default=.0125e-9)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.runtime,a.backend,a.case)
    else:execute(a.runtime,a.destination,a.case,a.grid_h,a.material_h,a.dt,a.end,a.wall,a.threads,a.startup_dt,a.mode=='check')
