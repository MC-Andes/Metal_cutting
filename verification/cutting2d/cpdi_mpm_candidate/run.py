"""Execute/checkpoint the MPM candidate; every output remains unaccepted.

JSON, NPZ and compensated-event utilities follow the independent direct-Q4
runner. This module has its own invocation and checkpoint schemas and only
dispatches MPM momentum solvers. Background caches are not material history.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
import traceback
import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[3]
CHECKPOINT_SCHEMA='cpdi-mpm-candidate-checkpoint-v2'
SOURCE_SCHEMA='cpdi-mpm-runtime-source-identity-v1'
QR_SOURCES=('verification/cutting2d/mpm_sparse_energy_candidate/qr.py',
            'verification/cutting2d/mpm_sparse_energy_candidate/native_sparseqr.cpp')
CERTIFIED_SOURCES=('verification/cutting2d/mpm_sparse_energy_candidate/accelerated.py',
    'verification/cutting2d/mpm_sparse_energy_candidate/rank_certificate.py',
    'verification/cutting2d/cpdi_mpm_candidate/cache_identity.py')
RANGE_SOURCES=('verification/cutting2d/mpm_sparse_energy_candidate/reduced.py',)
FAST_RANGE_SOURCES=('verification/cutting2d/mpm_sparse_energy_candidate/fast_range.py',
    'verification/cutting2d/mpm_sparse_energy_candidate/gram_certificate.py')
ROBUST_RANGE_SOURCES=('verification/cutting2d/mpm_sparse_energy_candidate/robust.py',)
RANGE_BACKENDS=('certified_range','certified_range_fast','certified_range_robust')
FAST_RANGE_BACKENDS=('certified_range_fast','certified_range_robust')
CERTIFIED_BACKENDS=('certified_gram',*RANGE_BACKENDS)
QR_BACKENDS=('sparse_qr',*CERTIFIED_BACKENDS)
BACKENDS=('dense',*QR_BACKENDS)
STRICT_INVOCATION=('grid_h_m','material_h_m','origin_m','backend','dt_max_s','threads',
                   'geometry_backend','integrator','mesh_profile','constitutive_options',
                   'startup_time_s','startup_dt_s')
REQUIRED_SOURCES=tuple('verification/cutting2d/cpdi_mpm_candidate/'+name for name in
    ('run.py','solver.py','implicit.py','mechanics.py','transfer.py','support.py','contact.py','reactions.py'))+(
    'cutting2d/transfer.py','cutting2d/material.py','cutting2d/cases.py','cutting2d/config.py',
    'verification/cutting2d/coulomb_scaling_candidate/solver.py',
    'verification/cutting2d/direct_q4_candidate/solver.py',
    'verification/cutting2d/direct_q4_candidate/mechanics.py',
    'verification/cutting2d/direct_q4_candidate/constitutive.py',
    'verification/cutting2d/direct_q4_candidate/thermal.py',
    'verification/cutting2d/direct_q4_candidate/thermal_native.py',
    'verification/cutting2d/direct_q4_candidate/thermal_native_build.py',
    'verification/cutting2d/direct_q4_candidate/native/thermal_stage.cpp',
    '.build/thermal_fct/selected.json',
    'verification/cutting2d/direct_q4_candidate/thermal_subcycling.py',
    'verification/cutting2d/direct_q4_candidate/coupling.py',
    'verification/cutting2d/direct_q4_candidate/native/finite_jc_batch.cpp',
    'verification/cutting2d/direct_q4_candidate/native/batch_binding.cpp.inc',
    'constitutive/finite_jc/finite_jc_core.cpp','constitutive/finite_jc/vendor/manifest.json')


def plain(value):
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,np.generic):return plain(value.item())
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return str(value)
    return value


def write(path,value):
    """Replace complete JSON atomically; a STOP cannot leave half a checkpoint."""
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def sha(data):return hashlib.sha256(data).hexdigest()


def step_to_event(clock,compensation,dt_cap,target):
    remaining=(target-clock)+compensation
    tolerance=8*np.finfo(float).eps*max(abs(target),abs(clock),abs(dt_cap),np.finfo(float).tiny)
    return remaining if remaining<=dt_cap+tolerance else dt_cap


def _source_files(backend=None):
    files={ROOT/name for name in REQUIRED_SOURCES}
    # The checkpoint schema also seals the historical material dependency.
    # Stable batch initialization no longer loads it as a side effect.
    from cutting2d.material import load_kernel
    load_kernel()
    # Constant/k=0 states may never instantiate a native thermal stage. Their
    # checkpoint still needs the verified selected runtime for a later step.
    from verification.cutting2d.direct_q4_candidate import thermal_native
    files.update(thermal_native.runtime_files())
    if backend in QR_BACKENDS:files.update(ROOT/name for name in QR_SOURCES)
    if backend in CERTIFIED_BACKENDS:files.update(ROOT/name for name in CERTIFIED_SOURCES)
    if backend in RANGE_BACKENDS:files.update(ROOT/name for name in RANGE_SOURCES)
    if backend in FAST_RANGE_BACKENDS:files.update(ROOT/name for name in FAST_RANGE_SOURCES)
    if backend=='certified_range_robust':files.update(ROOT/name for name in ROBUST_RANGE_SOURCES)
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if not name:continue
        path=Path(name).resolve()
        if path.suffix=='.pyc':
            try:path=Path(importlib.util.source_from_cache(str(path)))
            except ValueError:pass
        if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'.venv') and path.is_file():files.add(path)
        if path.is_relative_to(ROOT/'.build') and (path.parent/'manifest.json').is_file():files.add(path.parent/'manifest.json')
    files.update(path for path in (ROOT/'verification/cutting2d/direct_q4_candidate/native').glob('*') if path.is_file())
    # Native modules loaded through module_from_spec need not be represented
    # reliably by an ordinary import entry. Capture each already loaded cache.
    factories=(('verification.cutting2d.direct_q4_candidate.constitutive','load_native'),
               ('verification.cutting2d.direct_q4_candidate.thermal_native','load_native'),
               ('cutting2d.material','load_kernel'),
               ('verification.cutting2d.mpm_sparse_energy_candidate.qr','load_native'))
    for name,attribute in factories:
        module=sys.modules.get(name);factory=getattr(module,attribute,None)
        if factory is None or not factory.cache_info().currsize:continue
        native=factory();binary=Path(native.__file__).resolve();files.add(binary)
        if (binary.parent/'manifest.json').is_file():files.add(binary.parent/'manifest.json')
        if name.endswith('.thermal_native'):files.update(module.runtime_files())
        if name.endswith('.qr'):files.update(ROOT/name for name in QR_SOURCES)
    return files


def archive(destination,previous=None,*,backend=None):
    """Include late imports; preserve the first archived bytes of every source."""
    destination=Path(destination);result={} if previous is None else dict(previous['files'])
    changed=[]
    for path in sorted(_source_files(backend) | {ROOT/name for name in result}):
        relative=str(path.relative_to(ROOT))
        if not path.is_file():
            if relative in result:changed.append(relative);continue
            raise ValueError('Required MPM executable source missing: '+relative)
        data=path.read_bytes();digest=sha(data)
        if relative in result:
            if digest!=result[relative]:changed.append(relative)
            continue
        target=destination/'source_archive'/relative
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);result[relative]=digest
    identity=dict(schema=SOURCE_SCHEMA,files=result,
        source_sha256=sha(json.dumps(result,sort_keys=True).encode()),
        python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__)
    write(destination/'source_identity.json',identity)
    return identity,sorted(set(changed))


def output(destination,name,s):
    m=s.model;r=s.material.state;path=Path(destination)/(name+'.npz');temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,formulation='MPM_shared_domain_B2_four_persistent_GP_v1',accepted=False,
            grid_h_m=s.grid_h,material_h_m=s.material_h,origin_m=s.origin,backend=s.backend,
            time_s=s.time,step=s.step,X=m.X,cells=m.cells,q=s.q,v=s.v,
            displacement_hi=s.displacement_hi,displacement_lo=s.displacement_lo,
            mass_gp=m.mass,volume0=m.volume0,point_positions=m.point_positions(s.q),
            F=r.F,Fp=r.Fp,sigma=r.sigma,tau=r.tau,J=r.J,alpha=r.alpha,alpha_dot=r.alpha_dot,T=r.T,
            **{'energy_'+key:value for key,value in r.energies_J(m.volume0).items()})
    temporary.replace(path)


def physical_record(destination,row):
    """Keep measured wall time outside persistent physical diagnostics."""
    measurements={};timers={'mesh_validation_seconds','native_factorization_seconds','construction_seconds',
        'current_gram_factorization_seconds','qr_factorization_seconds','elapsed_s','wall_seconds',
        'formation_s','LU_s','inverse_check_s','dependent_check_s','qr_construction_seconds'}
    def split(value,path=()):
        if isinstance(value,dict):
            result={}
            for key,item in value.items():
                # Its checksum includes original certificate/cache wall times.
                if key=='operator_verification_cache':result[key]=item;continue
                if key in timers:measurements['/'.join(map(str,(*path,key)))]=float(item)
                else:result[key]=split(item,(*path,key))
            return result
        if isinstance(value,(list,tuple)):return [split(item,(*path,index)) for index,item in enumerate(value)]
        return value
    result=split(row)
    if measurements:
        with (Path(destination)/'timing_live.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=row['step'],time_s=row['time_s'],
                measured_seconds=measurements),allow_nan=False)+'\n')
    return result


def mesh_identity(model):
    return {name:dict(shape=list(value.shape),dtype=value.dtype.str,sha256=sha(value.tobytes()))
            for name in ('X','cells','mass','volume0','reference_points')
            for value in (np.ascontiguousarray(getattr(model,name)),)}


def _validate_source_identity(identity,checkpoint_directory,backend):
    if identity.get('schema')!=SOURCE_SCHEMA:raise ValueError('Checkpoint source identity schema differs')
    for key,value in (('python',sys.version),('platform',platform.platform()),('numpy',np.__version__),('scipy',scipy.__version__)):
        if identity.get(key)!=value:raise ValueError('Checkpoint numerical runtime differs: '+key)
    files=identity['files']
    if not isinstance(files,dict) or any(name not in files for name in REQUIRED_SOURCES):
        raise ValueError('Checkpoint source identity omits a required MPM source')
    if backend in QR_BACKENDS and any(name not in files for name in QR_SOURCES):
        raise ValueError('Checkpoint source identity omits a required sparse QR source')
    if backend in CERTIFIED_BACKENDS and any(name not in files for name in CERTIFIED_SOURCES):
        raise ValueError('Checkpoint source identity omits a required certified Gram source')
    if backend in RANGE_BACKENDS and any(name not in files for name in RANGE_SOURCES):
        raise ValueError('Checkpoint source identity omits a required certified range source')
    if backend in FAST_RANGE_BACKENDS and any(name not in files for name in FAST_RANGE_SOURCES):
        raise ValueError('Checkpoint source identity omits a required fast range certificate source')
    if backend=='certified_range_robust' and any(name not in files for name in ROBUST_RANGE_SOURCES):
        raise ValueError('Checkpoint source identity omits a required robust range solver source')
    if identity['source_sha256']!=sha(json.dumps(files,sort_keys=True).encode()):
        raise ValueError('Checkpoint aggregated source hash mismatch')
    prefixes=('.build/direct_q4/','.build/cutting2d/','.build/thermal_fct/')+(('.build/mpm_sparseqr/',) if backend in QR_BACKENDS else ())
    for prefix in prefixes:
        if not any(name.startswith(prefix) and Path(name).suffix in ('.so','.pyd','.dylib') for name in files):
            raise ValueError('Checkpoint source identity omits loaded native binary: '+prefix)
        if not any(name.startswith(prefix) and Path(name).name=='manifest.json' for name in files):
            raise ValueError('Checkpoint source identity omits native build manifest: '+prefix)
    for name,digest in files.items():
        relative=Path(name);target=(ROOT/relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not target.is_relative_to(ROOT):
            raise ValueError('Invalid checkpoint executable source path')
        if not target.is_file() or sha(target.read_bytes())!=digest:
            raise ValueError('Checkpoint executable source differs: '+name)
        archived=Path(checkpoint_directory)/'source_archive'/relative
        if not archived.is_file() or sha(archived.read_bytes())!=digest:
            raise ValueError('Checkpoint archived source differs: '+name)


def read_checkpoint(path,case,invocation):
    path=Path(path);envelope=json.loads(path.read_text());payload=envelope['state']
    canonical=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False)
    if envelope['sha256']!=sha(canonical.encode()):raise ValueError('Checkpoint content hash mismatch')
    if payload.get('schema')!=CHECKPOINT_SCHEMA:raise ValueError('Unknown MPM checkpoint schema')
    if payload.get('accepted') is not False:raise ValueError('Checkpoint must remain unaccepted')
    _validate_source_identity(payload['source_identity'],path.parent,payload['invocation']['backend'])
    if payload.get('sources_changed_during_run'):raise ValueError('Checkpoint run had changed executable sources')
    if payload['case']!=case:raise ValueError('Checkpoint physical case differs')
    for key in STRICT_INVOCATION:
        if payload['invocation'][key]!=invocation[key]:raise ValueError('Checkpoint MPM discretization differs: '+key)
    if not 0<=payload['time_s']<invocation['t_end_s']:raise ValueError('Resume target must follow the checkpoint time')
    return payload


def _roundoff_diagnostics(s):
    return {key:value for key,value in s.model.diagnostics.items()
            if key.startswith(('maximum_displacement_','maximum_display_position_'))}


def restore(s,payload):
    """Validate material and opaque operator proof before publishing state.

    The solver cache restore must construct all references before assigning
    its cache dictionary. Current grid coefficients/LU are never restored.
    """
    from verification.cutting2d.direct_q4_candidate.constitutive import ConstitutiveBatch
    from cutting2d.cases import make_tool
    if payload.get('schema')!=CHECKPOINT_SCHEMA:raise ValueError('Unknown MPM checkpoint schema')
    operator_cache=payload.get('operator_verification_cache')
    if not isinstance(operator_cache,dict) or set(operator_cache)-{'free','supported'}:
        raise ValueError('Checkpoint operator verification cache schema differs')
    if payload['invocation']['backend']!=s.backend:
        raise ValueError('Checkpoint operator verification backend differs')
    if s.backend not in CERTIFIED_BACKENDS and operator_cache:
        raise ValueError('Checkpoint operator verification cache requires certified Gram')
    if payload['material_mesh_identity']!=mesh_identity(s.model):raise ValueError('Checkpoint material quadrature differs')
    if payload['initial_energy_J']!=s.initial_energy:raise ValueError('Checkpoint initial physical energy differs')
    q=np.asarray(payload['q'],float);v=np.asarray(payload['v'],float)
    if q.shape!=s.q.shape or v.shape!=s.v.shape or not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError('Checkpoint material state has invalid shape or values')
    if np.any(q[s.fixed]!=s.model.X[s.fixed]) or np.any(v[s.fixed]!=0):raise ValueError('Checkpoint material support moved')
    material=ConstitutiveBatch.restore(payload['material'],threads=s.material.threads)
    if dict(material.material)!=dict(s.material.material) or dict(material.options)!=dict(s.material.options):
        raise ValueError('Checkpoint constitutive material or numerical options differ')
    hi=np.asarray(payload['displacement_hi'],float);lo=np.asarray(payload['displacement_lo'],float)
    old_diagnostics=s.model.diagnostics.copy()
    try:
        F,_=s.model.deformation_displacement(hi,lo)
        if not np.array_equal(s.model.display_positions(hi,lo),q):
            raise ValueError('Checkpoint geometry and displacement deformation representation disagree')
    finally:s.model.diagnostics=old_diagnostics
    if np.any(hi[s.fixed]!=0) or np.any(lo[s.fixed]!=0):raise ValueError('Checkpoint canonical support moved')
    if material.N!=s.model.point_count or not np.array_equal(F,material.state.F):
        raise ValueError('Checkpoint geometry and material deformation disagree')
    if np.min(s.conductor.corner_temperatures(material.state.T))<=0:
        raise ValueError('Checkpoint thermal polynomial is nonpositive')
    records=payload['records'];step=payload['step'];clock=payload['time_s'];compensation=payload['clock_compensation']
    if (type(step)!=int or step<0 or len(records)!=step or not np.isfinite(clock) or
        not np.isfinite(compensation) or not 0<payload['dt_next']<=payload['invocation']['dt_max_s']):
        raise ValueError('Checkpoint controller is inconsistent')
    previous=0.;correction=0.
    for index,row in enumerate(records):
        if not np.isfinite(row['dt_s']) or row['dt_s']<=0:raise ValueError('Checkpoint trajectory has invalid dt')
        increment=row['dt_s']-correction;updated=previous+increment;correction=(updated-previous)-increment
        if row['step']!=index+1 or row['time_s']!=updated or not row['integrator'].startswith('MPM_'):
            raise ValueError('Checkpoint trajectory has an inconsistent MPM physical clock')
        previous=updated
    if previous!=clock or correction!=compensation:raise ValueError('Checkpoint final compensated clock disagrees')
    diagnostics=payload['material_roundoff_diagnostics']
    if any(not key.startswith(('maximum_displacement_','maximum_display_position_')) or
           not np.isfinite(value) or value<0 for key,value in diagnostics.items()):
        raise ValueError('Checkpoint material roundoff diagnostics differ')
    s.geometry.validate(q,make_tool(s.config,clock),penetration_tolerance=s.penetration_tolerance)
    # Validate/reconstruct proofs first. This method is atomic on failure.
    # Do not pass the payload through timing extraction: its checksum is opaque.
    s.restore_operator_cache(operator_cache)
    s.reject();s.material=material;s.q=q.copy();s.v=v.copy();s.time=clock;s.clock_compensation=compensation;s.step=step
    s.displacement_hi=hi.copy();s.displacement_lo=lo.copy();s.bottom_trace=None
    s.model.diagnostics.update(diagnostics)


def _invocation(grid_h,material_h,origin,backend,dt_max,t_end,wall_seconds,threads,
                geometry_backend,integrator,mesh_profile,constitutive_options,startup_time,startup_dt):
    grid_h=float(grid_h);material_h=grid_h/2 if material_h is None else float(material_h)
    origin=np.zeros(2) if origin is None else np.asarray(origin,float)
    for name,value in (('grid_h',grid_h),('material_h',material_h),('dt_max',dt_max),('t_end',t_end)):
        if not np.isfinite(value) or value<=0:raise ValueError('Positive finite '+name+' required')
    if not np.isfinite(wall_seconds) or wall_seconds<0:raise ValueError('Nonnegative finite wall limit required')
    if origin.shape!=(2,) or not np.isfinite(origin).all():raise ValueError('Finite background origin required')
    if type(threads)!=int or not 1<=threads<=32:raise ValueError('Native thread count must be an integer in [1,32]')
    if mesh_profile!='uniform':raise ValueError('MPM runner currently supports only uniform material domains')
    if integrator not in ('explicit','implicit','hybrid'):raise ValueError('Unknown MPM time integrator')
    if integrator=='hybrid':
        startup_time=5e-9 if startup_time is None else float(startup_time)
        startup_dt=.0125e-9 if startup_dt is None else float(startup_dt)
        if not np.isfinite(startup_time) or startup_time<0:raise ValueError('Nonnegative finite startup time required')
        if not np.isfinite(startup_dt) or not 0<startup_dt<=dt_max:raise ValueError('Startup dt must be positive and no greater than dt_max')
    else:
        startup_time=0. if startup_time is None else float(startup_time)
        startup_dt=0. if startup_dt is None else float(startup_dt)
        if startup_time!=0. or startup_dt!=0.:raise ValueError('Nonzero startup controls require the hybrid integrator')
    if backend not in BACKENDS:raise ValueError('Unknown MPM momentum backend')
    if geometry_backend not in ('cached','historical'):raise ValueError('Unknown geometry backend')
    return dict(schema='cpdi-mpm-run-invocation-v1',grid_h_m=grid_h,material_h_m=material_h,
        origin_m=origin.tolist(),backend=backend,dt_max_s=float(dt_max),t_end_s=float(t_end),
        wall_seconds=float(wall_seconds),threads=threads,geometry_backend=geometry_backend,
        integrator=integrator,mesh_profile=mesh_profile,
        startup_time_s=startup_time,startup_dt_s=startup_dt,
        constitutive_options={'relative_tolerance':1e-13} if constitutive_options is None else dict(constitutive_options))


def run(destination,case,*,grid_h,material_h=None,origin=None,backend='dense',dt_max,t_end,
        wall_seconds=120.,threads=1,geometry_backend='cached',mesh_profile='uniform',integrator='explicit',
        resume_from=None,stop_after_steps=None,constitutive_options=None,startup_time=None,startup_dt=None):
    invocation=_invocation(grid_h,material_h,origin,backend,dt_max,t_end,wall_seconds,threads,
        geometry_backend,integrator,mesh_profile,constitutive_options,startup_time,startup_dt)
    if stop_after_steps is not None and (type(stop_after_steps)!=int or stop_after_steps<1):
        raise ValueError('Positive absolute pause step required')
    interval=case['output']['interval']
    if not np.isfinite(interval) or interval<=0:raise ValueError('Positive finite output interval required')
    payload=read_checkpoint(resume_from,case,invocation) if resume_from else None
    if integrator=='explicit':
        from .solver import MPMCutting as solver_class
    else:
        from .implicit import ImplicitMPMCutting as solver_class
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    write(destination/'case.json',case);write(destination/'invocation.json',invocation)
    started=time.perf_counter()
    s=solver_class(case,grid_h=invocation['grid_h_m'],material_h=invocation['material_h_m'],
        origin=invocation['origin_m'],backend=backend,threads=threads,geometry_backend=geometry_backend,
        constitutive_options=invocation['constitutive_options'])
    if backend in QR_BACKENDS:
        from verification.cutting2d.mpm_sparse_energy_candidate.qr import load_native
        load_native()  # Seal the selected binary/source before the first trial.
    write(destination/'mesh.json',s.mesh_diagnostics)
    identity,changed=archive(destination,backend=backend)
    if payload is not None:
        restore(s,payload)
        write(destination/'resume.json',dict(from_checkpoint=str(Path(resume_from).resolve()),
            checkpoint_sha256=sha(Path(resume_from).read_bytes()),start_time_s=s.time,start_step=s.step))
    output(destination,'resumed' if payload is not None else 'initial',s)
    records=[] if payload is None else list(payload['records'])
    rejected=[] if payload is None else list(payload['rejections'])
    startup_time=invocation['startup_time_s'];startup_dt=invocation['startup_dt_s']
    dt_next=min(dt_max,startup_dt) if integrator=='hybrid' and startup_time>0 else dt_max
    if payload is not None:dt_next=payload['dt_next']
    status='completed_unverified';failure=None;discarded_trial=None
    output_index=int(s.time//interval)
    while (output_index+1)*interval<=s.time:output_index+=1
    next_output=min((output_index+1)*interval,t_end)

    def pause_status():
        if (destination/'STOP').exists():return 'paused_requested'
        if stop_after_steps is not None and s.step>=stop_after_steps:return 'paused_step_limit'
        if time.perf_counter()-started>=wall_seconds:return 'paused_wall_limit'
        return None

    try:
        while s.time<t_end:
            paused=pause_status()
            if paused:status=paused;break
            landmarks=[t_end,next_output]+([5e-9] if s.time<5e-9 else [])
            if integrator=='hybrid' and s.time<startup_time:landmarks.append(startup_time)
            target=min(value for value in landmarks if value>s.time)
            startup=integrator=='hybrid' and s.time<startup_time
            dt_cap=min(dt_next,startup_dt) if startup else dt_next
            dt=step_to_event(s.time,s.clock_compensation,dt_cap,target)
            trial=None
            for attempt in range(case['solver']['max_retries']+1):
                paused=pause_status()
                if paused:status=paused;break
                try:
                    if startup:
                        from .solver import MPMCutting
                        trial=MPMCutting.trial(s,dt)
                    else:trial=s.trial(dt)
                except Exception as error:
                    s.reject();rejected.append(dict(step=s.step,time_s=s.time,dt_s=dt,attempt=attempt,
                        reason=str(error),type=type(error).__name__))
                    with (destination/'rejections_live.jsonl').open('a') as log:
                        log.write(json.dumps(plain(rejected[-1]),allow_nan=False)+'\n')
                    if attempt==case['solver']['max_retries'] or dt/2<case['solver']['dt_min']:raise
                    dt/=2;continue
                paused=pause_status()
                if paused:
                    s.reject();trial=None;status=paused
                    discarded_trial=dict(time_s=s.time,dt_s=dt,reason=paused)
                break
            if trial is None:break
            row=physical_record(destination,s.commit(trial));row['event_step_roundoff_extension_s']=max(0.,dt-dt_cap)
            records.append(row);dt_next=min(dt_max,1.25*dt)
            if s.time>=next_output:
                output_index+=1;output(destination,f'state_{output_index:05}',s)
                next_output=min((output_index+1)*interval,t_end)
            if s.step%100==0:
                write(destination/'progress.json',dict(status='running_unverified',accepted=False,step=s.step,time_s=s.time,
                    tool_advance_m=s.time*case['geometry']['tool_speed'],wall_s=time.perf_counter()-started,
                    rejected=len(rejected),alpha_max=row['alpha_max'],T_max_K=row['T_max_K']))
                write(destination/'controller_live.json',dict(dt_next_s=dt_next,dt_last_s=dt,
                    last_rejection=rejected[-1] if rejected else None))
    except Exception as error:
        s.reject();status='failed';failure=dict(type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
    identity,changed=archive(destination,identity,backend=backend)
    if changed and failure is None:
        status='failed_sources_changed';failure=dict(type='SourceChanged',reason='Executable sources changed during run')
    output(destination,'final',s)
    checkpoint=dict(schema=CHECKPOINT_SCHEMA,accepted=False,source_identity=identity,sources_changed_during_run=changed,
        case=case,invocation=invocation,material_mesh_identity=mesh_identity(s.model),initial_energy_J=s.initial_energy,
        q=s.q,v=s.v,time_s=s.time,clock_compensation=s.clock_compensation,step=s.step,
        displacement_hi=s.displacement_hi,displacement_lo=s.displacement_lo,
        material_roundoff_diagnostics=_roundoff_diagnostics(s),material=s.material.snapshot(),
        operator_verification_cache=s.snapshot_operator_cache(),
        records=records,rejections=rejected,dt_next=dt_next)
    canonical=json.dumps(plain(checkpoint),sort_keys=True,separators=(',',':'),allow_nan=False)
    write(destination/'checkpoint.json',dict(sha256=sha(canonical.encode()),state=checkpoint))
    write(destination/'records.json',records);write(destination/'rejections.json',rejected)
    work=sum(r['tool_work_J'] for r in records);tool_heat=sum(r['tool_heat_J'] for r in records)
    delta=records[-1]['total_energy_J']-s.initial_energy if records else 0.;windows=[]
    for a,b in ((0.,min(5e-9,s.time)),(0.,s.time)):
        selected=[r for r in records if a<r['time_s']<=b]
        if not selected:continue
        w=sum(r['tool_work_J'] for r in selected);e=sum(r['energy_defect_J'] for r in selected)
        scale=max(abs(w),abs(w+e),1e-12)
        windows.append(dict(start_s=a,end_s=b,energy_defect_J=e,scale_J=scale,relative=abs(e)/scale,
            passed=abs(e)<=max(case['acceptance']['energy_absolute_j'],case['acceptance']['energy_relative']*scale)))
    summary=dict(status=status,accepted=False,formulation='MPM_shared_domain_B2_four_persistent_GP_v1',
        step=s.step,rejected=len(rejected),time_s=s.time,target_time_s=t_end,
        tool_advance_m=s.time*case['geometry']['tool_speed'],developed_target_advance_m=.0005,
        wall_seconds=time.perf_counter()-started,failure=failure,discarded_trial=discarded_trial,
        source_sha256=identity['source_sha256'],sources_changed_during_run=changed,
        global_energy_defect_J=delta-work+tool_heat,tool_work_J=work,energy_windows=windows,
        final=records[-1] if records else None,
        scope='MPM candidate only; external numerical audit and complete cutting with a developed chip remain required. Spatial convergence is outside the current delivery scope.')
    write(destination/'summary.json',summary)
    print(json.dumps(plain({key:summary[key] for key in ('status','step','rejected','time_s','tool_advance_m','wall_seconds','failure')})),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination')
    p.add_argument('--case',default=str(ROOT/'cases/publication_v1/ck45_development_25mmin.json'))
    p.add_argument('--grid-h',type=float,default=50e-6);p.add_argument('--material-h',type=float)
    p.add_argument('--origin',type=float,nargs=2,default=[0.,0.]);p.add_argument('--backend',choices=BACKENDS,default='dense')
    p.add_argument('--dt',type=float,default=.5e-9);p.add_argument('--end',type=float,default=50e-9)
    p.add_argument('--wall',type=float,default=120.);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--geometry',choices=['cached','historical'],default='cached')
    p.add_argument('--integrator',choices=['explicit','implicit','hybrid'],default='explicit');p.add_argument('--mesh',choices=['uniform'],default='uniform')
    p.add_argument('--startup-time',type=float);p.add_argument('--startup-dt',type=float)
    p.add_argument('--constitutive-relative-tolerance',type=float,default=1e-13)
    p.add_argument('--resume');p.add_argument('--stop-after-steps',type=int)
    args=p.parse_args();run(args.destination,json.loads(Path(args.case).read_text()),
        grid_h=args.grid_h,material_h=args.material_h,origin=args.origin,backend=args.backend,
        dt_max=args.dt,t_end=args.end,wall_seconds=args.wall,threads=args.threads,
        geometry_backend=args.geometry,integrator=args.integrator,mesh_profile=args.mesh,
        constitutive_options={'relative_tolerance':args.constitutive_relative_tolerance},
        resume_from=args.resume,stop_after_steps=args.stop_after_steps,
        startup_time=args.startup_time,startup_dt=args.startup_dt)
