"""Strict continuation with exactly the frozen executable and checkpoint.

This external entry point is archived separately. It does not alter the
solver, checkpoint, ROOT, discretization, or certificate. Only horizon and
wall budget may change, as allowed by the original checkpoint reader.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if not sys.flags.isolated:raise RuntimeError('Use Python -I')
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[key]='1'
    sys.dont_write_bytecode=True
    parser=argparse.ArgumentParser();parser.add_argument('runtime');parser.add_argument('checkpoint')
    parser.add_argument('--destination');parser.add_argument('--end',type=float)
    parser.add_argument('--wall',type=float,default=43200.)
    parser.add_argument('--check-output')
    args=parser.parse_args()
    entry_hash=sha(__file__);runtime=Path(args.runtime).resolve();checkpoint=Path(args.checkpoint).resolve()
    metadata=json.loads((runtime/'runtime.json').read_text())
    helper=Path(metadata['source_root'])/metadata['launcher_relative']
    # Establish the launcher identity before importing it; verified_spec then
    # validates the complete metadata, sources and supplemental dependency.
    if sha(helper)!=metadata['launcher_sha256']:raise ValueError('Frozen launcher changed')
    spec=importlib.util.spec_from_file_location('frozen_resume_guard',helper)
    guard=importlib.util.module_from_spec(spec);sys.modules[spec.name]=guard;spec.loader.exec_module(guard)
    folder,metadata,root=guard.verified_spec(runtime)
    if guard.runtime_identity()!=metadata['environment']:raise ValueError('Frozen numerical environment changed')
    sys.meta_path.insert(0,guard.FrozenImports(root,metadata['source_identity']['files']))
    sys.path.insert(0,str(root))
    native=guard.native_preflight(root,metadata['source_identity']['files'],metadata['backend'])
    from verification.cutting2d.cpdi_mpm_candidate import run,implicit
    from verification.cutting2d.direct_q4_candidate import geometry,constitutive
    from cutting2d import material
    material.load_kernel();constitutive.load_native()
    if 'verification/cutting2d/direct_q4_candidate/thermal_native.py' in metadata['source_identity']['files']:
        from verification.cutting2d.direct_q4_candidate import thermal_native
        thermal_native.load_native()
    if metadata['backend']!='dense':
        from verification.cutting2d.mpm_sparse_energy_candidate.qr import load_native
        load_native()
    if run.ROOT!=root or constitutive.ROOT!=root or material.ROOT!=root:
        raise ValueError('Solver import escaped frozen source root')
    raw=json.loads(checkpoint.read_text())['state'];inv=dict(raw['invocation'])
    inv['t_end_s']=args.end or inv['t_end_s'];inv['wall_seconds']=args.wall
    payload=run.read_checkpoint(checkpoint,raw['case'],inv)
    if payload['source_identity']!=metadata['source_identity']:
        raise ValueError('Checkpoint and selected runtime identities differ')
    if payload['case']!=json.loads(Path(metadata['case']['path']).read_text()):
        raise ValueError('Checkpoint and frozen physical case differ')
    files={str(p.relative_to(root)):sha(p) for p in run._source_files(metadata['backend'])}
    if files!=metadata['source_identity']['files']:raise ValueError('Loaded executable source set differs')
    report=dict(accepted=False,preflight_passed=True,checkpoint=str(checkpoint),
                checkpoint_sha256=sha(checkpoint),source_identity_exact=True,
                source_sha256=metadata['source_identity']['source_sha256'],
                environment_digest=guard.digest(metadata['environment']),native=native,
                entrypoint_sha256=entry_hash,invocation=inv,simulation_steps=0,
                checkpoint_conversion=False,operator_substitution=False,ROOT_override=False)
    if args.check_output:
        solver=implicit.ImplicitMPMCutting(payload['case'],grid_h=inv['grid_h_m'],
            material_h=inv['material_h_m'],origin=inv['origin_m'],backend=inv['backend'],
            threads=inv['threads'],geometry_backend=inv['geometry_backend'],
            constitutive_options=inv['constitutive_options'])
        run.restore(solver,payload)
        import numpy as np
        exact=all(np.array_equal(getattr(solver,key),payload[key]) for key in
                  ('q','v','displacement_hi','displacement_lo'))
        exact=exact and solver.material.snapshot()==payload['material']
        exact=exact and solver.snapshot_operator_cache()==payload['operator_verification_cache']
        if not exact:raise ValueError('Restored state or opaque certificate differs')
        report['restored_state_and_certificate_exact']=True
        output=Path(args.check_output);output.parent.mkdir(parents=True,exist_ok=True)
        if output.exists():raise FileExistsError('Do not overwrite a continuation check')
        output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return
    if not args.destination:raise ValueError('Supply a new --destination or --check-output')
    destination=Path(args.destination).resolve()
    if destination.exists() or destination.is_relative_to(runtime):raise ValueError('Destination must be new and outside runtime')
    try:
        run.run(destination,payload['case'],grid_h=inv['grid_h_m'],material_h=inv['material_h_m'],
            origin=inv['origin_m'],backend=inv['backend'],dt_max=inv['dt_max_s'],t_end=inv['t_end_s'],
            wall_seconds=args.wall,threads=inv['threads'],geometry_backend=inv['geometry_backend'],
            mesh_profile=inv['mesh_profile'],integrator=inv['integrator'],resume_from=checkpoint,
            constitutive_options=inv['constitutive_options'],startup_time=inv['startup_time_s'],
            startup_dt=inv['startup_dt_s'])
    finally:
        if destination.exists():
            report['entrypoint_unchanged']=sha(__file__)==entry_hash
            report['checkpoint_unchanged']=sha(checkpoint)==report['checkpoint_sha256']
            shutil.copyfile(__file__,destination/'resume_entrypoint.py')
            run.write(destination/'resume_preflight.json',report)


if __name__=='__main__':main()
