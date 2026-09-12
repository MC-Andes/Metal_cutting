"""Explicit preparation-only build. Never imported or called by frozen execution."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time

if __package__:
    from . import thermal_native as loader
else:
    spec=importlib.util.spec_from_file_location('_thermal_build_loader',Path(__file__).with_name('thermal_native.py'))
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)


def _write_atomic(path,value):
    path=Path(path)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix='.select-',delete=False) as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');temporary=Path(stream.name)
    os.replace(temporary,path)


def build_native():
    import pybind11
    root=loader.ROOT;compiler=shlex.split(os.environ.get('CXX','c++'))
    if not compiler:raise ValueError('CXX must name a compiler')
    executable=shutil.which(compiler[0])
    if executable is None:raise RuntimeError('Compiler unavailable during explicit preparation')
    flags=['-std=c++17','-O3','-shared','-fPIC','-fvisibility=hidden','-fno-fast-math','-ffp-contract=off']
    if sys.platform=='darwin':flags+=['-undefined','dynamic_lookup']
    include=Path(pybind11.get_include());python_include=Path(sysconfig.get_paths()['include'])
    pybind_headers={str(p.relative_to(include)):loader.sha(p) for p in sorted(include.rglob('*')) if p.is_file()}
    source_hashes={name:loader.sha(loader.relative_file(name)) for name in loader.SOURCES}
    compiler_info=dict(argv=compiler,version=subprocess.check_output(compiler+['--version'],text=True),
                       executable_sha256=loader.sha(Path(executable).resolve()))
    identity=dict(module=loader.MODULE,policy=loader.POLICY,runtime_abi=loader.runtime_abi(),
        sources=source_hashes,compiler=compiler_info,flags=flags,platform=platform.platform(),
        pybind_headers_sha256=loader.digest(pybind_headers),python_config_header_sha256=loader.sha(python_include/'pyconfig.h'))
    key=loader.digest(identity);parent=root/'.build'/loader.FAMILY;parent.mkdir(parents=True,exist_ok=True)
    destination=parent/key;manifest=destination/'manifest.json'
    binary_name=loader.MODULE+sysconfig.get_config_var('EXT_SUFFIX')
    if destination.exists():
        if not manifest.is_file():raise RuntimeError('Incomplete thermal artifact retained; do not overwrite it')
        data=json.loads(manifest.read_text())
        if data.get('identity')!=identity or data.get('identity_sha256')!=key or not (destination/binary_name).is_file() or loader.sha(destination/binary_name)!=data.get('binary_sha256'):
            raise RuntimeError('Existing thermal artifact changed; no silent rebuild')
    else:
        with tempfile.TemporaryDirectory(dir=parent,prefix='.building-') as temp:
            temp=Path(temp);binary=temp/binary_name
            command=compiler+flags+['-I'+str(include),'-I'+str(python_include),str(root/loader.SOURCES[-1]),'-o',str(binary)]
            started=time.perf_counter();result=subprocess.run(command,capture_output=True,text=True,timeout=120)
            if result.returncode:
                # Preserve a failed build as evidence; it is never selected.
                failed=parent/('failed-'+key);failed.mkdir(exist_ok=True)
                (failed/'build.log').write_text(result.stdout+result.stderr)
                raise RuntimeError('Thermal native compilation failed: '+str(failed/'build.log'))
            if any(loader.sha(root/name)!=value for name,value in source_hashes.items()):
                raise RuntimeError('Thermal source changed while compiling')
            data=dict(schema=loader.SCHEMA,identity=identity,identity_sha256=key,
                binary=str((Path('.build')/loader.FAMILY/key/binary_name)),binary_sha256=loader.sha(binary),
                command_template=compiler+flags+['-I{pybind11_include}','-I{python_include}',loader.SOURCES[-1],'-o','{artifact}/'+binary_name],
                pybind_header_hashes=pybind_headers,elapsed_seconds=time.perf_counter()-started,
                stdout=result.stdout,stderr=result.stderr,
                scope='Explicit preparation only. Load verifies archived artifact; no compile on frozen execution.')
            (temp/'manifest.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
            # Rename the complete artifact atomically. No partially built path is selected.
            os.rename(temp,destination)
    selected=dict(schema='thermal-fct-native-selection-v1',identity_sha256=key,
        manifest=str(manifest.relative_to(root)),manifest_sha256=loader.sha(manifest))
    _write_atomic(root/loader.SELECTOR,selected)
    return loader.verify_native()


if __name__=='__main__':
    print(json.dumps(build_native(),indent=2))
