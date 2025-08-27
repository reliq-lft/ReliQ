import pathlib
import subprocess

import constants
import tools

### install ###

def install(
    spack: pathlib.Path, 
    path: pathlib.Path,
    version: str
) -> pathlib.Path:
    if path == constants.DEFAULT_STANDIN_PATH: 
        spack_bin = spack / 'bin' / 'spack'
        upcxx = 'upcxx@' + version
        try: 
            tools.exec(spack_bin, 'find', upcxx, **{'capture_output': True})
            print(upcxx, 'already installed')
        except subprocess.CalledProcessError:
            tools.exec(spack_bin, ' -e reliq add', upcxx)
            tools.exec(spack_bin, ' -e reliq install', upcxx)
    else: # if path specified, check that we can run a UPC++ program
        upcxx_run = path / 'bin' / 'upcxx-run'
        if not upcxx_run.exists():
            raise OSError(str(upcxx_run) + ': Does not exist in ' + str(path))
        try: tools.exec(upcxx_run, '-h', **{'capture_output': True})
        except subprocess.CalledProcessError:
            raise OSError(str(upcxx_run) + ': Permission denied')
    return path

### configure ###

def link(path: pathlib.Path, install_path: pathlib.Path) -> None: 
    if path != constants.DEFAULT_STANDIN_PATH:
        tools.exec('ln -sf', path / 'bin' / '/*', install_path / 'bin')
        tools.exec('ln -sf', path / 'include' / '/*', install_path / 'include')