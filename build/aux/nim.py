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
        nim = 'nim@' + version
        try: 
            tools.exec(spack_bin, 'find', nim, **{'capture_output': True})
            print(nim, 'already installed')
        except subprocess.CalledProcessError:
            tools.exec(spack_bin, ' -e reliq add', nim)
            tools.exec(spack_bin, ' -e reliq install', nim)
    else: # if path specified, check that we can compile a Nim program
        nim_bin = path / 'bin' / 'nim'
        if not nim_bin.exists():
            raise OSError(str(nim_bin) + ': Does not exist in ' + str(path))
        try: tools.exec(nim_bin, '-h', **{'capture_output': True})
        except subprocess.CalledProcessError:
            raise OSError(str(nim_bin) + ': Permission denied')
    return path

### configure ###

def link(path: pathlib.Path, install_path: pathlib.Path) -> None: 
    if path != constants.DEFAULT_STANDIN_PATH:
        tools.exec('ln -sf', path / 'bin' / '/*', install_path / 'bin')
        tools.exec('ln -sf', path / 'include' / '/*', install_path / 'include')