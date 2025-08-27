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
        kokkos = 'kokkos@' + version
        try: 
            tools.exec(spack_bin, 'find', kokkos, **{'capture_output': True})
            print(kokkos, 'already installed')
        except subprocess.CalledProcessError:
            tools.exec(spack_bin, ' -e reliq add', kokkos)
            tools.exec(spack_bin, ' -e reliq install', kokkos)
    return path

### configure ###

def link(path: pathlib.Path, install_path: pathlib.Path) -> None: 
    if path != constants.DEFAULT_STANDIN_PATH:
        tools.exec('ln -sf', path / 'bin' / '/*', install_path / 'bin')
        tools.exec('ln -sf', path / 'include' / '/*', install_path / 'include')