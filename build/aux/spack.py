import os
import pathlib
import subprocess
import typing

import constants
import tools

### constants ###

SPACK_URL: str = 'https://github.com/spack/spack.git'

### install ###

def install(install_path: pathlib.Path, path: pathlib.Path) -> pathlib.Path:
    os.chdir(install_path)
    if path == constants.DEFAULT_STANDIN_PATH:
        path = install_path / 'spack'
        if not path.is_dir():
            tools.exec('git', 'clone -c feature.manyFiles=true', SPACK_URL)
            os.chdir(path)
            tools.exec('.', 'share/spack/setup-env.sh')
    else: # if path specified, check that we can use spack and symlink it
        local_path = install_path / 'spack'
        spack_bin = local_path / 'bin' / 'spack'
        if not spack_bin.exists():
            raise OSError(str(spack_bin) + ': Does not exist in ' + str(path))
        if path.is_dir(): exec('ln -sf', path, local_path)
        else: raise OSError(str(path) + ': No such directory')
        try: tools.exec(spack_bin, '-h', **{'capture_output': True})
        except subprocess.CalledProcessError:
            raise OSError(str(spack_bin) + ': Permission denied')
        path = local_path
    return path

### configure ###

def create_env(path: pathlib.Path) -> None:
    env = path / 'var' / 'spack' / 'environments' / 'reliq'
    if not env.is_dir(): tools.exec(path / 'bin' / 'spack', 'env create reliq')

def link(path: pathlib.Path, install_path: pathlib.Path) -> None: 
    view = path / 'var' / 'spack' / 'environments' / 'reliq' / '.spack-env' / 'view'
    tools.exec('ln -sf', str(view) + '/*', install_path)