"""
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: setup/spack.py
  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 reliq-lft
  
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.
  
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import pathlib
import subprocess

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