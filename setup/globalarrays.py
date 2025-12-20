"""
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: setup/globalarrays.py
  Contact: reliq-lft@proton.me

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

import argparse
import pathlib
import subprocess

import constants
import tools

### install ###

def install(
    options: argparse.Namespace,
    spack: pathlib.Path, 
    path: pathlib.Path,
    version: str
) -> pathlib.Path:
    if path == constants.DEFAULT_STANDIN_PATH: 
        spack_bin = spack / 'bin' / 'spack'
        jobs = '-j ' + str(options.jobs)
        globalarrays = 'globalarrays@' + version
        try: 
            tools.exec(spack_bin, 'find', globalarrays, **{'capture_output': True})
            print(globalarrays, 'already installed')
        except subprocess.CalledProcessError:
            specs = ['^mpi']
            tools.exec(spack_bin, '-e reliq add', globalarrays, *specs)
            try: tools.exec(spack_bin, '-e reliq install', jobs, globalarrays, *specs)
            except subprocess.CalledProcessError as err:
                tools.exec(spack_bin, '-e reliq remove', globalarrays, *specs)
                raise err
            except KeyboardInterrupt as err:
                tools.exec(spack_bin, '-e reliq remove', globalarrays, *specs)
                raise err
    else: 
        # if path specified, check that we can find GlobalArrays libraries
        ga_lib = path / 'lib' / 'libga.a'  # or libga.so
        if not ga_lib.exists():
            ga_lib = path / 'lib' / 'libga.so'
            if not ga_lib.exists():
                raise OSError(str(ga_lib) + ': GlobalArrays library not found in ' + str(path))
    return path

### configure ###

def link(path: pathlib.Path, install_path: pathlib.Path) -> None: 
    if path != constants.DEFAULT_STANDIN_PATH:
        tools.exec('ln -sf', path / 'bin' / '/*', install_path / 'bin')
        tools.exec('ln -sf', path / 'include' / '/*', install_path / 'include')
        tools.exec('ln -sf', path / 'lib' / '/*', install_path / 'lib')