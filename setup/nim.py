"""
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: setup/nim.py
  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 Curtis Taylor Peterson
  
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
            specs = []
            tools.exec(spack_bin, '-e reliq add', nim, *specs)
            tools.exec(spack_bin, '-e reliq install', nim, *specs)
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