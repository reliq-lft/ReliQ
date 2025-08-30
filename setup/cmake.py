"""
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: setup/cmake.py
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

def install(spack: pathlib.Path, version: str) -> pathlib.Path:
    spack_bin = spack / 'bin' / 'spack'
    cmake = 'cmake@' + version
    try: 
        tools.exec(spack_bin, 'find', cmake, **{'capture_output': True})
        print(cmake, 'already installed')
    except subprocess.CalledProcessError:
        specs = []
        tools.exec(spack_bin, '-e reliq add', cmake, *specs)
        tools.exec(spack_bin, '-e reliq install', cmake, *specs)