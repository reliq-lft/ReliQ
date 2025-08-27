"""
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: build/aux/tools.py
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

import argparse
import pathlib
import subprocess

import constants

def exec(
    *args: pathlib.Path | str, 
    **kwargs: dict[str, any]
) -> subprocess.CompletedProcess: 
    cmd = ' '.join([*map(str, args)])
    return subprocess.run(cmd, shell = True, check = True, **kwargs)

def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = constants.DESCRIPTION)
    parser.add_argument(
        '-p', 
        '--prefix', 
        help = 'ReliQ installation path',
        type = str,
        default = constants.CWD / 'deps'
    )
    flags = ['--spack', '--nim', '--upcxx', '--kokkos']
    helpers = [
        'Path to Spack',
        'Path to Nim',
        'Path to UPC++',
        'Path to Kokkos'
    ]
    defaults = [
        constants.DEFAULT_STANDIN_PATH,
        constants.DEFAULT_STANDIN_PATH,
        constants.DEFAULT_STANDIN_PATH,
        constants.DEFAULT_STANDIN_PATH,
    ]
    for (flag, help, default) in zip(flags, helpers, defaults):
        parser.add_argument(flag, help = help, type = str, default = default)
    return parser.parse_args()