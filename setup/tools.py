"""
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: setup/tools.py
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

import sys
import argparse
import pathlib
import subprocess

import constants

import cpu
import gpu

### constants ###

PYTHON: str = sys.executable

SPACK_URL: str = 'spack.io'
NIM_URL: str = 'nim-lang.org'
UPCXX_URL: str = 'docs.nersc.gov/development/programming-models/upcxx'
KOKKOS_URL: str = 'kokkos.org'

### tools ###

# executes Bash/Linux commands
def exec(
    *args: pathlib.Path | str, 
    **kwargs: dict[str, any]
) -> subprocess.CompletedProcess: 
    cmd = ' '.join([*map(str, args)])
    return subprocess.run(cmd, shell = True, check = True, **kwargs)

# executes lspci command
def lspci(*args: str, **kwargs: dict[str, any]) -> list[str]: 
    result = subprocess.check_output(
        ['lspci'] + [*args], 
        stderr = subprocess.DEVNULL, 
        text = True, 
        **kwargs
    )
    return result.strip().split("\n")

# grabs lspci info
def lspci_info(device: str) -> dict[str, any]:
    result: dict[str, any] = {}
    for info in lspci('-vvv', '-s', device.split()[0]):
        if info.strip() == '': continue
        try:
            key, value = [s.strip() for s in info.strip().split(':', 1)]
            result[key] = value
        except ValueError: continue
    return result

# gets command line arguments
def args(
    cpus: cpu.CentralProcessingUnits,
    gpus: gpu.GraphicsProcessingUnits
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = constants.DESCRIPTION,
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    
    # path to place "external" folder containing ReliQ's dependencies
    parser.add_argument(
        '-p', 
        '--prefix', 
        help = 'ReliQ installation path',
        type = str,
        default = constants.CWD / 'external'
    )

    # Spack package manager
    spack = parser.add_argument_group(
        'Spack (' + SPACK_URL + ')', 
        'High performance computing package manager; installs ReliQ\'s depenencies'
    )
    spack.add_argument(
        '--spack', 
        help = 'Path to Spack; if PREFIX, then installed by ReliQ', 
        type = str, 
        default = constants.DEFAULT_STANDIN_PATH
    )

    # Nim systems programming language
    nim = parser.add_argument_group(
        'Nim (' + NIM_URL + ')', 
        'Frontend systems programming language used to write ReliQ\'s source'
    )
    nim.add_argument(
        '--nim',
        help = 'Path to Nim; if PREFIX, then installed by Spack',
        type = str,
        default = constants.DEFAULT_STANDIN_PATH
    )

    # UPC++ partitioned global address space
    upcxx = parser.add_argument_group(
        'Unified Parallel C++ (' + UPCXX_URL + ')', 
        'Backend for ReliQ\'s partitioned global address space'
    )
    upcxx.add_argument(
        '--upcxx',
        help = 'Path to UPC++; if PREFIX, then installed by Spack',
        type = str,
        default = constants.DEFAULT_STANDIN_PATH
    )
    upcxx_flags = ['-gasnet', '-mpi']
    upcxx_helpers = [
        'Override embedded GASNet-EX with Spack\'s',
        'Enables MPI-based spawners and mpi-conduit',
    ]
    upcxx_choices = [
        [constants.TRUE, constants.FALSE],
        [constants.TRUE, constants.FALSE]
    ]
    upcxx_defaults = [
        constants.TRUE,
        constants.FALSE
    ]
    upcxx_options = [upcxx_flags, upcxx_helpers, upcxx_choices, upcxx_defaults]
    for (f, h, c, d) in zip(*upcxx_options):
        upcxx.add_argument(f, help = h, choices = c, default = d, type = str)

    # Kokkos shared memory parallelism
    kokkos = parser.add_argument_group(
        'Kokkos (' + KOKKOS_URL + ')', 
        'Backend for ReliQ\'s portable shared memory parallelism'
    )
    kokkos.add_argument(
        '--kokkos',
        help = 'Path to Kokkos; if PREFIX, then installed by Spack',
        type = str,
        default = constants.DEFAULT_STANDIN_PATH
    )
    kokkos_flags = [
        '-openmp',
        '-serial',
    ]
    kokkos_helpers = [
        'Build with OpenMP backend',
        'Build with serial backend'
    ]
    kokkos_choices = [
        [constants.TRUE, constants.FALSE],
        [constants.TRUE, constants.FALSE],
    ]
    kokkos_defaults = [
        constants.TRUE,
        constants.TRUE,
    ]
    kokkos_options = [kokkos_flags, kokkos_helpers, kokkos_choices, kokkos_defaults]
    for (f, h, c, d) in zip(*kokkos_options):
        kokkos.add_argument(f, help = h, choices = c, default = d, type = str)

    # UPC++ + Kokkos
    reliq = parser.add_argument_group(
        'Kokkos and Unified Parallel C++', 
        'Options that apply to both Kokkos and Unified Parallel C++ installation'
    )
    flags = [
        '-cuda',
        '-rocm',
    ]
    helpers = [
        'Build with CUDA',
        'Build with ROCm',
    ]
    choices = [
        [constants.TRUE, constants.FALSE],
        [constants.TRUE, constants.FALSE]
    ]
    defaults = [
        gpus.has_nvidia,
        gpus.has_amd
    ]
    for (f, h, c, d) in zip(flags, helpers, choices, defaults):
        reliq.add_argument(f, help = h, choices = c, default = d, type = str)

    # return parsed command line arguments
    return parser.parse_args()

# installs Python packages for execution
def install_python_packages() -> None:
    req = [PYTHON, '-m', 'pip', 'install', '-r', constants.SWD / 'requirements.txt']
    exec(*req, **{'capture_output': True})

# convert bool to argument string for bool
def bool_to_argbool(boolean: bool) -> str: return 'true' if boolean else 'false'

# convert argument string to Spack +/~
def argbool_to_spackbool(argbool: str) -> str:
    return '+' if argbool == 'true' else '~'