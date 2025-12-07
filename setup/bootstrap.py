"""
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: setup/bootstrap
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

### imports ###

import pathlib
import json

import constants
import tools

#import cpu
#import gpu

import spack
import nim
import globalarrays
import kokkos

### execute installation script ###

if __name__ == '__main__':
    tools.install_python_packages()

    #cpus = cpu.CentralProcessingUnits()
    #gpus = gpu.GraphicsProcessingUnits()

    args = tools.args() #cpus, gpus)

    install_path = pathlib.Path(args.prefix)
    if not install_path.is_dir(): install_path.mkdir()

    version_path = constants.SWD / 'version.json'
    with version_path.open() as f: versions = json.load(f)

    #cmakev = versions['cmake']
    nimv = versions['nim']
    globalarraysv = versions['globalarrays']
    kokkosv = versions['kokkos']

    spack_path = spack.install(install_path, pathlib.Path(args.spack))
    spack.create_env(spack_path)

    #cmake.install(spack_path, cmakev)
    nim_path = nim.install(args, spack_path, pathlib.Path(args.nim), nimv)
    globalarrays_path = globalarrays.install(args, spack_path, pathlib.Path(args.globalarrays), globalarraysv)
    kokkos_path = kokkos.install(args, spack_path, pathlib.Path(args.kokkos), kokkosv)

    spack.link(spack_path, install_path)
    nim.link(nim_path, install_path)
    globalarrays.link(globalarrays_path, install_path)
    kokkos.link(kokkos_path, install_path)
