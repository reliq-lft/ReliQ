"""
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: build/flags.py
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

import subprocess
import pathlib
import argparse
import json

DUMMY: str = 'CMakeFiles/hello_world.dir/hello_world.cpp.o -o hello_world'
ERR: str = 'Error: parameter passed to kokkos-meta must be one of CXXFLAGS LIBFLAGS'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmake_tmp')
    cmake_tmp: str = pathlib.Path(parser.parse_args().cmake_tmp)

    kokkos_defines: str = ''
    kokkos_includes: str = ''
    kokkos_flags: str = ''
    kokkos_link: str = ''

    kokkos_build_info = cmake_tmp / 'CMakeFiles' / 'hello_world.dir'
    with (kokkos_build_info / 'flags.make').open() as f: 
        for line in f.readlines():
            if 'CXX_DEFINES' in line: kokkos_defines = line.split('=', 1)[-1]
            elif 'CXX_INCLUDES' in line: kokkos_includes = line.split('=', 1)[-1]
            elif 'CXX_FLAGS' in line: kokkos_flags = line.split('=', 1)[-1]
    with (kokkos_build_info / 'link.txt').open() as f:
        kokkos_link = f.readlines()[0].split(DUMMY)[-1]
    
    pass_c: str = ' '.join(
        [s.strip() for s in ['-std=c++17', kokkos_defines, kokkos_includes, kokkos_flags]]
    )
    pass_l: str = kokkos_link.strip().replace('../external', 'external')

    with (cmake_tmp / '..' / 'external' / 'bin' / 'kokkos-meta').open('w+') as f:
        f.write(f"""#!/usr/bin/env python3
import sys
try:
    if sys.argv[1] == \'CXXFLAGS\': print(\'{pass_c}\')
    elif sys.argv[1] == \'LIBFLAGS\': print(\'{pass_l}\')
except IndexError: print(\'{ERR}\')\n""")