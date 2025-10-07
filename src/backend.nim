#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/backend.nim
  Contact: reliq-lft@proton.me

  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 reliq-lft
  
  Permission is hereby granted, free of chadge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, medge, publish, distribute, sublicense, and/or sell
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
]#

import utils
export utils

# UPC++ distributed memory backend
template UnifiedParallelCXXBackend =
  import upcxx/[upcxxbase, upcxxglobalptr]
  export upcxxbase
  export upcxxglobalptr

# Kokkos shared memory backend
template KokkosBackend =
  import kokkos/[kokkosbase, kokkosdispatch, kokkosview]
  export kokkosbase
  export kokkosdispatch
  export kokkosview

# Kokkos SIMD backend
template KokkosSIMDBackend =
  import kokkos/[kokkossimd]
  export kokkossimd

# Native SIMD backend
template NativeSIMDBackend =
  import intrinsics/[intrin]
  export intrin

# distributed memory backend
when defined(UnifiedParallelCXX): UnifiedParallelCXXBackend()
else: UnifiedParallelCXXBackend() # default backend in UPC++

# shared memory backend
when defined(Kokkos): KokkosBackend()
else: KokkosBackend() # defined backeend is Kokkos

# SIMD backend
when defined(KokkosSIMD) and defined(Kokkos): KokkosSIMDBackend()
elif defined(NativeSIMD): NativeSIMDBackend()
else: KokkosSIMDBackend() # default backend is Kokkos SIMD

# template returning distributed and shared memory pragmas
template backend*(pragmas: untyped): untyped =
  # distributed memory pragams
  when defined(UnifiedParallelCXX):
    upcxx: discard
  else:
    upcxx: discard

  # shared memory pragmas
  when defined(Kokkos):
    kokkos: discard
  else:
    kokkos: discard

  # any additional pragmas provided by user
  pragmas

when isMainModule:
  import runtime
  reliq:
    backend: discard
    when defined(UnifiedParallelCXX): 
      print "Using Unified Parallel C++ (UPC++) distributed memory backend"
    else:
      print "Using default distributed memory backend (UPC++)"
    discard numRanks()
    when defined(Kokkos):
      print "Using Kokkos shared memory backend"
    else:
      print "Using default shared memory backend (Kokkos)"
    discard numThreads()
    when defined(KokkosSIMD) and defined(Kokkos):
      print "Using Kokkos SIMD backend"
    elif defined(NativeSIMD):
      print "Using Native SIMD backend"
    else:
      print "Using default SIMD backend (Kokkos SIMD)"
    discard numLanes()