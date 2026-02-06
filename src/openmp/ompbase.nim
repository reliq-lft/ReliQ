#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/openmp/ompbase.nim
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
]#

## OpenMP Backend Base Utilities
##
## Provides OpenMP pragmas and helper functions for parallel CPU execution.
## This module uses Nim's foreign function interface to call OpenMP runtime
## functions and emit OpenMP pragmas into the generated C code.

{.passC: "-fopenmp".}
{.passL: "-fopenmp".}

# OpenMP runtime function bindings
proc omp_get_num_threads*(): cint {.importc, header: "<omp.h>".}
proc omp_get_thread_num*(): cint {.importc, header: "<omp.h>".}
proc omp_get_max_threads*(): cint {.importc, header: "<omp.h>".}
proc omp_set_num_threads*(n: cint) {.importc, header: "<omp.h>".}
proc omp_get_num_procs*(): cint {.importc, header: "<omp.h>".}
proc omp_in_parallel*(): cint {.importc, header: "<omp.h>".}
proc omp_get_wtime*(): cdouble {.importc, header: "<omp.h>".}

# Global initialization state
var ompInitialized {.global.}: bool = false

proc initOpenMP*() =
  ## Initialize OpenMP runtime
  ## Respects OMP_NUM_THREADS environment variable if set
  if not ompInitialized:
    # omp_get_max_threads() respects OMP_NUM_THREADS env var
    let numThreads = omp_get_max_threads()
    echo "OpenMP: Initialized with ", numThreads, " threads"
    ompInitialized = true

proc getNumThreads*(): int {.inline.} =
  ## Get current number of OpenMP threads
  omp_get_max_threads().int

proc getThreadId*(): int {.inline.} =
  ## Get current thread ID (0 to numThreads-1)
  omp_get_thread_num().int

# OpenMP parallel for pragma template
template ompParallelFor*(body: untyped) =
  ## Execute a loop body in parallel using OpenMP
  ## The loop variable and bounds should be set up before calling this
  {.emit: "#pragma omp parallel for".}
  body

template ompParallelForReduction*(op: string, varName: string, body: untyped) =
  ## Execute a parallel for with reduction
  {.emit: ["#pragma omp parallel for reduction(", op, ":", varName, ")"].}
  body

template ompParallel*(body: untyped) =
  ## Execute a block in parallel
  ## Note: For OpenMP, we just initialize OpenMP and execute the body.
  ## The actual parallelization happens in the `each` loop.
  ## The emit pragma for #pragma omp parallel would need to be inside a function.
  initOpenMP()
  body

template ompSingle*(body: untyped) =
  ## Execute by a single thread within a parallel region
  {.emit: "#pragma omp single".}
  body

template ompCritical*(body: untyped) =
  ## Critical section - only one thread at a time
  {.emit: "#pragma omp critical".}
  body

template ompBarrier*() =
  ## Synchronization barrier
  {.emit: "#pragma omp barrier".}

template ompAtomic*(body: untyped) =
  ## Atomic operation
  {.emit: "#pragma omp atomic".}
  body

# Dummy parallel template for backend compatibility
template ompBackendParallel*(body: untyped): untyped =
  ## Wrapper for compatibility with parallel.nim
  initOpenMP()
  body

when isMainModule:
  initOpenMP()
  echo "Max threads: ", getNumThreads()
  
  var sum: int = 0
  {.emit: "#pragma omp parallel for reduction(+:sum)".}
  for i in 0..<100:
    sum += i
  echo "Sum of 0..99 = ", sum
