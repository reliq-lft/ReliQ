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
## Provides OpenMP initialization and helper functions for parallel CPU execution.

{.passC: "-fopenmp".}
{.passL: "-fopenmp".}

# OpenMP runtime function bindings
proc omp_get_max_threads*(): cint {.importc, header: "<omp.h>".}
proc omp_get_thread_num*(): cint {.importc, header: "<omp.h>".}

# Global initialization state
var ompInitialized {.global.}: bool = false

proc initOpenMP*() =
  ## Initialize OpenMP runtime
  ## Respects OMP_NUM_THREADS environment variable if set
  if not ompInitialized:
    let numThreads = omp_get_max_threads()
    echo "OpenMP: Initialized with ", numThreads, " threads"
    ompInitialized = true

proc getNumThreads*(): int {.inline.} =
  ## Get current number of OpenMP threads
  omp_get_max_threads().int

proc getThreadId*(): int {.inline.} =
  ## Get current thread ID (0 to numThreads-1)
  omp_get_thread_num().int

template ompParallel*(body: untyped) =
  ## Execute a block in parallel context
  ## Initializes OpenMP and executes the body
  initOpenMP()
  body

when isMainModule:
  initOpenMP()
  echo "Max threads: ", getNumThreads()
