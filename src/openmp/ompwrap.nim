#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/openmp/ompwrap.nim
  Contact: reliq-lft@proton.me

  OpenMP C wrapper bindings for Nim
  
  This module provides Nim bindings to the OpenMP C wrappers in omp_wrapper.c.
  These allow proper OpenMP parallelization from Nim code.
]#

{.compile: "omp_wrapper.c".}
{.passC: "-fopenmp".}
{.passL: "-fopenmp".}

type
  OmpLoopCallback* = proc(idx: int64, context: pointer) {.cdecl.}

# ============================================================================
# Generic callback-based parallel loops
# ============================================================================

proc ompParallelFor*(start, `end`: int64, callback: OmpLoopCallback, 
                     context: pointer) {.importc: "omp_parallel_for", cdecl.}
  ## CPU parallel for loop with static scheduling

# ============================================================================
# Utility Functions
# ============================================================================

proc ompGetNumThreads*(): cint {.importc: "omp_get_num_threads_wrapper", cdecl.}
  ## Get number of available OpenMP threads

proc ompSetNumThreads*(numThreads: cint) {.importc: "omp_set_num_threads_wrapper", cdecl.}
  ## Set number of threads
