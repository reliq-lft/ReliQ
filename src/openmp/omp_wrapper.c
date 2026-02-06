/*
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/openmp/omp_wrapper.c
  Contact: reliq-lft@proton.me

  OpenMP C wrappers for Nim integration
  
  Provides CPU thread parallelization via #pragma omp parallel for
*/

#include "omp_wrapper.h"
#include <omp.h>

/* ============================================================================
   Generic callback-based parallel loop
   ============================================================================ */

void omp_parallel_for(
    int64_t start,
    int64_t end,
    omp_loop_callback callback,
    void* context
) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = start; i < end; i++) {
        callback(i, context);
    }
}

/* ============================================================================
   Utility Functions
   ============================================================================ */

int omp_get_num_threads_wrapper(void) {
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    return num_threads;
}

void omp_set_num_threads_wrapper(int num_threads) {
    omp_set_num_threads(num_threads);
}
