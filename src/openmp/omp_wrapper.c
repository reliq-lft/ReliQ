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
   Chunked parallel loop - enables SIMD within each thread's chunk
   ============================================================================ */

void omp_parallel_for_chunked(
    int64_t start,
    int64_t end,
    omp_chunk_callback callback,
    void* context
) {
    int64_t total = end - start;
    if (total <= 0) return;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int64_t chunk_size = (total + nthreads - 1) / nthreads;
        int64_t chunk_start = start + tid * chunk_size;
        int64_t chunk_end = chunk_start + chunk_size;
        if (chunk_end > end) chunk_end = end;
        if (chunk_start < end) {
            callback(chunk_start, chunk_end, context);
        }
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
