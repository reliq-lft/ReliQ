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
   Reduction: parallel for with OpenMP reduction(+:)
   ============================================================================ */

double omp_parallel_reduce_sum(
    int64_t start,
    int64_t end,
    omp_reduce_callback callback,
    void* context
) {
    double total = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:total)
    for (int64_t i = start; i < end; i++) {
        total += callback(i, context);
    }
    return total;
}

/* ============================================================================
   Chunked Reduction: each thread reduces a [start, end) chunk,
   OpenMP reduction(+:) sums partial sums across threads.
   This is the reduction analog of omp_parallel_for_chunked.
   ============================================================================ */

double omp_parallel_reduce_sum_chunked(
    int64_t start,
    int64_t end,
    omp_chunk_reduce_callback callback,
    void* context
) {
    double total = 0.0;
    int64_t range = end - start;
    if (range <= 0) return 0.0;

    #pragma omp parallel reduction(+:total)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int64_t chunk_size = (range + nthreads - 1) / nthreads;
        int64_t chunk_start = start + tid * chunk_size;
        int64_t chunk_end = chunk_start + chunk_size;
        if (chunk_end > end) chunk_end = end;
        if (chunk_start < end) {
            total += callback(chunk_start, chunk_end, context);
        }
    }
    return total;
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
