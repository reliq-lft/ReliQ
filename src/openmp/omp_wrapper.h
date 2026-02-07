/*
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/openmp/omp_wrapper.h
  Contact: reliq-lft@proton.me

  OpenMP C wrappers for Nim integration
  
  Provides CPU thread parallelization via #pragma omp parallel for
*/

#ifndef OMP_WRAPPER_H
#define OMP_WRAPPER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Callback function type for loop body execution (per-iteration) */
typedef void (*omp_loop_callback)(int64_t idx, void* context);

/* Callback function type for chunked loop body (range per thread) */
typedef void (*omp_chunk_callback)(int64_t start, int64_t end, void* context);

/* CPU parallel for loop with static scheduling (per-iteration callback) */
void omp_parallel_for(
    int64_t start,
    int64_t end,
    omp_loop_callback callback,
    void* context
);

/* CPU parallel for loop with chunked dispatch (range per thread)
 * Each thread receives a contiguous [chunk_start, chunk_end) range,
 * allowing the callback to use SIMD within its chunk. */
void omp_parallel_for_chunked(
    int64_t start,
    int64_t end,
    omp_chunk_callback callback,
    void* context
);

/* Utility functions */
int omp_get_num_threads_wrapper(void);
void omp_set_num_threads_wrapper(int num_threads);

#ifdef __cplusplus
}
#endif

#endif /* OMP_WRAPPER_H */
