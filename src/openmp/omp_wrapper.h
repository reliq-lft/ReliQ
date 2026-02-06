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

/* Callback function type for loop body execution */
typedef void (*omp_loop_callback)(int64_t idx, void* context);

/* CPU parallel for loop with static scheduling */
void omp_parallel_for(
    int64_t start,
    int64_t end,
    omp_loop_callback callback,
    void* context
);

/* Utility functions */
int omp_get_num_threads_wrapper(void);
void omp_set_num_threads_wrapper(int num_threads);

#ifdef __cplusplus
}
#endif

#endif /* OMP_WRAPPER_H */
