/*
 * ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
 * Source file: src/openmp/simd_intrinsics.h
 * Contact: reliq-lft@proton.me
 *
 * Portable SIMD intrinsics abstraction layer.
 *
 * Supports SSE2, AVX/AVX2, and AVX-512 across four element types:
 *   double (64-bit float), float (32-bit float),
 *   int (32-bit integer), long long (64-bit integer).
 *
 * The VW (VectorWidth) parameter determines how many elements are processed
 * per "SIMD group".  When VW exceeds the native register width, multiple
 * registers are used automatically via unrolled loops.
 *
 * Before including this header, define:
 *   VW            -- the vector width (number of sites per group)
 *   One of: SIMD_USE_DOUBLE, SIMD_USE_FLOAT, SIMD_USE_INT32, SIMD_USE_INT64
 *
 * The selected type gets generic aliases:
 *   simd_v, simd_load, simd_store, simd_set1, simd_setzero,
 *   simd_add, simd_sub, simd_mul, simd_fmadd, simd_neg, simd_hadd,
 *   simd_gather, simd_scatter
 *
 * MIT License -- Copyright (c) 2025 reliq-lft
 */

#ifndef RELIQ_SIMD_INTRINSICS_H
#define RELIQ_SIMD_INTRINSICS_H

#include <immintrin.h>

#ifndef VW
  #error "VW (VectorWidth) must be defined before including simd_intrinsics.h"
#endif

/* ========================================================================
 * ISA detection -- element counts per native register
 * ======================================================================== */

#if defined(__AVX512F__)
  #define SIMD_REG_DOUBLES 8
#elif defined(__AVX__) || defined(__AVX2__)
  #define SIMD_REG_DOUBLES 4
#elif defined(__SSE2__)
  #define SIMD_REG_DOUBLES 2
#else
  #define SIMD_REG_DOUBLES 1
#endif

/* Number of native SIMD registers needed */
#if VW <= SIMD_REG_DOUBLES
  #define SIMD_NREGS_D 1
#else
  #define SIMD_NREGS_D (VW / SIMD_REG_DOUBLES)
#endif

/* ========================================================================
 * SECTION 1: DOUBLE (float64)
 * ======================================================================== */

#if SIMD_REG_DOUBLES == 8
  typedef __m512d simd_reg_d;
#elif SIMD_REG_DOUBLES == 4
  typedef __m256d simd_reg_d;
#elif SIMD_REG_DOUBLES == 2
  typedef __m128d simd_reg_d;
#else
  typedef double  simd_reg_d;
#endif

static inline simd_reg_d simd_reg_loadu_d(const double *p) {
#if SIMD_REG_DOUBLES == 8
  return _mm512_loadu_pd(p);
#elif SIMD_REG_DOUBLES == 4
  return _mm256_loadu_pd(p);
#elif SIMD_REG_DOUBLES == 2
  return _mm_loadu_pd(p);
#else
  return *p;
#endif
}
static inline void simd_reg_storeu_d(double *p, simd_reg_d v) {
#if SIMD_REG_DOUBLES == 8
  _mm512_storeu_pd(p, v);
#elif SIMD_REG_DOUBLES == 4
  _mm256_storeu_pd(p, v);
#elif SIMD_REG_DOUBLES == 2
  _mm_storeu_pd(p, v);
#else
  *p = v;
#endif
}
static inline simd_reg_d simd_reg_set1_d(double s) {
#if SIMD_REG_DOUBLES == 8
  return _mm512_set1_pd(s);
#elif SIMD_REG_DOUBLES == 4
  return _mm256_set1_pd(s);
#elif SIMD_REG_DOUBLES == 2
  return _mm_set1_pd(s);
#else
  return s;
#endif
}
static inline simd_reg_d simd_reg_setzero_d(void) {
#if SIMD_REG_DOUBLES == 8
  return _mm512_setzero_pd();
#elif SIMD_REG_DOUBLES == 4
  return _mm256_setzero_pd();
#elif SIMD_REG_DOUBLES == 2
  return _mm_setzero_pd();
#else
  return 0.0;
#endif
}
static inline simd_reg_d simd_reg_add_d(simd_reg_d a, simd_reg_d b) {
#if SIMD_REG_DOUBLES == 8
  return _mm512_add_pd(a, b);
#elif SIMD_REG_DOUBLES == 4
  return _mm256_add_pd(a, b);
#elif SIMD_REG_DOUBLES == 2
  return _mm_add_pd(a, b);
#else
  return a + b;
#endif
}
static inline simd_reg_d simd_reg_sub_d(simd_reg_d a, simd_reg_d b) {
#if SIMD_REG_DOUBLES == 8
  return _mm512_sub_pd(a, b);
#elif SIMD_REG_DOUBLES == 4
  return _mm256_sub_pd(a, b);
#elif SIMD_REG_DOUBLES == 2
  return _mm_sub_pd(a, b);
#else
  return a - b;
#endif
}
static inline simd_reg_d simd_reg_mul_d(simd_reg_d a, simd_reg_d b) {
#if SIMD_REG_DOUBLES == 8
  return _mm512_mul_pd(a, b);
#elif SIMD_REG_DOUBLES == 4
  return _mm256_mul_pd(a, b);
#elif SIMD_REG_DOUBLES == 2
  return _mm_mul_pd(a, b);
#else
  return a * b;
#endif
}
static inline simd_reg_d simd_reg_fmadd_d(simd_reg_d a, simd_reg_d b, simd_reg_d c) {
#if SIMD_REG_DOUBLES == 8
  return _mm512_fmadd_pd(a, b, c);
#elif SIMD_REG_DOUBLES == 4
  #ifdef __FMA__
    return _mm256_fmadd_pd(a, b, c);
  #else
    return _mm256_add_pd(_mm256_mul_pd(a, b), c);
  #endif
#elif SIMD_REG_DOUBLES == 2
  #ifdef __FMA__
    return _mm_fmadd_pd(a, b, c);
  #else
    return _mm_add_pd(_mm_mul_pd(a, b), c);
  #endif
#else
  return a * b + c;
#endif
}
static inline simd_reg_d simd_reg_fnmadd_d(simd_reg_d a, simd_reg_d b, simd_reg_d c) {
#if SIMD_REG_DOUBLES == 8
  return _mm512_fnmadd_pd(a, b, c);
#elif SIMD_REG_DOUBLES == 4
  #ifdef __FMA__
    return _mm256_fnmadd_pd(a, b, c);
  #else
    return _mm256_sub_pd(c, _mm256_mul_pd(a, b));
  #endif
#elif SIMD_REG_DOUBLES == 2
  #ifdef __FMA__
    return _mm_fnmadd_pd(a, b, c);
  #else
    return _mm_sub_pd(c, _mm_mul_pd(a, b));
  #endif
#else
  return c - a * b;
#endif
}
static inline simd_reg_d simd_reg_neg_d(simd_reg_d a) {
  return simd_reg_sub_d(simd_reg_setzero_d(), a);
}
static inline double simd_reg_reduce_d(simd_reg_d v) {
#if SIMD_REG_DOUBLES == 8
  return _mm512_reduce_add_pd(v);
#elif SIMD_REG_DOUBLES == 4
  __m128d lo = _mm256_castpd256_pd128(v);
  __m128d hi = _mm256_extractf128_pd(v, 1);
  __m128d s2 = _mm_add_pd(lo, hi);
  __m128d sh = _mm_shuffle_pd(s2, s2, 1);
  return _mm_cvtsd_f64(_mm_add_sd(s2, sh));
#elif SIMD_REG_DOUBLES == 2
  __m128d sh = _mm_shuffle_pd(v, v, 1);
  return _mm_cvtsd_f64(_mm_add_sd(v, sh));
#else
  return v;
#endif
}

/* Composite type */
typedef struct { simd_reg_d r[SIMD_NREGS_D]; } simd_vd;

static inline simd_vd simd_load_d(const double *p) {
  simd_vd v; for (int i = 0; i < SIMD_NREGS_D; i++) v.r[i] = simd_reg_loadu_d(p + i*SIMD_REG_DOUBLES); return v;
}
static inline void simd_store_d(double *p, simd_vd v) {
  for (int i = 0; i < SIMD_NREGS_D; i++) simd_reg_storeu_d(p + i*SIMD_REG_DOUBLES, v.r[i]);
}
static inline simd_vd simd_set1_d(double s) {
  simd_vd v; for (int i = 0; i < SIMD_NREGS_D; i++) v.r[i] = simd_reg_set1_d(s); return v;
}
static inline simd_vd simd_setzero_d(void) {
  simd_vd v; for (int i = 0; i < SIMD_NREGS_D; i++) v.r[i] = simd_reg_setzero_d(); return v;
}
static inline simd_vd simd_add_d(simd_vd a, simd_vd b) {
  simd_vd v; for (int i = 0; i < SIMD_NREGS_D; i++) v.r[i] = simd_reg_add_d(a.r[i], b.r[i]); return v;
}
static inline simd_vd simd_sub_d(simd_vd a, simd_vd b) {
  simd_vd v; for (int i = 0; i < SIMD_NREGS_D; i++) v.r[i] = simd_reg_sub_d(a.r[i], b.r[i]); return v;
}
static inline simd_vd simd_mul_d(simd_vd a, simd_vd b) {
  simd_vd v; for (int i = 0; i < SIMD_NREGS_D; i++) v.r[i] = simd_reg_mul_d(a.r[i], b.r[i]); return v;
}
static inline simd_vd simd_fmadd_d(simd_vd a, simd_vd b, simd_vd c) {
  simd_vd v; for (int i = 0; i < SIMD_NREGS_D; i++) v.r[i] = simd_reg_fmadd_d(a.r[i], b.r[i], c.r[i]); return v;
}
static inline simd_vd simd_fnmadd_d(simd_vd a, simd_vd b, simd_vd c) {
  simd_vd v; for (int i = 0; i < SIMD_NREGS_D; i++) v.r[i] = simd_reg_fnmadd_d(a.r[i], b.r[i], c.r[i]); return v;
}
static inline simd_vd simd_neg_d(simd_vd a) {
  simd_vd v; for (int i = 0; i < SIMD_NREGS_D; i++) v.r[i] = simd_reg_neg_d(a.r[i]); return v;
}
static inline double simd_hadd_d(simd_vd v) {
  double s = 0.0; for (int i = 0; i < SIMD_NREGS_D; i++) s += simd_reg_reduce_d(v.r[i]); return s;
}
static inline simd_vd simd_gather_d(const double *base, const int *indices, int elem, int elems) {
  double __attribute__((aligned(64))) buf[VW];
  for (int l = 0; l < VW; l++) { int g = indices[l]/VW, ln = indices[l]%VW; buf[l] = base[g*(VW*elems)+elem*VW+ln]; }
  return simd_load_d(buf);
}
static inline void simd_scatter_d(double *base, const int *indices, int elem, int elems, simd_vd val) {
  double __attribute__((aligned(64))) buf[VW];
  simd_store_d(buf, val);
  for (int l = 0; l < VW; l++) { int g = indices[l]/VW, ln = indices[l]%VW; base[g*(VW*elems)+elem*VW+ln] = buf[l]; }
}
/* Complex helpers (double) */
static inline void simd_cmul_d(simd_vd *rre, simd_vd *rim, simd_vd are, simd_vd aim, simd_vd bre, simd_vd bim) {
  *rre = simd_fnmadd_d(aim, bim, simd_mul_d(are, bre));
  *rim = simd_fmadd_d(aim, bre, simd_mul_d(are, bim));
}
static inline void simd_cmadd_d(simd_vd *sre, simd_vd *sim, simd_vd are, simd_vd aim, simd_vd bre, simd_vd bim) {
  *sre = simd_fmadd_d(are, bre, *sre); *sre = simd_fnmadd_d(aim, bim, *sre);
  *sim = simd_fmadd_d(are, bim, *sim); *sim = simd_fmadd_d(aim, bre, *sim);
}

/* ========================================================================
 * SECTION 2: FLOAT (float32)
 *
 * For VW=8: 8 floats = 32 bytes = one AVX register (256-bit) or 2 SSE regs.
 * On AVX-512 with VW=8, we use 256-bit __m256 (no waste).
 * ======================================================================== */

#if defined(__AVX__)
  typedef __m256  simd_reg_f;
  #define SIMD_REG_FLOATS_ACT 8
#elif defined(__SSE__)
  typedef __m128  simd_reg_f;
  #define SIMD_REG_FLOATS_ACT 4
#else
  typedef float   simd_reg_f;
  #define SIMD_REG_FLOATS_ACT 1
#endif

#if VW <= SIMD_REG_FLOATS_ACT
  #define SIMD_NREGS_F_ACT 1
#else
  #define SIMD_NREGS_F_ACT (VW / SIMD_REG_FLOATS_ACT)
#endif

static inline simd_reg_f simd_reg_loadu_f(const float *p) {
#if SIMD_REG_FLOATS_ACT == 8
  return _mm256_loadu_ps(p);
#elif SIMD_REG_FLOATS_ACT == 4
  return _mm_loadu_ps(p);
#else
  return *p;
#endif
}
static inline void simd_reg_storeu_f(float *p, simd_reg_f v) {
#if SIMD_REG_FLOATS_ACT == 8
  _mm256_storeu_ps(p, v);
#elif SIMD_REG_FLOATS_ACT == 4
  _mm_storeu_ps(p, v);
#else
  *p = v;
#endif
}
static inline simd_reg_f simd_reg_set1_f(float s) {
#if SIMD_REG_FLOATS_ACT == 8
  return _mm256_set1_ps(s);
#elif SIMD_REG_FLOATS_ACT == 4
  return _mm_set1_ps(s);
#else
  return s;
#endif
}
static inline simd_reg_f simd_reg_setzero_f(void) {
#if SIMD_REG_FLOATS_ACT == 8
  return _mm256_setzero_ps();
#elif SIMD_REG_FLOATS_ACT == 4
  return _mm_setzero_ps();
#else
  return 0.0f;
#endif
}
static inline simd_reg_f simd_reg_add_f(simd_reg_f a, simd_reg_f b) {
#if SIMD_REG_FLOATS_ACT == 8
  return _mm256_add_ps(a, b);
#elif SIMD_REG_FLOATS_ACT == 4
  return _mm_add_ps(a, b);
#else
  return a + b;
#endif
}
static inline simd_reg_f simd_reg_sub_f(simd_reg_f a, simd_reg_f b) {
#if SIMD_REG_FLOATS_ACT == 8
  return _mm256_sub_ps(a, b);
#elif SIMD_REG_FLOATS_ACT == 4
  return _mm_sub_ps(a, b);
#else
  return a - b;
#endif
}
static inline simd_reg_f simd_reg_mul_f(simd_reg_f a, simd_reg_f b) {
#if SIMD_REG_FLOATS_ACT == 8
  return _mm256_mul_ps(a, b);
#elif SIMD_REG_FLOATS_ACT == 4
  return _mm_mul_ps(a, b);
#else
  return a * b;
#endif
}
static inline simd_reg_f simd_reg_fmadd_f(simd_reg_f a, simd_reg_f b, simd_reg_f c) {
#if SIMD_REG_FLOATS_ACT == 8
  #ifdef __FMA__
    return _mm256_fmadd_ps(a, b, c);
  #else
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
  #endif
#elif SIMD_REG_FLOATS_ACT == 4
  #ifdef __FMA__
    return _mm_fmadd_ps(a, b, c);
  #else
    return _mm_add_ps(_mm_mul_ps(a, b), c);
  #endif
#else
  return a * b + c;
#endif
}
static inline simd_reg_f simd_reg_fnmadd_f(simd_reg_f a, simd_reg_f b, simd_reg_f c) {
#if SIMD_REG_FLOATS_ACT == 8
  #ifdef __FMA__
    return _mm256_fnmadd_ps(a, b, c);
  #else
    return _mm256_sub_ps(c, _mm256_mul_ps(a, b));
  #endif
#elif SIMD_REG_FLOATS_ACT == 4
  #ifdef __FMA__
    return _mm_fnmadd_ps(a, b, c);
  #else
    return _mm_sub_ps(c, _mm_mul_ps(a, b));
  #endif
#else
  return c - a * b;
#endif
}
static inline simd_reg_f simd_reg_neg_f(simd_reg_f a) {
  return simd_reg_sub_f(simd_reg_setzero_f(), a);
}
static inline float simd_reg_reduce_f(simd_reg_f v) {
#if SIMD_REG_FLOATS_ACT == 8
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s4 = _mm_add_ps(lo, hi);
  __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
  __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
  return _mm_cvtss_f32(s1);
#elif SIMD_REG_FLOATS_ACT == 4
  __m128 s2 = _mm_add_ps(v, _mm_movehl_ps(v, v));
  __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
  return _mm_cvtss_f32(s1);
#else
  return v;
#endif
}

typedef struct { simd_reg_f r[SIMD_NREGS_F_ACT]; } simd_vf;

static inline simd_vf simd_load_f(const float *p) {
  simd_vf v; for (int i = 0; i < SIMD_NREGS_F_ACT; i++) v.r[i] = simd_reg_loadu_f(p + i*SIMD_REG_FLOATS_ACT); return v;
}
static inline void simd_store_f(float *p, simd_vf v) {
  for (int i = 0; i < SIMD_NREGS_F_ACT; i++) simd_reg_storeu_f(p + i*SIMD_REG_FLOATS_ACT, v.r[i]);
}
static inline simd_vf simd_set1_f(float s) {
  simd_vf v; for (int i = 0; i < SIMD_NREGS_F_ACT; i++) v.r[i] = simd_reg_set1_f(s); return v;
}
static inline simd_vf simd_setzero_f(void) {
  simd_vf v; for (int i = 0; i < SIMD_NREGS_F_ACT; i++) v.r[i] = simd_reg_setzero_f(); return v;
}
static inline simd_vf simd_add_f(simd_vf a, simd_vf b) {
  simd_vf v; for (int i = 0; i < SIMD_NREGS_F_ACT; i++) v.r[i] = simd_reg_add_f(a.r[i], b.r[i]); return v;
}
static inline simd_vf simd_sub_f(simd_vf a, simd_vf b) {
  simd_vf v; for (int i = 0; i < SIMD_NREGS_F_ACT; i++) v.r[i] = simd_reg_sub_f(a.r[i], b.r[i]); return v;
}
static inline simd_vf simd_mul_f(simd_vf a, simd_vf b) {
  simd_vf v; for (int i = 0; i < SIMD_NREGS_F_ACT; i++) v.r[i] = simd_reg_mul_f(a.r[i], b.r[i]); return v;
}
static inline simd_vf simd_fmadd_f(simd_vf a, simd_vf b, simd_vf c) {
  simd_vf v; for (int i = 0; i < SIMD_NREGS_F_ACT; i++) v.r[i] = simd_reg_fmadd_f(a.r[i], b.r[i], c.r[i]); return v;
}
static inline simd_vf simd_fnmadd_f(simd_vf a, simd_vf b, simd_vf c) {
  simd_vf v; for (int i = 0; i < SIMD_NREGS_F_ACT; i++) v.r[i] = simd_reg_fnmadd_f(a.r[i], b.r[i], c.r[i]); return v;
}
static inline simd_vf simd_neg_f(simd_vf a) {
  simd_vf v; for (int i = 0; i < SIMD_NREGS_F_ACT; i++) v.r[i] = simd_reg_neg_f(a.r[i]); return v;
}
static inline float simd_hadd_f(simd_vf v) {
  float s = 0.0f; for (int i = 0; i < SIMD_NREGS_F_ACT; i++) s += simd_reg_reduce_f(v.r[i]); return s;
}
static inline simd_vf simd_gather_f(const float *base, const int *indices, int elem, int elems) {
  float __attribute__((aligned(64))) buf[VW];
  for (int l = 0; l < VW; l++) { int g = indices[l]/VW, ln = indices[l]%VW; buf[l] = base[g*(VW*elems)+elem*VW+ln]; }
  return simd_load_f(buf);
}
static inline void simd_scatter_f(float *base, const int *indices, int elem, int elems, simd_vf val) {
  float __attribute__((aligned(64))) buf[VW];
  simd_store_f(buf, val);
  for (int l = 0; l < VW; l++) { int g = indices[l]/VW, ln = indices[l]%VW; base[g*(VW*elems)+elem*VW+ln] = buf[l]; }
}
/* Complex helpers (float) */
static inline void simd_cmul_f(simd_vf *rre, simd_vf *rim, simd_vf are, simd_vf aim, simd_vf bre, simd_vf bim) {
  *rre = simd_fnmadd_f(aim, bim, simd_mul_f(are, bre));
  *rim = simd_fmadd_f(aim, bre, simd_mul_f(are, bim));
}
static inline void simd_cmadd_f(simd_vf *sre, simd_vf *sim, simd_vf are, simd_vf aim, simd_vf bre, simd_vf bim) {
  *sre = simd_fmadd_f(are, bre, *sre); *sre = simd_fnmadd_f(aim, bim, *sre);
  *sim = simd_fmadd_f(are, bim, *sim); *sim = simd_fmadd_f(aim, bre, *sim);
}

/* ========================================================================
 * SECTION 3: INT32
 *
 * AVX2 has full 256-bit integer support (_epi32). SSE2 has 128-bit.
 * Multiply: _mm256_mullo_epi32 (AVX2) / _mm_mullo_epi32 (SSE4.1).
 * ======================================================================== */

#if defined(__AVX2__)
  typedef __m256i simd_reg_i;
  #define SIMD_REG_I_ELEMS 8
#elif defined(__SSE2__)
  typedef __m128i simd_reg_i;
  #define SIMD_REG_I_ELEMS 4
#else
  typedef int     simd_reg_i;
  #define SIMD_REG_I_ELEMS 1
#endif

#if VW <= SIMD_REG_I_ELEMS
  #define SIMD_NREGS_I_ACT 1
#else
  #define SIMD_NREGS_I_ACT (VW / SIMD_REG_I_ELEMS)
#endif

static inline simd_reg_i simd_reg_loadu_i(const int *p) {
#if SIMD_REG_I_ELEMS == 8
  return _mm256_loadu_si256((const __m256i*)p);
#elif SIMD_REG_I_ELEMS == 4
  return _mm_loadu_si128((const __m128i*)p);
#else
  return *p;
#endif
}
static inline void simd_reg_storeu_i(int *p, simd_reg_i v) {
#if SIMD_REG_I_ELEMS == 8
  _mm256_storeu_si256((__m256i*)p, v);
#elif SIMD_REG_I_ELEMS == 4
  _mm_storeu_si128((__m128i*)p, v);
#else
  *p = v;
#endif
}
static inline simd_reg_i simd_reg_set1_i(int s) {
#if SIMD_REG_I_ELEMS == 8
  return _mm256_set1_epi32(s);
#elif SIMD_REG_I_ELEMS == 4
  return _mm_set1_epi32(s);
#else
  return s;
#endif
}
static inline simd_reg_i simd_reg_setzero_i(void) {
#if SIMD_REG_I_ELEMS == 8
  return _mm256_setzero_si256();
#elif SIMD_REG_I_ELEMS == 4
  return _mm_setzero_si128();
#else
  return 0;
#endif
}
static inline simd_reg_i simd_reg_add_i(simd_reg_i a, simd_reg_i b) {
#if SIMD_REG_I_ELEMS == 8
  return _mm256_add_epi32(a, b);
#elif SIMD_REG_I_ELEMS == 4
  return _mm_add_epi32(a, b);
#else
  return a + b;
#endif
}
static inline simd_reg_i simd_reg_sub_i(simd_reg_i a, simd_reg_i b) {
#if SIMD_REG_I_ELEMS == 8
  return _mm256_sub_epi32(a, b);
#elif SIMD_REG_I_ELEMS == 4
  return _mm_sub_epi32(a, b);
#else
  return a - b;
#endif
}
static inline simd_reg_i simd_reg_mul_i(simd_reg_i a, simd_reg_i b) {
#if SIMD_REG_I_ELEMS == 8
  return _mm256_mullo_epi32(a, b);
#elif SIMD_REG_I_ELEMS == 4
  #ifdef __SSE4_1__
    return _mm_mullo_epi32(a, b);
  #else
    /* SSE2 fallback for int32 multiply */
    __m128i t0 = _mm_mul_epu32(a, b);
    __m128i t1 = _mm_mul_epu32(_mm_srli_si128(a,4), _mm_srli_si128(b,4));
    return _mm_unpacklo_epi32(_mm_shuffle_epi32(t0, _MM_SHUFFLE(0,0,2,0)),
                              _mm_shuffle_epi32(t1, _MM_SHUFFLE(0,0,2,0)));
  #endif
#else
  return a * b;
#endif
}
static inline simd_reg_i simd_reg_fmadd_i(simd_reg_i a, simd_reg_i b, simd_reg_i c) {
  return simd_reg_add_i(simd_reg_mul_i(a, b), c);
}
static inline simd_reg_i simd_reg_fnmadd_i(simd_reg_i a, simd_reg_i b, simd_reg_i c) {
  return simd_reg_sub_i(c, simd_reg_mul_i(a, b));
}
static inline simd_reg_i simd_reg_neg_i(simd_reg_i a) {
  return simd_reg_sub_i(simd_reg_setzero_i(), a);
}
static inline int simd_reg_reduce_i(simd_reg_i v) {
#if SIMD_REG_I_ELEMS == 8
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i s4 = _mm_add_epi32(lo, hi);
  __m128i s2 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, _MM_SHUFFLE(1,0,3,2)));
  __m128i s1 = _mm_add_epi32(s2, _mm_shuffle_epi32(s2, _MM_SHUFFLE(0,1,0,1)));
  return _mm_cvtsi128_si32(s1);
#elif SIMD_REG_I_ELEMS == 4
  __m128i s2 = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1,0,3,2)));
  __m128i s1 = _mm_add_epi32(s2, _mm_shuffle_epi32(s2, _MM_SHUFFLE(0,1,0,1)));
  return _mm_cvtsi128_si32(s1);
#else
  return v;
#endif
}

typedef struct { simd_reg_i r[SIMD_NREGS_I_ACT]; } simd_vi;

static inline simd_vi simd_load_i(const int *p) {
  simd_vi v; for (int i = 0; i < SIMD_NREGS_I_ACT; i++) v.r[i] = simd_reg_loadu_i(p + i*SIMD_REG_I_ELEMS); return v;
}
static inline void simd_store_i(int *p, simd_vi v) {
  for (int i = 0; i < SIMD_NREGS_I_ACT; i++) simd_reg_storeu_i(p + i*SIMD_REG_I_ELEMS, v.r[i]);
}
static inline simd_vi simd_set1_i(int s) {
  simd_vi v; for (int i = 0; i < SIMD_NREGS_I_ACT; i++) v.r[i] = simd_reg_set1_i(s); return v;
}
static inline simd_vi simd_setzero_i(void) {
  simd_vi v; for (int i = 0; i < SIMD_NREGS_I_ACT; i++) v.r[i] = simd_reg_setzero_i(); return v;
}
static inline simd_vi simd_add_i(simd_vi a, simd_vi b) {
  simd_vi v; for (int i = 0; i < SIMD_NREGS_I_ACT; i++) v.r[i] = simd_reg_add_i(a.r[i], b.r[i]); return v;
}
static inline simd_vi simd_sub_i(simd_vi a, simd_vi b) {
  simd_vi v; for (int i = 0; i < SIMD_NREGS_I_ACT; i++) v.r[i] = simd_reg_sub_i(a.r[i], b.r[i]); return v;
}
static inline simd_vi simd_mul_i(simd_vi a, simd_vi b) {
  simd_vi v; for (int i = 0; i < SIMD_NREGS_I_ACT; i++) v.r[i] = simd_reg_mul_i(a.r[i], b.r[i]); return v;
}
static inline simd_vi simd_fmadd_i(simd_vi a, simd_vi b, simd_vi c) {
  simd_vi v; for (int i = 0; i < SIMD_NREGS_I_ACT; i++) v.r[i] = simd_reg_fmadd_i(a.r[i], b.r[i], c.r[i]); return v;
}
static inline simd_vi simd_fnmadd_i(simd_vi a, simd_vi b, simd_vi c) {
  simd_vi v; for (int i = 0; i < SIMD_NREGS_I_ACT; i++) v.r[i] = simd_reg_fnmadd_i(a.r[i], b.r[i], c.r[i]); return v;
}
static inline simd_vi simd_neg_i(simd_vi a) {
  simd_vi v; for (int i = 0; i < SIMD_NREGS_I_ACT; i++) v.r[i] = simd_reg_neg_i(a.r[i]); return v;
}
static inline int simd_hadd_i(simd_vi v) {
  int s = 0; for (int i = 0; i < SIMD_NREGS_I_ACT; i++) s += simd_reg_reduce_i(v.r[i]); return s;
}
static inline simd_vi simd_gather_i(const int *base, const int *indices, int elem, int elems) {
  int __attribute__((aligned(64))) buf[VW];
  for (int l = 0; l < VW; l++) { int g = indices[l]/VW, ln = indices[l]%VW; buf[l] = base[g*(VW*elems)+elem*VW+ln]; }
  return simd_load_i(buf);
}
static inline void simd_scatter_i(int *base, const int *indices, int elem, int elems, simd_vi val) {
  int __attribute__((aligned(64))) buf[VW];
  simd_store_i(buf, val);
  for (int l = 0; l < VW; l++) { int g = indices[l]/VW, ln = indices[l]%VW; base[g*(VW*elems)+elem*VW+ln] = buf[l]; }
}

/* ========================================================================
 * SECTION 4: INT64 (long long)
 *
 * Same register sizes as double (8 bytes per element).
 * Multiply: AVX-512DQ has _mm512_mullo_epi64. For AVX2/SSE, scalar fallback.
 * ======================================================================== */

#if defined(__AVX512F__)
  typedef __m512i simd_reg_l;
  #define SIMD_REG_L_ELEMS 8
#elif defined(__AVX2__)
  typedef __m256i simd_reg_l;
  #define SIMD_REG_L_ELEMS 4
#elif defined(__SSE2__)
  typedef __m128i simd_reg_l;
  #define SIMD_REG_L_ELEMS 2
#else
  typedef long long simd_reg_l;
  #define SIMD_REG_L_ELEMS 1
#endif

#if VW <= SIMD_REG_L_ELEMS
  #define SIMD_NREGS_L_ACT 1
#else
  #define SIMD_NREGS_L_ACT (VW / SIMD_REG_L_ELEMS)
#endif

static inline simd_reg_l simd_reg_loadu_l(const long long *p) {
#if SIMD_REG_L_ELEMS == 8
  return _mm512_loadu_si512(p);
#elif SIMD_REG_L_ELEMS == 4
  return _mm256_loadu_si256((const __m256i*)p);
#elif SIMD_REG_L_ELEMS == 2
  return _mm_loadu_si128((const __m128i*)p);
#else
  return *p;
#endif
}
static inline void simd_reg_storeu_l(long long *p, simd_reg_l v) {
#if SIMD_REG_L_ELEMS == 8
  _mm512_storeu_si512(p, v);
#elif SIMD_REG_L_ELEMS == 4
  _mm256_storeu_si256((__m256i*)p, v);
#elif SIMD_REG_L_ELEMS == 2
  _mm_storeu_si128((__m128i*)p, v);
#else
  *p = v;
#endif
}
static inline simd_reg_l simd_reg_set1_l(long long s) {
#if SIMD_REG_L_ELEMS == 8
  return _mm512_set1_epi64(s);
#elif SIMD_REG_L_ELEMS == 4
  return _mm256_set1_epi64x(s);
#elif SIMD_REG_L_ELEMS == 2
  return _mm_set1_epi64x(s);
#else
  return s;
#endif
}
static inline simd_reg_l simd_reg_setzero_l(void) {
#if SIMD_REG_L_ELEMS == 8
  return _mm512_setzero_si512();
#elif SIMD_REG_L_ELEMS == 4
  return _mm256_setzero_si256();
#elif SIMD_REG_L_ELEMS == 2
  return _mm_setzero_si128();
#else
  return 0LL;
#endif
}
static inline simd_reg_l simd_reg_add_l(simd_reg_l a, simd_reg_l b) {
#if SIMD_REG_L_ELEMS == 8
  return _mm512_add_epi64(a, b);
#elif SIMD_REG_L_ELEMS == 4
  return _mm256_add_epi64(a, b);
#elif SIMD_REG_L_ELEMS == 2
  return _mm_add_epi64(a, b);
#else
  return a + b;
#endif
}
static inline simd_reg_l simd_reg_sub_l(simd_reg_l a, simd_reg_l b) {
#if SIMD_REG_L_ELEMS == 8
  return _mm512_sub_epi64(a, b);
#elif SIMD_REG_L_ELEMS == 4
  return _mm256_sub_epi64(a, b);
#elif SIMD_REG_L_ELEMS == 2
  return _mm_sub_epi64(a, b);
#else
  return a - b;
#endif
}
static inline simd_reg_l simd_reg_mul_l(simd_reg_l a, simd_reg_l b) {
#if SIMD_REG_L_ELEMS == 8 && defined(__AVX512DQ__)
  return _mm512_mullo_epi64(a, b);
#elif SIMD_REG_L_ELEMS == 8
  long long __attribute__((aligned(64))) ba[8], bb[8], br[8];
  _mm512_storeu_si512(ba, a); _mm512_storeu_si512(bb, b);
  for (int i = 0; i < 8; i++) br[i] = ba[i] * bb[i];
  return _mm512_loadu_si512(br);
#elif SIMD_REG_L_ELEMS == 4
  long long __attribute__((aligned(32))) ba[4], bb[4], br[4];
  _mm256_storeu_si256((__m256i*)ba, a); _mm256_storeu_si256((__m256i*)bb, b);
  for (int i = 0; i < 4; i++) br[i] = ba[i] * bb[i];
  return _mm256_loadu_si256((const __m256i*)br);
#elif SIMD_REG_L_ELEMS == 2
  long long __attribute__((aligned(16))) ba[2], bb[2], br[2];
  _mm_storeu_si128((__m128i*)ba, a); _mm_storeu_si128((__m128i*)bb, b);
  for (int i = 0; i < 2; i++) br[i] = ba[i] * bb[i];
  return _mm_loadu_si128((const __m128i*)br);
#else
  return a * b;
#endif
}
static inline simd_reg_l simd_reg_fmadd_l(simd_reg_l a, simd_reg_l b, simd_reg_l c) {
  return simd_reg_add_l(simd_reg_mul_l(a, b), c);
}
static inline simd_reg_l simd_reg_fnmadd_l(simd_reg_l a, simd_reg_l b, simd_reg_l c) {
  return simd_reg_sub_l(c, simd_reg_mul_l(a, b));
}
static inline simd_reg_l simd_reg_neg_l(simd_reg_l a) {
  return simd_reg_sub_l(simd_reg_setzero_l(), a);
}
static inline long long simd_reg_reduce_l(simd_reg_l v) {
#if SIMD_REG_L_ELEMS == 8
  return _mm512_reduce_add_epi64(v);
#elif SIMD_REG_L_ELEMS == 4
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i s2 = _mm_add_epi64(lo, hi);
  __m128i s1 = _mm_add_epi64(s2, _mm_shuffle_epi32(s2, _MM_SHUFFLE(1,0,3,2)));
  return (long long)_mm_cvtsi128_si64(s1);
#elif SIMD_REG_L_ELEMS == 2
  __m128i s1 = _mm_add_epi64(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1,0,3,2)));
  return (long long)_mm_cvtsi128_si64(s1);
#else
  return v;
#endif
}

typedef struct { simd_reg_l r[SIMD_NREGS_L_ACT]; } simd_vl;

static inline simd_vl simd_load_l(const long long *p) {
  simd_vl v; for (int i = 0; i < SIMD_NREGS_L_ACT; i++) v.r[i] = simd_reg_loadu_l(p + i*SIMD_REG_L_ELEMS); return v;
}
static inline void simd_store_l(long long *p, simd_vl v) {
  for (int i = 0; i < SIMD_NREGS_L_ACT; i++) simd_reg_storeu_l(p + i*SIMD_REG_L_ELEMS, v.r[i]);
}
static inline simd_vl simd_set1_l(long long s) {
  simd_vl v; for (int i = 0; i < SIMD_NREGS_L_ACT; i++) v.r[i] = simd_reg_set1_l(s); return v;
}
static inline simd_vl simd_setzero_l(void) {
  simd_vl v; for (int i = 0; i < SIMD_NREGS_L_ACT; i++) v.r[i] = simd_reg_setzero_l(); return v;
}
static inline simd_vl simd_add_l(simd_vl a, simd_vl b) {
  simd_vl v; for (int i = 0; i < SIMD_NREGS_L_ACT; i++) v.r[i] = simd_reg_add_l(a.r[i], b.r[i]); return v;
}
static inline simd_vl simd_sub_l(simd_vl a, simd_vl b) {
  simd_vl v; for (int i = 0; i < SIMD_NREGS_L_ACT; i++) v.r[i] = simd_reg_sub_l(a.r[i], b.r[i]); return v;
}
static inline simd_vl simd_mul_l(simd_vl a, simd_vl b) {
  simd_vl v; for (int i = 0; i < SIMD_NREGS_L_ACT; i++) v.r[i] = simd_reg_mul_l(a.r[i], b.r[i]); return v;
}
static inline simd_vl simd_fmadd_l(simd_vl a, simd_vl b, simd_vl c) {
  simd_vl v; for (int i = 0; i < SIMD_NREGS_L_ACT; i++) v.r[i] = simd_reg_fmadd_l(a.r[i], b.r[i], c.r[i]); return v;
}
static inline simd_vl simd_fnmadd_l(simd_vl a, simd_vl b, simd_vl c) {
  simd_vl v; for (int i = 0; i < SIMD_NREGS_L_ACT; i++) v.r[i] = simd_reg_fnmadd_l(a.r[i], b.r[i], c.r[i]); return v;
}
static inline simd_vl simd_neg_l(simd_vl a) {
  simd_vl v; for (int i = 0; i < SIMD_NREGS_L_ACT; i++) v.r[i] = simd_reg_neg_l(a.r[i]); return v;
}
static inline long long simd_hadd_l(simd_vl v) {
  long long s = 0LL; for (int i = 0; i < SIMD_NREGS_L_ACT; i++) s += simd_reg_reduce_l(v.r[i]); return s;
}
static inline simd_vl simd_gather_l(const long long *base, const int *indices, int elem, int elems) {
  long long __attribute__((aligned(64))) buf[VW];
  for (int l = 0; l < VW; l++) { int g = indices[l]/VW, ln = indices[l]%VW; buf[l] = base[g*(VW*elems)+elem*VW+ln]; }
  return simd_load_l(buf);
}
static inline void simd_scatter_l(long long *base, const int *indices, int elem, int elems, simd_vl val) {
  long long __attribute__((aligned(64))) buf[VW];
  simd_store_l(buf, val);
  for (int l = 0; l < VW; l++) { int g = indices[l]/VW, ln = indices[l]%VW; base[g*(VW*elems)+elem*VW+ln] = buf[l]; }
}

#endif /* RELIQ_SIMD_INTRINSICS_H */

/* ========================================================================
 * SECTION 5: Generic aliases  (OUTSIDE the include guard)
 *
 * Because all kernel .c files are compiled in a single translation unit
 * (via -include), this section must be re-evaluated every time the header
 * is included.  We undef all previous aliases first.
 *
 * Selected by defining SIMD_USE_DOUBLE, SIMD_USE_FLOAT, SIMD_USE_INT32,
 * or SIMD_USE_INT64 before including this header.
 * ======================================================================== */

/* Undef previous aliases so they can be redefined for the current kernel */
#undef simd_v
#undef simd_load
#undef simd_store
#undef simd_set1
#undef simd_setzero
#undef simd_add
#undef simd_sub
#undef simd_mul
#undef simd_fmadd
#undef simd_fnmadd
#undef simd_neg
#undef simd_hadd
#undef simd_gather
#undef simd_scatter
#undef simd_cmul
#undef simd_cmadd

#if defined(SIMD_USE_DOUBLE)
  #define simd_v            simd_vd
  #define simd_load(p)      simd_load_d((const double*)(p))
  #define simd_store(p,v)   simd_store_d((double*)(p), v)
  #define simd_set1(s)      simd_set1_d((double)(s))
  #define simd_setzero()    simd_setzero_d()
  #define simd_add(a,b)     simd_add_d(a,b)
  #define simd_sub(a,b)     simd_sub_d(a,b)
  #define simd_mul(a,b)     simd_mul_d(a,b)
  #define simd_fmadd(a,b,c) simd_fmadd_d(a,b,c)
  #define simd_fnmadd(a,b,c) simd_fnmadd_d(a,b,c)
  #define simd_neg(a)       simd_neg_d(a)
  #define simd_hadd(v)      simd_hadd_d(v)
  #define simd_gather(b,i,e,n) simd_gather_d((const double*)(b),i,e,n)
  #define simd_scatter(b,i,e,n,v) simd_scatter_d((double*)(b),i,e,n,v)
  #define simd_cmul   simd_cmul_d
  #define simd_cmadd  simd_cmadd_d
#elif defined(SIMD_USE_FLOAT)
  #define simd_v            simd_vf
  #define simd_load(p)      simd_load_f((const float*)(p))
  #define simd_store(p,v)   simd_store_f((float*)(p), v)
  #define simd_set1(s)      simd_set1_f((float)(s))
  #define simd_setzero()    simd_setzero_f()
  #define simd_add(a,b)     simd_add_f(a,b)
  #define simd_sub(a,b)     simd_sub_f(a,b)
  #define simd_mul(a,b)     simd_mul_f(a,b)
  #define simd_fmadd(a,b,c) simd_fmadd_f(a,b,c)
  #define simd_fnmadd(a,b,c) simd_fnmadd_f(a,b,c)
  #define simd_neg(a)       simd_neg_f(a)
  #define simd_hadd(v)      simd_hadd_f(v)
  #define simd_gather(b,i,e,n) simd_gather_f((const float*)(b),i,e,n)
  #define simd_scatter(b,i,e,n,v) simd_scatter_f((float*)(b),i,e,n,v)
  #define simd_cmul   simd_cmul_f
  #define simd_cmadd  simd_cmadd_f
#elif defined(SIMD_USE_INT32)
  #define simd_v            simd_vi
  #define simd_load(p)      simd_load_i((const int*)(p))
  #define simd_store(p,v)   simd_store_i((int*)(p), v)
  #define simd_set1(s)      simd_set1_i((int)(s))
  #define simd_setzero()    simd_setzero_i()
  #define simd_add(a,b)     simd_add_i(a,b)
  #define simd_sub(a,b)     simd_sub_i(a,b)
  #define simd_mul(a,b)     simd_mul_i(a,b)
  #define simd_fmadd(a,b,c) simd_fmadd_i(a,b,c)
  #define simd_fnmadd(a,b,c) simd_fnmadd_i(a,b,c)
  #define simd_neg(a)       simd_neg_i(a)
  #define simd_hadd(v)      simd_hadd_i(v)
  #define simd_gather(b,i,e,n) simd_gather_i((const int*)(b),i,e,n)
  #define simd_scatter(b,i,e,n,v) simd_scatter_i((int*)(b),i,e,n,v)
#elif defined(SIMD_USE_INT64)
  #define simd_v            simd_vl
  #define simd_load(p)      simd_load_l((const long long*)(p))
  #define simd_store(p,v)   simd_store_l((long long*)(p), v)
  #define simd_set1(s)      simd_set1_l((long long)(s))
  #define simd_setzero()    simd_setzero_l()
  #define simd_add(a,b)     simd_add_l(a,b)
  #define simd_sub(a,b)     simd_sub_l(a,b)
  #define simd_mul(a,b)     simd_mul_l(a,b)
  #define simd_fmadd(a,b,c) simd_fmadd_l(a,b,c)
  #define simd_fnmadd(a,b,c) simd_fnmadd_l(a,b,c)
  #define simd_neg(a)       simd_neg_l(a)
  #define simd_hadd(v)      simd_hadd_l(v)
  #define simd_gather(b,i,e,n) simd_gather_l((const long long*)(b),i,e,n)
  #define simd_scatter(b,i,e,n,v) simd_scatter_l((long long*)(b),i,e,n,v)
#else
  /* Default to double if nothing specified */
  #define simd_v            simd_vd
  #define simd_load(p)      simd_load_d((const double*)(p))
  #define simd_store(p,v)   simd_store_d((double*)(p), v)
  #define simd_set1(s)      simd_set1_d((double)(s))
  #define simd_setzero()    simd_setzero_d()
  #define simd_add(a,b)     simd_add_d(a,b)
  #define simd_sub(a,b)     simd_sub_d(a,b)
  #define simd_mul(a,b)     simd_mul_d(a,b)
  #define simd_fmadd(a,b,c) simd_fmadd_d(a,b,c)
  #define simd_fnmadd(a,b,c) simd_fnmadd_d(a,b,c)
  #define simd_neg(a)       simd_neg_d(a)
  #define simd_hadd(v)      simd_hadd_d(v)
  #define simd_gather(b,i,e,n) simd_gather_d((const double*)(b),i,e,n)
  #define simd_scatter(b,i,e,n,v) simd_scatter_d((double*)(b),i,e,n,v)
  #define simd_cmul   simd_cmul_d
  #define simd_cmadd  simd_cmadd_d
#endif

/* Clean up the per-kernel type selector so the next kernel can set its own */
#undef SIMD_USE_DOUBLE
#undef SIMD_USE_FLOAT
#undef SIMD_USE_INT32
#undef SIMD_USE_INT64
