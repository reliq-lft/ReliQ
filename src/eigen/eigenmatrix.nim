#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/eigen/eigenmatrix.nim
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

import record/[record]
import utils/[complex]

{.emit: """/*TYPESECTION*/
#include <Eigen/Dense>

#define MatrixTemplateArgs Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor
#define MapTemplateArgs Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>

#define EigenStride(outer, inner) Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(outer, inner)

/* base matrix types */

using EigenMatrixRS = Eigen::Matrix<float, MatrixTemplateArgs>;
using EigenMatrixRD = Eigen::Matrix<double, MatrixTemplateArgs>;
using EigenMatrixCS = Eigen::Matrix<std::complex<float>, MatrixTemplateArgs>;
using EigenMatrixCD = Eigen::Matrix<std::complex<double>, MatrixTemplateArgs>;

using EigenMapMatrixRS = Eigen::Map<EigenMatrixRS, MapTemplateArgs>;
using EigenMapMatrixRD = Eigen::Map<EigenMatrixRD, MapTemplateArgs>;
using EigenMapMatrixCS = Eigen::Map<EigenMatrixCS, MapTemplateArgs>;
using EigenMapMatrixCD = Eigen::Map<EigenMatrixCD, MapTemplateArgs>;

struct EigenMatrixHandleRS { EigenMapMatrixRS* mat; };
struct EigenMatrixHandleRD { EigenMapMatrixRD* mat; };
struct EigenMatrixHandleCS { EigenMapMatrixCS* mat; };
struct EigenMatrixHandleCD { EigenMapMatrixCD* mat; };

/* base matrix constructors/destructors */

inline EigenMatrixHandleRS createEigenMatrixRS(
  float* data, int rows, int cols, int outer, int inner
) {
  auto* matrix = new EigenMapMatrixRS(data, rows, cols, EigenStride(outer, inner));
  return EigenMatrixHandleRS{matrix};
}

inline EigenMatrixHandleRD createEigenMatrixRD(
  double* data, int rows, int cols, int outer, int inner
) {
  auto* matrix = new EigenMapMatrixRD(data, rows, cols, EigenStride(outer, inner));
  return EigenMatrixHandleRD{matrix};
}

inline EigenMatrixHandleCS createEigenMatrixCS(
  void* data, int rows, int cols, int outer, int inner
) {
  auto* matrix = new EigenMapMatrixCS(static_cast<std::complex<float>*>(data), rows, cols, EigenStride(outer, inner));
  return EigenMatrixHandleCS{matrix};
}

inline EigenMatrixHandleCD createEigenMatrixCD(
  void* data, int rows, int cols, int outer, int inner
) {
  auto* matrix = new EigenMapMatrixCD(static_cast<std::complex<double>*>(data), rows, cols, EigenStride(outer, inner));
  return EigenMatrixHandleCD{matrix};
}

/* temporary matrix constructors — allocate owned data buffer */

inline EigenMatrixHandleRS createTempEigenMatrixRS(
  int rows, int cols, int outer, int inner
) {
  auto* data = new float[rows * cols]();
  auto* matrix = new EigenMapMatrixRS(data, rows, cols, EigenStride(outer, inner));
  return EigenMatrixHandleRS{matrix};
}

inline EigenMatrixHandleRD createTempEigenMatrixRD(
  int rows, int cols, int outer, int inner
) {
  auto* data = new double[rows * cols]();
  auto* matrix = new EigenMapMatrixRD(data, rows, cols, EigenStride(outer, inner));
  return EigenMatrixHandleRD{matrix};
}

inline EigenMatrixHandleCS createTempEigenMatrixCS(
  int rows, int cols, int outer, int inner
) {
  auto* data = new std::complex<float>[rows * cols]();
  auto* matrix = new EigenMapMatrixCS(data, rows, cols, EigenStride(outer, inner));
  return EigenMatrixHandleCS{matrix};
}

inline EigenMatrixHandleCD createTempEigenMatrixCD(
  int rows, int cols, int outer, int inner
) {
  auto* data = new std::complex<double>[rows * cols]();
  auto* matrix = new EigenMapMatrixCD(data, rows, cols, EigenStride(outer, inner));
  return EigenMatrixHandleCD{matrix};
}

/* destructors */

inline void destroyEigenMatrixRS(EigenMatrixHandleRS handle, bool ownsData) {
  if (ownsData) delete[] handle.mat->data();
  delete handle.mat;
}
inline void destroyEigenMatrixRD(EigenMatrixHandleRD handle, bool ownsData) {
  if (ownsData) delete[] handle.mat->data();
  delete handle.mat;
}
inline void destroyEigenMatrixCS(EigenMatrixHandleCS handle, bool ownsData) {
  if (ownsData) delete[] handle.mat->data();
  delete handle.mat;
}
inline void destroyEigenMatrixCD(EigenMatrixHandleCD handle, bool ownsData) {
  if (ownsData) delete[] handle.mat->data();
  delete handle.mat;
}

/* accessors */

inline float eigenMatrixRSGet(const EigenMatrixHandleRS handle, int row, int col) {
  return (*handle.mat)(row, col);
}

inline double eigenMatrixRDGet(const EigenMatrixHandleRD handle, int row, int col) {
  return (*handle.mat)(row, col);
}

inline void eigenMatrixCSGet(
  const EigenMatrixHandleCS handle, 
  int row, 
  int col,
  void* out
) { *static_cast<std::complex<float>*>(out) = (*handle.mat)(row, col); }

inline void eigenMatrixCDGet(
  const EigenMatrixHandleCD handle, 
  int row, 
  int col,
  void* out
) { *static_cast<std::complex<double>*>(out) = (*handle.mat)(row, col); }

inline void eigenMatrixRSSet(
  EigenMatrixHandleRS handle, 
  int row, 
  int col, 
  float value
) { (*handle.mat)(row, col) = value; }

inline void eigenMatrixRDSet(
  EigenMatrixHandleRD handle, 
  int row, 
  int col, 
  double value
) { (*handle.mat)(row, col) = value; }

inline void eigenMatrixCSSet(
  EigenMatrixHandleCS handle, 
  int row, 
  int col, 
  const void* value
) { (*handle.mat)(row, col) = *static_cast<const std::complex<float>*>(value); }

inline void eigenMatrixCDSet(
  EigenMatrixHandleCD handle, 
  int row, 
  int col, 
  const void* value
) { (*handle.mat)(row, col) = *static_cast<const std::complex<double>*>(value); }

/* algebra */

inline void eigenMatrixRSAdd(
  const EigenMatrixHandleRS a, 
  const EigenMatrixHandleRS b, 
  EigenMatrixHandleRS out
) { *out.mat = *a.mat + *b.mat; }

inline void eigenMatrixRDAdd(
  const EigenMatrixHandleRD a, 
  const EigenMatrixHandleRD b, 
  EigenMatrixHandleRD out
) { *out.mat = *a.mat + *b.mat; }

inline void eigenMatrixCSAdd(
  const EigenMatrixHandleCS a, 
  const EigenMatrixHandleCS b, 
  EigenMatrixHandleCS out
) { *out.mat = *a.mat + *b.mat; }

inline void eigenMatrixCDAdd(
  const EigenMatrixHandleCD a, 
  const EigenMatrixHandleCD b, 
  EigenMatrixHandleCD out
) { *out.mat = *a.mat + *b.mat; }

inline void eigenMatrixRSSub(
  const EigenMatrixHandleRS a, 
  const EigenMatrixHandleRS b, 
  EigenMatrixHandleRS out
) { *out.mat = *a.mat - *b.mat; }

inline void eigenMatrixRDSub(
  const EigenMatrixHandleRD a, 
  const EigenMatrixHandleRD b, 
  EigenMatrixHandleRD out
) { *out.mat = *a.mat - *b.mat; }

inline void eigenMatrixCSSub(
  const EigenMatrixHandleCS a, 
  const EigenMatrixHandleCS b, 
  EigenMatrixHandleCS out
) { *out.mat = *a.mat - *b.mat; }

inline void eigenMatrixCDSub(
  const EigenMatrixHandleCD a, 
  const EigenMatrixHandleCD b, 
  EigenMatrixHandleCD out
) { *out.mat = *a.mat - *b.mat; }

inline void eigenMatrixRSMul(
  const EigenMatrixHandleRS a, 
  const EigenMatrixHandleRS b, 
  EigenMatrixHandleRS out
) { *out.mat = (*a.mat) * (*b.mat); }

inline void eigenMatrixRDMul(
  const EigenMatrixHandleRD a, 
  const EigenMatrixHandleRD b, 
  EigenMatrixHandleRD out
) { *out.mat = (*a.mat) * (*b.mat); }

inline void eigenMatrixCSMul(
  const EigenMatrixHandleCS a, 
  const EigenMatrixHandleCS b, 
  EigenMatrixHandleCS out
) { *out.mat = (*a.mat) * (*b.mat); }

inline void eigenMatrixCDMul(
  const EigenMatrixHandleCD a, 
  const EigenMatrixHandleCD b, 
  EigenMatrixHandleCD out
) { *out.mat = (*a.mat) * (*b.mat); }

inline void eigenMatrixRSAddAssign(
  EigenMatrixHandleRS a, 
  const EigenMatrixHandleRS b
) { *a.mat += *b.mat; }

inline void eigenMatrixRDAddAssign(
  EigenMatrixHandleRD a, 
  const EigenMatrixHandleRD b
) { *a.mat += *b.mat; }

inline void eigenMatrixCSAddAssign(
  EigenMatrixHandleCS a, 
  const EigenMatrixHandleCS b
) { *a.mat += *b.mat; }

inline void eigenMatrixCDAddAssign(
  EigenMatrixHandleCD a, 
  const EigenMatrixHandleCD b
) { *a.mat += *b.mat; }

inline void eigenMatrixRSSubAssign(
  EigenMatrixHandleRS a, 
  const EigenMatrixHandleRS b
) { *a.mat -= *b.mat; }

inline void eigenMatrixRDSubAssign(
  EigenMatrixHandleRD a, 
  const EigenMatrixHandleRD b
) { *a.mat -= *b.mat; }

inline void eigenMatrixCSSubAssign(
  EigenMatrixHandleCS a, 
  const EigenMatrixHandleCS b
) { *a.mat -= *b.mat; }

inline void eigenMatrixCDSubAssign(
  EigenMatrixHandleCD a, 
  const EigenMatrixHandleCD b
) { *a.mat -= *b.mat; }

inline void eigenMatrixRSMulAssign(
  EigenMatrixHandleRS a, 
  const EigenMatrixHandleRS b
) { *a.mat *= *b.mat; }

inline void eigenMatrixRDMulAssign(
  EigenMatrixHandleRD a, 
  const EigenMatrixHandleRD b
) { *a.mat *= *b.mat; }

inline void eigenMatrixCSMulAssign(
  EigenMatrixHandleCS a, 
  const EigenMatrixHandleCS b
) { *a.mat *= *b.mat; }

inline void eigenMatrixCDMulAssign(
  EigenMatrixHandleCD a, 
  const EigenMatrixHandleCD b
) { *a.mat *= *b.mat; }

/* transpose — out = a^T */

inline void eigenMatrixRSTranspose(
  const EigenMatrixHandleRS a, EigenMatrixHandleRS out
) { *out.mat = a.mat->transpose(); }

inline void eigenMatrixRDTranspose(
  const EigenMatrixHandleRD a, EigenMatrixHandleRD out
) { *out.mat = a.mat->transpose(); }

inline void eigenMatrixCSTranspose(
  const EigenMatrixHandleCS a, EigenMatrixHandleCS out
) { *out.mat = a.mat->transpose(); }

inline void eigenMatrixCDTranspose(
  const EigenMatrixHandleCD a, EigenMatrixHandleCD out
) { *out.mat = a.mat->transpose(); }

/* adjoint (conjugate transpose) — out = a^H */

inline void eigenMatrixRSAdjoint(
  const EigenMatrixHandleRS a, EigenMatrixHandleRS out
) { *out.mat = a.mat->adjoint(); }

inline void eigenMatrixRDAdjoint(
  const EigenMatrixHandleRD a, EigenMatrixHandleRD out
) { *out.mat = a.mat->adjoint(); }

inline void eigenMatrixCSAdjoint(
  const EigenMatrixHandleCS a, EigenMatrixHandleCS out
) { *out.mat = a.mat->adjoint(); }

inline void eigenMatrixCDAdjoint(
  const EigenMatrixHandleCD a, EigenMatrixHandleCD out
) { *out.mat = a.mat->adjoint(); }

/* conjugate — out = conj(a) */

inline void eigenMatrixRSConjugate(
  const EigenMatrixHandleRS a, EigenMatrixHandleRS out
) { *out.mat = a.mat->conjugate(); }

inline void eigenMatrixRDConjugate(
  const EigenMatrixHandleRD a, EigenMatrixHandleRD out
) { *out.mat = a.mat->conjugate(); }

inline void eigenMatrixCSConjugate(
  const EigenMatrixHandleCS a, EigenMatrixHandleCS out
) { *out.mat = a.mat->conjugate(); }

inline void eigenMatrixCDConjugate(
  const EigenMatrixHandleCD a, EigenMatrixHandleCD out
) { *out.mat = a.mat->conjugate(); }

/* trace — returns sum of diagonal */

inline float eigenMatrixRSTrace(const EigenMatrixHandleRS handle) {
  return handle.mat->trace();
}

inline double eigenMatrixRDTrace(const EigenMatrixHandleRD handle) {
  return handle.mat->trace();
}

inline void eigenMatrixCSTrace(const EigenMatrixHandleCS handle, void* out) {
  *static_cast<std::complex<float>*>(out) = handle.mat->trace();
}

inline void eigenMatrixCDTrace(const EigenMatrixHandleCD handle, void* out) {
  *static_cast<std::complex<double>*>(out) = handle.mat->trace();
}

/* determinant */

inline float eigenMatrixRSDet(const EigenMatrixHandleRS handle) {
  return Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>(*handle.mat).determinant();
}

inline double eigenMatrixRDDet(const EigenMatrixHandleRD handle) {
  return Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(*handle.mat).determinant();
}

inline void eigenMatrixCSDet(const EigenMatrixHandleCS handle, void* out) {
  *static_cast<std::complex<float>*>(out) = Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic>(*handle.mat).determinant();
}

inline void eigenMatrixCDDet(const EigenMatrixHandleCD handle, void* out) {
  *static_cast<std::complex<double>*>(out) = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>(*handle.mat).determinant();
}

/* inverse */

inline void eigenMatrixRSInverse(
  const EigenMatrixHandleRS a, EigenMatrixHandleRS out
) { *out.mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>(*a.mat).inverse(); }

inline void eigenMatrixRDInverse(
  const EigenMatrixHandleRD a, EigenMatrixHandleRD out
) { *out.mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(*a.mat).inverse(); }

inline void eigenMatrixCSInverse(
  const EigenMatrixHandleCS a, EigenMatrixHandleCS out
) { *out.mat = Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic>(*a.mat).inverse(); }

inline void eigenMatrixCDInverse(
  const EigenMatrixHandleCD a, EigenMatrixHandleCD out
) { *out.mat = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>(*a.mat).inverse(); }

/* scalar multiply */

inline void eigenMatrixRSScalarMul(
  const EigenMatrixHandleRS a, float s, EigenMatrixHandleRS out
) { *out.mat = *a.mat * s; }

inline void eigenMatrixRDScalarMul(
  const EigenMatrixHandleRD a, double s, EigenMatrixHandleRD out
) { *out.mat = *a.mat * s; }

inline void eigenMatrixCSScalarMul(
  const EigenMatrixHandleCS a, const void* s, EigenMatrixHandleCS out
) { *out.mat = *a.mat * *static_cast<const std::complex<float>*>(s); }

inline void eigenMatrixCDScalarMul(
  const EigenMatrixHandleCD a, const void* s, EigenMatrixHandleCD out
) { *out.mat = *a.mat * *static_cast<const std::complex<double>*>(s); }

inline void eigenMatrixRSScalarMulAssign(EigenMatrixHandleRS a, float s) {
  *a.mat *= s;
}

inline void eigenMatrixRDScalarMulAssign(EigenMatrixHandleRD a, double s) {
  *a.mat *= s;
}

inline void eigenMatrixCSScalarMulAssign(EigenMatrixHandleCS a, const void* s) {
  *a.mat *= *static_cast<const std::complex<float>*>(s);
}

inline void eigenMatrixCDScalarMulAssign(EigenMatrixHandleCD a, const void* s) {
  *a.mat *= *static_cast<const std::complex<double>*>(s);
}

/* negate */

inline void eigenMatrixRSNegate(
  const EigenMatrixHandleRS a, EigenMatrixHandleRS out
) { *out.mat = -(*a.mat); }

inline void eigenMatrixRDNegate(
  const EigenMatrixHandleRD a, EigenMatrixHandleRD out
) { *out.mat = -(*a.mat); }

inline void eigenMatrixCSNegate(
  const EigenMatrixHandleCS a, EigenMatrixHandleCS out
) { *out.mat = -(*a.mat); }

inline void eigenMatrixCDNegate(
  const EigenMatrixHandleCD a, EigenMatrixHandleCD out
) { *out.mat = -(*a.mat); }

/* Frobenius norm */

inline float eigenMatrixRSNorm(const EigenMatrixHandleRS handle) {
  return handle.mat->norm();
}

inline double eigenMatrixRDNorm(const EigenMatrixHandleRD handle) {
  return handle.mat->norm();
}

inline float eigenMatrixCSNorm(const EigenMatrixHandleCS handle) {
  return handle.mat->norm();
}

inline double eigenMatrixCDNorm(const EigenMatrixHandleCD handle) {
  return handle.mat->norm();
}
""".}

type
  EigenMatrixHandleRS* {.importcpp: "EigenMatrixHandleRS".} = object
  EigenMatrixHandleRD* {.importcpp: "EigenMatrixHandleRD".} = object
  EigenMatrixHandleCS* {.importcpp: "EigenMatrixHandleCS".} = object
  EigenMatrixHandleCD* {.importcpp: "EigenMatrixHandleCD".} = object

record EigenMatrix*[R: static[int], T]:
  var rows: int
  var cols: int
  var inner: int
  var outer: int
  var ownsData: bool
  when isReal32(T):
    var data*: EigenMatrixHandleRS
  elif isReal64(T):
    var data*: EigenMatrixHandleRD
  elif isComplex32(T):
    var data*: EigenMatrixHandleCS
  elif isComplex64(T):
    var data*: EigenMatrixHandleCD

#[ eigen wrapper ]#

# constructors

proc createEigenMatrixRS(
  data: ptr float32; 
  rows, cols, outer, inner: int
): EigenMatrixHandleRS {.importcpp: "createEigenMatrixRS(@)".}

proc createEigenMatrixRD(
  data: ptr float64; 
  rows, cols, outer, inner: int
): EigenMatrixHandleRD {.importcpp: "createEigenMatrixRD(@)".}

proc createEigenMatrixCS(
  data: pointer; 
  rows, cols, outer, inner: int
): EigenMatrixHandleCS {.importcpp: "createEigenMatrixCS(@)".}

proc createEigenMatrixCD(
  data: pointer; 
  rows, cols, outer, inner: int
): EigenMatrixHandleCD {.importcpp: "createEigenMatrixCD(@)".}
  
# temp constructors

proc createTempEigenMatrixRS(
  rows, cols, outer, inner: int
): EigenMatrixHandleRS {.importcpp: "createTempEigenMatrixRS(@)".}

proc createTempEigenMatrixRD(
  rows, cols, outer, inner: int
): EigenMatrixHandleRD {.importcpp: "createTempEigenMatrixRD(@)".}

proc createTempEigenMatrixCS(
  rows, cols, outer, inner: int
): EigenMatrixHandleCS {.importcpp: "createTempEigenMatrixCS(@)".}

proc createTempEigenMatrixCD(
  rows, cols, outer, inner: int
): EigenMatrixHandleCD {.importcpp: "createTempEigenMatrixCD(@)".}

# destructors

proc destroyEigenMatrixRS(handle: EigenMatrixHandleRS, ownsData: bool) 
  {.importcpp: "destroyEigenMatrixRS(@)".}

proc destroyEigenMatrixRD(handle: EigenMatrixHandleRD, ownsData: bool) 
  {.importcpp: "destroyEigenMatrixRD(@)".}

proc destroyEigenMatrixCS(handle: EigenMatrixHandleCS, ownsData: bool) 
  {.importcpp: "destroyEigenMatrixCS(@)".}

proc destroyEigenMatrixCD(handle: EigenMatrixHandleCD, ownsData: bool) 
  {.importcpp: "destroyEigenMatrixCD(@)".}

# accessors

proc eigenMatrixRSGet(handle: EigenMatrixHandleRS, row, col: int): float32 
  {.importcpp: "eigenMatrixRSGet(@)".}

proc eigenMatrixRDGet(handle: EigenMatrixHandleRD, row, col: int): float64 
  {.importcpp: "eigenMatrixRDGet(@)".}

proc eigenMatrixCSGet(handle: EigenMatrixHandleCS, row, col: int, outVal: pointer) 
  {.importcpp: "eigenMatrixCSGet(@)".}

proc eigenMatrixCDGet(handle: EigenMatrixHandleCD, row, col: int, outVal: pointer) 
  {.importcpp: "eigenMatrixCDGet(@)".}

proc eigenMatrixRSSet(handle: EigenMatrixHandleRS, row, col: int, value: float32) 
  {.importcpp: "eigenMatrixRSSet(@)".}

proc eigenMatrixRDSet(handle: EigenMatrixHandleRD, row, col: int, value: float64) 
  {.importcpp: "eigenMatrixRDSet(@)".}

proc eigenMatrixCSSet(handle: EigenMatrixHandleCS, row, col: int, value: pointer) 
  {.importcpp: "eigenMatrixCSSet(@)".}

proc eigenMatrixCDSet(handle: EigenMatrixHandleCD, row, col: int, value: pointer) 
  {.importcpp: "eigenMatrixCDSet(@)".}

# algebra

proc eigenMatrixRSAdd(a, b: EigenMatrixHandleRS; c: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSAdd(@)".}

proc eigenMatrixRDAdd(a, b: EigenMatrixHandleRD; c: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDAdd(@)".}

proc eigenMatrixCSAdd(a, b: EigenMatrixHandleCS; c: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSAdd(@)".}

proc eigenMatrixCDAdd(a, b: EigenMatrixHandleCD; c: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDAdd(@)".}

proc eigenMatrixRSSub(a, b: EigenMatrixHandleRS; c: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSSub(@)".}

proc eigenMatrixRDSub(a, b: EigenMatrixHandleRD; c: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDSub(@)".}

proc eigenMatrixCSSub(a, b: EigenMatrixHandleCS; c: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSSub(@)".}

proc eigenMatrixCDSub(a, b: EigenMatrixHandleCD; c: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDSub(@)".}

proc eigenMatrixRSMul(a, b: EigenMatrixHandleRS; c: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSMul(@)".}

proc eigenMatrixRDMul(a, b: EigenMatrixHandleRD; c: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDMul(@)".}

proc eigenMatrixCSMul(a, b: EigenMatrixHandleCS; c: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSMul(@)".}

proc eigenMatrixCDMul(a, b: EigenMatrixHandleCD; c: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDMul(@)".}

proc eigenMatrixRSAddAssign(a, b: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSAddAssign(@)".}

proc eigenMatrixRDAddAssign(a, b: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDAddAssign(@)".}

proc eigenMatrixCSAddAssign(a, b: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSAddAssign(@)".}

proc eigenMatrixCDAddAssign(a, b: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDAddAssign(@)".}

proc eigenMatrixRSSubAssign(a, b: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSSubAssign(@)".}

proc eigenMatrixRDSubAssign(a, b: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDSubAssign(@)".}

proc eigenMatrixCSSubAssign(a, b: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSSubAssign(@)".}

proc eigenMatrixCDSubAssign(a, b: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDSubAssign(@)".}

proc eigenMatrixRSMulAssign(a, b: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSMulAssign(@)".}

proc eigenMatrixRDMulAssign(a, b: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDMulAssign(@)".}

proc eigenMatrixCSMulAssign(a, b: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSMulAssign(@)".}

proc eigenMatrixCDMulAssign(a, b: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDMulAssign(@)".}

# transpose

proc eigenMatrixRSTranspose(a: EigenMatrixHandleRS, c: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSTranspose(@)".}

proc eigenMatrixRDTranspose(a: EigenMatrixHandleRD, c: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDTranspose(@)".}

proc eigenMatrixCSTranspose(a: EigenMatrixHandleCS, c: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSTranspose(@)".}

proc eigenMatrixCDTranspose(a: EigenMatrixHandleCD, c: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDTranspose(@)".}

# adjoint

proc eigenMatrixRSAdjoint(a: EigenMatrixHandleRS, c: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSAdjoint(@)".}

proc eigenMatrixRDAdjoint(a: EigenMatrixHandleRD, c: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDAdjoint(@)".}

proc eigenMatrixCSAdjoint(a: EigenMatrixHandleCS, c: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSAdjoint(@)".}

proc eigenMatrixCDAdjoint(a: EigenMatrixHandleCD, c: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDAdjoint(@)".}

# conjugate

proc eigenMatrixRSConjugate(a: EigenMatrixHandleRS, c: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSConjugate(@)".}

proc eigenMatrixRDConjugate(a: EigenMatrixHandleRD, c: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDConjugate(@)".}

proc eigenMatrixCSConjugate(a: EigenMatrixHandleCS, c: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSConjugate(@)".}

proc eigenMatrixCDConjugate(a: EigenMatrixHandleCD, c: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDConjugate(@)".}

# trace

proc eigenMatrixRSTrace(handle: EigenMatrixHandleRS): float32 
  {.importcpp: "eigenMatrixRSTrace(@)".}

proc eigenMatrixRDTrace(handle: EigenMatrixHandleRD): float64 
  {.importcpp: "eigenMatrixRDTrace(@)".}

proc eigenMatrixCSTrace(handle: EigenMatrixHandleCS, outVal: pointer) 
  {.importcpp: "eigenMatrixCSTrace(@)".}

proc eigenMatrixCDTrace(handle: EigenMatrixHandleCD, outVal: pointer) 
  {.importcpp: "eigenMatrixCDTrace(@)".}

# determinant

proc eigenMatrixRSDet(handle: EigenMatrixHandleRS): float32 
  {.importcpp: "eigenMatrixRSDet(@)".}

proc eigenMatrixRDDet(handle: EigenMatrixHandleRD): float64 
  {.importcpp: "eigenMatrixRDDet(@)".}

proc eigenMatrixCSDet(handle: EigenMatrixHandleCS, outVal: pointer) 
  {.importcpp: "eigenMatrixCSDet(@)".}

proc eigenMatrixCDDet(handle: EigenMatrixHandleCD, outVal: pointer) 
  {.importcpp: "eigenMatrixCDDet(@)".}

# inverse

proc eigenMatrixRSInverse(a: EigenMatrixHandleRS, c: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSInverse(@)".}

proc eigenMatrixRDInverse(a: EigenMatrixHandleRD, c: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDInverse(@)".}

proc eigenMatrixCSInverse(a: EigenMatrixHandleCS, c: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSInverse(@)".}

proc eigenMatrixCDInverse(a: EigenMatrixHandleCD, c: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDInverse(@)".}

# scalar multiply

proc eigenMatrixRSScalarMul(a: EigenMatrixHandleRS, s: float32, c: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSScalarMul(@)".}

proc eigenMatrixRDScalarMul(a: EigenMatrixHandleRD, s: float64, c: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDScalarMul(@)".}

proc eigenMatrixCSScalarMul(a: EigenMatrixHandleCS, s: pointer, c: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSScalarMul(@)".}

proc eigenMatrixCDScalarMul(a: EigenMatrixHandleCD, s: pointer, c: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDScalarMul(@)".}

proc eigenMatrixRSScalarMulAssign(a: EigenMatrixHandleRS, s: float32) 
  {.importcpp: "eigenMatrixRSScalarMulAssign(@)".}

proc eigenMatrixRDScalarMulAssign(a: EigenMatrixHandleRD, s: float64) 
  {.importcpp: "eigenMatrixRDScalarMulAssign(@)".}

proc eigenMatrixCSScalarMulAssign(a: EigenMatrixHandleCS, s: pointer) 
  {.importcpp: "eigenMatrixCSScalarMulAssign(@)".}

proc eigenMatrixCDScalarMulAssign(a: EigenMatrixHandleCD, s: pointer) 
  {.importcpp: "eigenMatrixCDScalarMulAssign(@)".}

# negate

proc eigenMatrixRSNegate(a: EigenMatrixHandleRS, c: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSNegate(@)".}

proc eigenMatrixRDNegate(a: EigenMatrixHandleRD, c: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDNegate(@)".}

proc eigenMatrixCSNegate(a: EigenMatrixHandleCS, c: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSNegate(@)".}

proc eigenMatrixCDNegate(a: EigenMatrixHandleCD, c: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDNegate(@)".}

# norm (Frobenius)

proc eigenMatrixRSNorm(handle: EigenMatrixHandleRS): float32 
  {.importcpp: "eigenMatrixRSNorm(@)".}

proc eigenMatrixRDNorm(handle: EigenMatrixHandleRD): float64 
  {.importcpp: "eigenMatrixRDNorm(@)".}

proc eigenMatrixCSNorm(handle: EigenMatrixHandleCS): float32 
  {.importcpp: "eigenMatrixCSNorm(@)".}

proc eigenMatrixCDNorm(handle: EigenMatrixHandleCD): float64 
  {.importcpp: "eigenMatrixCDNorm(@)".}

#[ EigenMatrix implementation ]#

impl EigenMatrix:
  #[ constructor/destructor ]#

  method init(rawData: ptr T; shape: array[R, int]; inner, outer: int) =
    assert R == 2, "only rank-2 supported"
    this.rows = shape[0]
    this.cols = shape[1]
    this.inner = inner
    this.outer = outer
    this.ownsData = false

    when isReal32(T):
      this.data = createEigenMatrixRS(rawData, this.rows, this.cols, outer, inner)
    elif isReal64(T):
      this.data = createEigenMatrixRD(rawData, this.rows, this.cols, outer, inner)
    elif isComplex32(T):
      this.data = createEigenMatrixCS(cast[pointer](rawData), this.rows, this.cols, outer, inner)
    elif isComplex64(T):
      this.data = createEigenMatrixCD(cast[pointer](rawData), this.rows, this.cols, outer, inner)

  method init(numRows, numCols, inner, outer: int) =
    assert R == 2, "only rank-2 supported"
    this.rows = numRows
    this.cols = numCols
    this.inner = inner
    this.outer = outer
    this.ownsData = true

    when isReal32(T):
      this.data = createTempEigenMatrixRS(numRows, numCols, outer, inner)
    elif isReal64(T):
      this.data = createTempEigenMatrixRD(numRows, numCols, outer, inner)
    elif isComplex32(T):
      this.data = createTempEigenMatrixCS(numRows, numCols, outer, inner)
    elif isComplex64(T):
      this.data = createTempEigenMatrixCD(numRows, numCols, outer, inner)
  
  method deinit() =
    when isReal32(T): destroyEigenMatrixRS(this.data, this.ownsData)
    elif isReal64(T): destroyEigenMatrixRD(this.data, this.ownsData)
    elif isComplex32(T): destroyEigenMatrixCS(this.data, this.ownsData)
    elif isComplex64(T): destroyEigenMatrixCD(this.data, this.ownsData)
  
  #[ accessors ]#

  method `[]`(row, col: int): T =
    when isReal32(T):
      return eigenMatrixRSGet(this.data, row, col)
    elif isReal64(T):
      return eigenMatrixRDGet(this.data, row, col)
    elif isComplex32(T):
      var res: Complex32
      eigenMatrixCSGet(this.data, row, col, addr res)
      return res
    elif isComplex64(T):
      var res: Complex64
      eigenMatrixCDGet(this.data, row, col, addr res)
      return res
  
  method `[]=`(row, col: int, value: T) =
    when isReal32(T):
      eigenMatrixRSSet(this.data, row, col, value)
    elif isReal64(T):
      eigenMatrixRDSet(this.data, row, col, value)
    elif isComplex32(T):
      var v = value
      eigenMatrixCSSet(this.data, row, col, addr v)
    elif isComplex64(T):
      var v = value
      eigenMatrixCDSet(this.data, row, col, addr v)
  
  #[ algebra ]#

  method `+`(other: EigenMatrix[R, T]): EigenMatrix[R, T] =
    assert this.cols == other.cols and this.rows == other.rows, "shape mismatch"
    assert this.inner == other.inner and this.outer == other.outer, "stride mismatch"
    result = newEigenMatrix[R, T](this.rows, this.cols, this.inner, this.outer)
    when isReal32(T): eigenMatrixRSAdd(this.data, other.data, result.data)
    elif isReal64(T): eigenMatrixRDAdd(this.data, other.data, result.data)
    elif isComplex32(T): eigenMatrixCSAdd(this.data, other.data, result.data)
    elif isComplex64(T): eigenMatrixCDAdd(this.data, other.data, result.data)
  
  method `-`(other: EigenMatrix[R, T]): EigenMatrix[R, T] =
    assert this.cols == other.cols and this.rows == other.rows, "shape mismatch"
    assert this.inner == other.inner and this.outer == other.outer, "stride mismatch"
    result = newEigenMatrix[R, T](this.rows, this.cols, this.inner, this.outer)
    when isReal32(T): eigenMatrixRSSub(this.data, other.data, result.data)
    elif isReal64(T): eigenMatrixRDSub(this.data, other.data, result.data)
    elif isComplex32(T): eigenMatrixCSSub(this.data, other.data, result.data)
    elif isComplex64(T): eigenMatrixCDSub(this.data, other.data, result.data)
  
  method `*`(other: EigenMatrix[R, T]): EigenMatrix[R, T] =
    assert this.cols == other.rows, "shape mismatch for multiplication"
    assert this.inner == other.inner and this.outer == other.outer, "stride mismatch"
    result = newEigenMatrix[R, T](this.rows, other.cols, this.inner, this.outer)
    when isReal32(T): eigenMatrixRSMul(this.data, other.data, result.data)
    elif isReal64(T): eigenMatrixRDMul(this.data, other.data, result.data)
    elif isComplex32(T): eigenMatrixCSMul(this.data, other.data, result.data)
    elif isComplex64(T): eigenMatrixCDMul(this.data, other.data, result.data)
  
  method `+=`(other: EigenMatrix[R, T]) =
    assert this.cols == other.cols and this.rows == other.rows, "shape mismatch"
    assert this.inner == other.inner and this.outer == other.outer, "stride mismatch"
    when isReal32(T): eigenMatrixRSAddAssign(this.data, other.data)
    elif isReal64(T): eigenMatrixRDAddAssign(this.data, other.data)
    elif isComplex32(T): eigenMatrixCSAddAssign(this.data, other.data)
    elif isComplex64(T): eigenMatrixCDAddAssign(this.data, other.data)
  
  method `-=`(other: EigenMatrix[R, T]) =
    assert this.cols == other.cols and this.rows == other.rows, "shape mismatch"
    assert this.inner == other.inner and this.outer == other.outer, "stride mismatch"
    when isReal32(T): eigenMatrixRSSubAssign(this.data, other.data)
    elif isReal64(T): eigenMatrixRDSubAssign(this.data, other.data)
    elif isComplex32(T): eigenMatrixCSSubAssign(this.data, other.data)
    elif isComplex64(T): eigenMatrixCDSubAssign(this.data, other.data)
  
  method `*=`(other: EigenMatrix[R, T]) =
    assert this.cols == other.rows, "shape mismatch for multiplication"
    assert this.inner == other.inner and this.outer == other.outer, "stride mismatch"
    when isReal32(T): eigenMatrixRSMulAssign(this.data, other.data)
    elif isReal64(T): eigenMatrixRDMulAssign(this.data, other.data)
    elif isComplex32(T): eigenMatrixCSMulAssign(this.data, other.data)
    elif isComplex64(T): eigenMatrixCDMulAssign(this.data, other.data)
  
  #[ unary operations ]#

  method transpose: EigenMatrix[R, T] =
    result = newEigenMatrix[R, T](this.cols, this.rows, this.inner, this.outer)
    when isReal32(T): eigenMatrixRSTranspose(this.data, result.data)
    elif isReal64(T): eigenMatrixRDTranspose(this.data, result.data)
    elif isComplex32(T): eigenMatrixCSTranspose(this.data, result.data)
    elif isComplex64(T): eigenMatrixCDTranspose(this.data, result.data)
  
  method adjoint: EigenMatrix[R, T] =
    result = newEigenMatrix[R, T](this.cols, this.rows, this.inner, this.outer)
    when isReal32(T): eigenMatrixRSAdjoint(this.data, result.data)
    elif isReal64(T): eigenMatrixRDAdjoint(this.data, result.data)
    elif isComplex32(T): eigenMatrixCSAdjoint(this.data, result.data)
    elif isComplex64(T): eigenMatrixCDAdjoint(this.data, result.data)
  
  method conjugate: EigenMatrix[R, T] =
    result = newEigenMatrix[R, T](this.rows, this.cols, this.inner, this.outer)
    when isReal32(T): eigenMatrixRSConjugate(this.data, result.data)
    elif isReal64(T): eigenMatrixRDConjugate(this.data, result.data)
    elif isComplex32(T): eigenMatrixCSConjugate(this.data, result.data)
    elif isComplex64(T): eigenMatrixCDConjugate(this.data, result.data)
  
  method trace: T =
    assert this.rows == this.cols, "trace requires square matrix"
    when isReal32(T):
      return eigenMatrixRSTrace(this.data)
    elif isReal64(T):
      return eigenMatrixRDTrace(this.data)
    elif isComplex32(T):
      var res: Complex32
      eigenMatrixCSTrace(this.data, addr res)
      return res
    elif isComplex64(T):
      var res: Complex64
      eigenMatrixCDTrace(this.data, addr res)
      return res
  
  method determinant: T =
    assert this.rows == this.cols, "determinant requires square matrix"
    when isReal32(T):
      return eigenMatrixRSDet(this.data)
    elif isReal64(T):
      return eigenMatrixRDDet(this.data)
    elif isComplex32(T):
      var res: Complex32
      eigenMatrixCSDet(this.data, addr res)
      return res
    elif isComplex64(T):
      var res: Complex64
      eigenMatrixCDDet(this.data, addr res)
      return res
  
  method inverse: EigenMatrix[R, T] =
    assert this.rows == this.cols, "inverse requires square matrix"
    result = newEigenMatrix[R, T](this.rows, this.cols, this.inner, this.outer)
    when isReal32(T): eigenMatrixRSInverse(this.data, result.data)
    elif isReal64(T): eigenMatrixRDInverse(this.data, result.data)
    elif isComplex32(T): eigenMatrixCSInverse(this.data, result.data)
    elif isComplex64(T): eigenMatrixCDInverse(this.data, result.data)
  
  method `*`(scalar: T): EigenMatrix[R, T] =
    result = newEigenMatrix[R, T](this.rows, this.cols, this.inner, this.outer)
    when isReal32(T): eigenMatrixRSScalarMul(this.data, scalar, result.data)
    elif isReal64(T): eigenMatrixRDScalarMul(this.data, scalar, result.data)
    elif isComplex32(T):
      var s = scalar
      eigenMatrixCSScalarMul(this.data, addr s, result.data)
    elif isComplex64(T):
      var s = scalar
      eigenMatrixCDScalarMul(this.data, addr s, result.data)
  
  method `*=`(scalar: T) =
    when isReal32(T): eigenMatrixRSScalarMulAssign(this.data, scalar)
    elif isReal64(T): eigenMatrixRDScalarMulAssign(this.data, scalar)
    elif isComplex32(T):
      var s = scalar
      eigenMatrixCSScalarMulAssign(this.data, addr s)
    elif isComplex64(T):
      var s = scalar
      eigenMatrixCDScalarMulAssign(this.data, addr s)
  
  method `-`: EigenMatrix[R, T] =
    result = newEigenMatrix[R, T](this.rows, this.cols, this.inner, this.outer)
    when isReal32(T): eigenMatrixRSNegate(this.data, result.data)
    elif isReal64(T): eigenMatrixRDNegate(this.data, result.data)
    elif isComplex32(T): eigenMatrixCSNegate(this.data, result.data)
    elif isComplex64(T): eigenMatrixCDNegate(this.data, result.data)
  
  method norm: auto =
    when isReal32(T):
      return eigenMatrixRSNorm(this.data)
    elif isReal64(T):
      return eigenMatrixRDNorm(this.data)
    elif isComplex32(T):
      return eigenMatrixCSNorm(this.data)
    elif isComplex64(T):
      return eigenMatrixCDNorm(this.data)

when isMainModule:
  import std/[unittest, math]

  suite "EigenMatrix tests":
    var rdata = newSeq[float64](4)
    var cdata = newSeq[Complex64](4)

    test "real creation, destruction, and access":
      for i in 0..<4:
        rdata[i] = float64(i + 1)

      var rmtrx = newEigenMatrix(addr rdata[0], [2, 2], 1, 2)

      for i in 0..<2:
        for j in 0..<2: check rmtrx[i, j] == float64(i * 2 + j + 1)

      # write through matrix → underlying buffer
      rmtrx[0, 1] = 99.0
      check rdata[1] == 99.0

      # write through buffer → visible via matrix
      rdata[2] = 42.0
      check rmtrx[1, 0] == 42.0

    test "complex creation, destruction, and access":
      for i in 0..<4:
        cdata[i] = complex(float64(i + 1), float64(i + 2))

      var cmtrx = newEigenMatrix(addr cdata[0], [2, 2], 1, 2)
    
      for i in 0..<2:
        for j in 0..<2: 
          check cmtrx[i, j].re == float64(i * 2 + j + 1)
          check cmtrx[i, j].im == float64(i * 2 + j + 2)

      # write through complex matrix → underlying buffer
      cmtrx[1, 0] = complex(77.0, 88.0)
      check cdata[2].re == 77.0
      check cdata[2].im == 88.0

    test "temporary matrix creation and element access":
      var tmtrx = newEigenMatrix[2, float64](2, 3, 1, 3)
      check tmtrx.ownsData == true

      # zero-initialized
      for i in 0..<2:
        for j in 0..<3: check tmtrx[i, j] == 0.0

      # write and read back
      tmtrx[0, 0] = 10.0
      tmtrx[0, 2] = 30.0
      tmtrx[1, 1] = 50.0
      check tmtrx[0, 0] == 10.0
      check tmtrx[0, 2] == 30.0
      check tmtrx[1, 1] == 50.0

    test "real matrix addition":
      # A = [[1, 2], [3, 4]]  B = [[5, 6], [7, 8]]
      var adata = [1.0, 2.0, 3.0, 4.0]
      var bdata = [5.0, 6.0, 7.0, 8.0]
      var a = newEigenMatrix(addr adata[0], [2, 2], 1, 2)
      var b = newEigenMatrix(addr bdata[0], [2, 2], 1, 2)
      var c = a + b
      # C = [[6, 8], [10, 12]]
      check c[0, 0] == 6.0
      check c[0, 1] == 8.0
      check c[1, 0] == 10.0
      check c[1, 1] == 12.0

    test "real matrix subtraction":
      var adata = [5.0, 6.0, 7.0, 8.0]
      var bdata = [1.0, 2.0, 3.0, 4.0]
      var a = newEigenMatrix(addr adata[0], [2, 2], 1, 2)
      var b = newEigenMatrix(addr bdata[0], [2, 2], 1, 2)
      var c = a - b
      check c[0, 0] == 4.0
      check c[0, 1] == 4.0
      check c[1, 0] == 4.0
      check c[1, 1] == 4.0

    test "real matrix multiplication":
      # A = [[1, 2], [3, 4]]  B = [[5, 6], [7, 8]]
      # A*B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
      var adata = [1.0, 2.0, 3.0, 4.0]
      var bdata = [5.0, 6.0, 7.0, 8.0]
      var a = newEigenMatrix(addr adata[0], [2, 2], 1, 2)
      var b = newEigenMatrix(addr bdata[0], [2, 2], 1, 2)
      var c = a * b
      check c[0, 0] == 19.0
      check c[0, 1] == 22.0
      check c[1, 0] == 43.0
      check c[1, 1] == 50.0

    test "real compound assignment (+=, -=, *=)":
      var adata = [1.0, 2.0, 3.0, 4.0]
      var bdata = [5.0, 6.0, 7.0, 8.0]
      var a = newEigenMatrix(addr adata[0], [2, 2], 1, 2)
      var b = newEigenMatrix(addr bdata[0], [2, 2], 1, 2)

      # += : a becomes [[6, 8], [10, 12]]
      a += b
      check a[0, 0] == 6.0
      check a[0, 1] == 8.0
      check a[1, 0] == 10.0
      check a[1, 1] == 12.0
      # verify write-through to underlying buffer
      check adata[0] == 6.0

      # -= : a becomes [[1, 2], [3, 4]] again
      a -= b
      check a[0, 0] == 1.0
      check a[0, 1] == 2.0
      check a[1, 0] == 3.0
      check a[1, 1] == 4.0

      # *= : a becomes a*b = [[19, 22], [43, 50]]
      a *= b
      check a[0, 0] == 19.0
      check a[0, 1] == 22.0
      check a[1, 0] == 43.0
      check a[1, 1] == 50.0

    test "complex matrix algebra":
      var ca = [complex(1.0, 1.0), complex(2.0, 0.0),
                complex(0.0, 1.0), complex(3.0, -1.0)]
      var cb = [complex(1.0, 0.0), complex(0.0, 1.0),
                complex(1.0, 1.0), complex(2.0, 0.0)]
      var a = newEigenMatrix(addr ca[0], [2, 2], 1, 2)
      var b = newEigenMatrix(addr cb[0], [2, 2], 1, 2)

      # addition
      var s = a + b
      check s[0, 0] == complex(2.0, 1.0)
      check s[0, 1] == complex(2.0, 1.0)
      check s[1, 0] == complex(1.0, 2.0)
      check s[1, 1] == complex(5.0, -1.0)

      # subtraction
      var d = a - b
      check d[0, 0] == complex(0.0, 1.0)
      check d[0, 1] == complex(2.0, -1.0)
      check d[1, 0] == complex(-1.0, 0.0)
      check d[1, 1] == complex(1.0, -1.0)

      # multiplication: a*b
      # [0,0] = (1+i)(1+0i) + (2+0i)(1+i) = (1+i) + (2+2i) = 3+3i
      # [0,1] = (1+i)(0+i) + (2+0i)(2+0i) = (-1+i) + (4+0i) = 3+i
      # [1,0] = (0+i)(1+0i) + (3-i)(1+i) = (0+i) + (4+2i) = 4+3i
      # [1,1] = (0+i)(0+i) + (3-i)(2+0i) = (-1+0i) + (6-2i) = 5-2i
      var p = a * b
      check p[0, 0] == complex(3.0, 3.0)
      check p[0, 1] == complex(3.0, 1.0)
      check p[1, 0] == complex(4.0, 3.0)
      check p[1, 1] == complex(5.0, -2.0)

    test "real matrix transpose":
      # A = [[1, 2, 3], [4, 5, 6]]  →  A^T = [[1, 4], [2, 5], [3, 6]]
      var adata = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
      var a = newEigenMatrix(addr adata[0], [2, 3], 1, 3)
      var t = a.transpose()
      check t.rows == 3
      check t.cols == 2
      check t[0, 0] == 1.0
      check t[0, 1] == 4.0
      check t[1, 0] == 2.0
      check t[1, 1] == 5.0
      check t[2, 0] == 3.0
      check t[2, 1] == 6.0

    test "complex adjoint (conjugate transpose)":
      # A = [[1+i, 2+3i], [4-i, 5]]  →  A^H = [[1-i, 4+i], [2-3i, 5]]
      var ca = [complex(1.0, 1.0), complex(2.0, 3.0),
                complex(4.0, -1.0), complex(5.0, 0.0)]
      var a = newEigenMatrix(addr ca[0], [2, 2], 1, 2)
      var ah = a.adjoint()
      check ah[0, 0] == complex(1.0, -1.0)
      check ah[0, 1] == complex(4.0, 1.0)
      check ah[1, 0] == complex(2.0, -3.0)
      check ah[1, 1] == complex(5.0, 0.0)

    test "complex conjugate":
      var ca = [complex(1.0, 2.0), complex(3.0, -4.0),
                complex(-5.0, 6.0), complex(7.0, 0.0)]
      var a = newEigenMatrix(addr ca[0], [2, 2], 1, 2)
      var c = a.conjugate()
      check c[0, 0] == complex(1.0, -2.0)
      check c[0, 1] == complex(3.0, 4.0)
      check c[1, 0] == complex(-5.0, -6.0)
      check c[1, 1] == complex(7.0, 0.0)

    test "trace":
      # A = [[1, 2], [3, 4]]  →  tr(A) = 5
      var adata = [1.0, 2.0, 3.0, 4.0]
      var a = newEigenMatrix(addr adata[0], [2, 2], 1, 2)
      check a.trace() == 5.0

    test "complex trace":
      var ca = [complex(1.0, 2.0), complex(0.0, 0.0),
                complex(0.0, 0.0), complex(3.0, -1.0)]
      var a = newEigenMatrix(addr ca[0], [2, 2], 1, 2)
      var t = a.trace()
      check t == complex(4.0, 1.0)

    test "determinant":
      # A = [[1, 2], [3, 4]]  →  det = 1*4 - 2*3 = -2
      var adata = [1.0, 2.0, 3.0, 4.0]
      var a = newEigenMatrix(addr adata[0], [2, 2], 1, 2)
      check abs(a.determinant() - (-2.0)) < 1e-12

    test "inverse":
      # A = [[1, 2], [3, 4]]  →  A^-1 = [[-2, 1], [1.5, -0.5]]
      var adata = [1.0, 2.0, 3.0, 4.0]
      var a = newEigenMatrix(addr adata[0], [2, 2], 1, 2)
      var inv = a.inverse()
      check abs(inv[0, 0] - (-2.0)) < 1e-12
      check abs(inv[0, 1] - 1.0) < 1e-12
      check abs(inv[1, 0] - 1.5) < 1e-12
      check abs(inv[1, 1] - (-0.5)) < 1e-12

      # A * A^-1 ≈ I
      var identity = a * inv
      check abs(identity[0, 0] - 1.0) < 1e-12
      check abs(identity[0, 1] - 0.0) < 1e-12
      check abs(identity[1, 0] - 0.0) < 1e-12
      check abs(identity[1, 1] - 1.0) < 1e-12

    test "scalar multiply and scalar *=":
      var adata = [1.0, 2.0, 3.0, 4.0]
      var a = newEigenMatrix(addr adata[0], [2, 2], 1, 2)
      var s = a * 3.0
      check s[0, 0] == 3.0
      check s[0, 1] == 6.0
      check s[1, 0] == 9.0
      check s[1, 1] == 12.0

      # in-place scalar *=
      a *= 10.0
      check a[0, 0] == 10.0
      check a[1, 1] == 40.0
      check adata[0] == 10.0  # write-through

    test "unary negate":
      var adata = [1.0, -2.0, 3.0, -4.0]
      var a = newEigenMatrix(addr adata[0], [2, 2], 1, 2)
      var n = -a
      check n[0, 0] == -1.0
      check n[0, 1] == 2.0
      check n[1, 0] == -3.0
      check n[1, 1] == 4.0

    test "Frobenius norm":
      # ||[[1, 2], [3, 4]]|| = sqrt(1+4+9+16) = sqrt(30)
      var adata = [1.0, 2.0, 3.0, 4.0]
      var a = newEigenMatrix(addr adata[0], [2, 2], 1, 2)
      check abs(a.norm() - sqrt(30.0)) < 1e-12