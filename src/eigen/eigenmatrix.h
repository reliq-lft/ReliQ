/** 
 * ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
 * Source file: src/eigen/eigenmatrix.h
 * Contact: reliq-lft@proton.me
 *
 * Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>
 *
 * MIT License
 * 
 * Copyright (c) 2025 reliq-lft
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
  * The above copyright notice and this permission notice shall be included in all
  * copies or substantial portions of the Software.
  * 
  * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
  * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
  * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <Eigen/Dense>

#pragma once

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
  size_t needed = (size_t)(rows - 1) * outer + (size_t)(cols - 1) * inner + 1;
  auto* data = new float[needed]();
  auto* matrix = new EigenMapMatrixRS(data, rows, cols, EigenStride(outer, inner));
  return EigenMatrixHandleRS{matrix};
}

inline EigenMatrixHandleRD createTempEigenMatrixRD(
  int rows, int cols, int outer, int inner
) {
  size_t needed = (size_t)(rows - 1) * outer + (size_t)(cols - 1) * inner + 1;
  auto* data = new double[needed]();
  auto* matrix = new EigenMapMatrixRD(data, rows, cols, EigenStride(outer, inner));
  return EigenMatrixHandleRD{matrix};
}

inline EigenMatrixHandleCS createTempEigenMatrixCS(
  int rows, int cols, int outer, int inner
) {
  size_t needed = (size_t)(rows - 1) * outer + (size_t)(cols - 1) * inner + 1;
  auto* data = new std::complex<float>[needed]();
  auto* matrix = new EigenMapMatrixCS(data, rows, cols, EigenStride(outer, inner));
  return EigenMatrixHandleCS{matrix};
}

inline EigenMatrixHandleCD createTempEigenMatrixCD(
  int rows, int cols, int outer, int inner
) {
  size_t needed = (size_t)(rows - 1) * outer + (size_t)(cols - 1) * inner + 1;
  auto* data = new std::complex<double>[needed]();
  auto* matrix = new EigenMapMatrixCD(data, rows, cols, EigenStride(outer, inner));
  return EigenMatrixHandleCD{matrix};
}

/* destructors */

inline void destroyEigenMatrixRS(EigenMatrixHandleRS handle, bool ownsData) {
  if (!handle.mat) return;
  if (ownsData) delete[] handle.mat->data();
  delete handle.mat;
}
inline void destroyEigenMatrixRD(EigenMatrixHandleRD handle, bool ownsData) {
  if (!handle.mat) return;
  if (ownsData) delete[] handle.mat->data();
  delete handle.mat;
}
inline void destroyEigenMatrixCS(EigenMatrixHandleCS handle, bool ownsData) {
  if (!handle.mat) return;
  if (ownsData) delete[] handle.mat->data();
  delete handle.mat;
}
inline void destroyEigenMatrixCD(EigenMatrixHandleCD handle, bool ownsData) {
  if (!handle.mat) return;
  if (ownsData) delete[] handle.mat->data();
  delete handle.mat;
}

/* clone — create new Map pointing to same data (ownsData=false copy) */

inline EigenMatrixHandleRS cloneEigenMatrixRS(EigenMatrixHandleRS handle) {
  if (!handle.mat) return EigenMatrixHandleRS{nullptr};
  auto* mat = new EigenMapMatrixRS(handle.mat->data(), handle.mat->rows(), handle.mat->cols(),
                                   EigenStride(handle.mat->outerStride(), handle.mat->innerStride()));
  return EigenMatrixHandleRS{mat};
}
inline EigenMatrixHandleRD cloneEigenMatrixRD(EigenMatrixHandleRD handle) {
  if (!handle.mat) return EigenMatrixHandleRD{nullptr};
  auto* mat = new EigenMapMatrixRD(handle.mat->data(), handle.mat->rows(), handle.mat->cols(),
                                   EigenStride(handle.mat->outerStride(), handle.mat->innerStride()));
  return EigenMatrixHandleRD{mat};
}
inline EigenMatrixHandleCS cloneEigenMatrixCS(EigenMatrixHandleCS handle) {
  if (!handle.mat) return EigenMatrixHandleCS{nullptr};
  auto* mat = new EigenMapMatrixCS(handle.mat->data(), handle.mat->rows(), handle.mat->cols(),
                                   EigenStride(handle.mat->outerStride(), handle.mat->innerStride()));
  return EigenMatrixHandleCS{mat};
}
inline EigenMatrixHandleCD cloneEigenMatrixCD(EigenMatrixHandleCD handle) {
  if (!handle.mat) return EigenMatrixHandleCD{nullptr};
  auto* mat = new EigenMapMatrixCD(handle.mat->data(), handle.mat->rows(), handle.mat->cols(),
                                   EigenStride(handle.mat->outerStride(), handle.mat->innerStride()));
  return EigenMatrixHandleCD{mat};
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

/* copy-from — write all elements of src into dst through the view */

inline void eigenMatrixRSCopyFrom(EigenMatrixHandleRS dst, const EigenMatrixHandleRS src) {
  *dst.mat = *src.mat;
}
inline void eigenMatrixRDCopyFrom(EigenMatrixHandleRD dst, const EigenMatrixHandleRD src) {
  *dst.mat = *src.mat;
}
inline void eigenMatrixCSCopyFrom(EigenMatrixHandleCS dst, const EigenMatrixHandleCS src) {
  *dst.mat = *src.mat;
}
inline void eigenMatrixCDCopyFrom(EigenMatrixHandleCD dst, const EigenMatrixHandleCD src) {
  *dst.mat = *src.mat;
}

/* fill — set all elements to a scalar value */

inline void eigenMatrixRSFill(EigenMatrixHandleRS handle, float value) {
  handle.mat->fill(value);
}
inline void eigenMatrixRDFill(EigenMatrixHandleRD handle, double value) {
  handle.mat->fill(value);
}
inline void eigenMatrixCSFill(EigenMatrixHandleCS handle, const void* value) {
  handle.mat->fill(*static_cast<const std::complex<float>*>(value));
}
inline void eigenMatrixCDFill(EigenMatrixHandleCD handle, const void* value) {
  handle.mat->fill(*static_cast<const std::complex<double>*>(value));
}