/** 
 * ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
 * Source file: src/eigen/eigenvector.h
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

#define VectorStride Eigen::InnerStride<Eigen::Dynamic>

/* base vector types */

using EigenVectorRS = Eigen::Matrix<float, Eigen::Dynamic, 1>;
using EigenVectorRD = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using EigenVectorCS = Eigen::Matrix<std::complex<float>, Eigen::Dynamic, 1>;
using EigenVectorCD = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;

using EigenMapVectorRS = Eigen::Map<EigenVectorRS, Eigen::Unaligned, VectorStride>;
using EigenMapVectorRD = Eigen::Map<EigenVectorRD, Eigen::Unaligned, VectorStride>;
using EigenMapVectorCS = Eigen::Map<EigenVectorCS, Eigen::Unaligned, VectorStride>;
using EigenMapVectorCD = Eigen::Map<EigenVectorCD, Eigen::Unaligned, VectorStride>;

struct EigenVectorHandleRS { EigenMapVectorRS* vec; };
struct EigenVectorHandleRD { EigenMapVectorRD* vec; };
struct EigenVectorHandleCS { EigenMapVectorCS* vec; };
struct EigenVectorHandleCD { EigenMapVectorCD* vec; };

/* constructors */

inline EigenVectorHandleRS createEigenVectorRS(
  float* data, int size, int stride
) {
  auto* vec = new EigenMapVectorRS(data, size, VectorStride(stride));
  return EigenVectorHandleRS{vec};
}

inline EigenVectorHandleRD createEigenVectorRD(
  double* data, int size, int stride
) {
  auto* vec = new EigenMapVectorRD(data, size, VectorStride(stride));
  return EigenVectorHandleRD{vec};
}

inline EigenVectorHandleCS createEigenVectorCS(
  void* data, int size, int stride
) {
  auto* vec = new EigenMapVectorCS(static_cast<std::complex<float>*>(data), size, VectorStride(stride));
  return EigenVectorHandleCS{vec};
}

inline EigenVectorHandleCD createEigenVectorCD(
  void* data, int size, int stride
) {
  auto* vec = new EigenMapVectorCD(static_cast<std::complex<double>*>(data), size, VectorStride(stride));
  return EigenVectorHandleCD{vec};
}

/* temporary constructors — allocate owned data buffer */

inline EigenVectorHandleRS createTempEigenVectorRS(int size, int stride) {
  size_t needed = (size_t)(size - 1) * stride + 1;
  auto* data = new float[needed]();
  auto* vec = new EigenMapVectorRS(data, size, VectorStride(stride));
  return EigenVectorHandleRS{vec};
}

inline EigenVectorHandleRD createTempEigenVectorRD(int size, int stride) {
  size_t needed = (size_t)(size - 1) * stride + 1;
  auto* data = new double[needed]();
  auto* vec = new EigenMapVectorRD(data, size, VectorStride(stride));
  return EigenVectorHandleRD{vec};
}

inline EigenVectorHandleCS createTempEigenVectorCS(int size, int stride) {
  size_t needed = (size_t)(size - 1) * stride + 1;
  auto* data = new std::complex<float>[needed]();
  auto* vec = new EigenMapVectorCS(data, size, VectorStride(stride));
  return EigenVectorHandleCS{vec};
}

inline EigenVectorHandleCD createTempEigenVectorCD(int size, int stride) {
  size_t needed = (size_t)(size - 1) * stride + 1;
  auto* data = new std::complex<double>[needed]();
  auto* vec = new EigenMapVectorCD(data, size, VectorStride(stride));
  return EigenVectorHandleCD{vec};
}

/* destructors */

inline void destroyEigenVectorRS(EigenVectorHandleRS handle, bool ownsData) {
  if (!handle.vec) return;
  if (ownsData) delete[] handle.vec->data();
  delete handle.vec;
}
inline void destroyEigenVectorRD(EigenVectorHandleRD handle, bool ownsData) {
  if (!handle.vec) return;
  if (ownsData) delete[] handle.vec->data();
  delete handle.vec;
}
inline void destroyEigenVectorCS(EigenVectorHandleCS handle, bool ownsData) {
  if (!handle.vec) return;
  if (ownsData) delete[] handle.vec->data();
  delete handle.vec;
}
inline void destroyEigenVectorCD(EigenVectorHandleCD handle, bool ownsData) {
  if (!handle.vec) return;
  if (ownsData) delete[] handle.vec->data();
  delete handle.vec;
}

/* clone — create new Map pointing to same data (ownsData=false copy) */

inline EigenVectorHandleRS cloneEigenVectorRS(EigenVectorHandleRS handle) {
  if (!handle.vec) return EigenVectorHandleRS{nullptr};
  auto* vec = new EigenMapVectorRS(handle.vec->data(), handle.vec->rows(),
                                   VectorStride(handle.vec->innerStride()));
  return EigenVectorHandleRS{vec};
}
inline EigenVectorHandleRD cloneEigenVectorRD(EigenVectorHandleRD handle) {
  if (!handle.vec) return EigenVectorHandleRD{nullptr};
  auto* vec = new EigenMapVectorRD(handle.vec->data(), handle.vec->rows(),
                                   VectorStride(handle.vec->innerStride()));
  return EigenVectorHandleRD{vec};
}
inline EigenVectorHandleCS cloneEigenVectorCS(EigenVectorHandleCS handle) {
  if (!handle.vec) return EigenVectorHandleCS{nullptr};
  auto* vec = new EigenMapVectorCS(handle.vec->data(), handle.vec->rows(),
                                   VectorStride(handle.vec->innerStride()));
  return EigenVectorHandleCS{vec};
}
inline EigenVectorHandleCD cloneEigenVectorCD(EigenVectorHandleCD handle) {
  if (!handle.vec) return EigenVectorHandleCD{nullptr};
  auto* vec = new EigenMapVectorCD(handle.vec->data(), handle.vec->rows(),
                                   VectorStride(handle.vec->innerStride()));
  return EigenVectorHandleCD{vec};
}

/* accessors */

inline float eigenVectorRSGet(const EigenVectorHandleRS handle, int idx) {
  return (*handle.vec)(idx);
}

inline double eigenVectorRDGet(const EigenVectorHandleRD handle, int idx) {
  return (*handle.vec)(idx);
}

inline void eigenVectorCSGet(
  const EigenVectorHandleCS handle, int idx, void* out
) { *static_cast<std::complex<float>*>(out) = (*handle.vec)(idx); }

inline void eigenVectorCDGet(
  const EigenVectorHandleCD handle, int idx, void* out
) { *static_cast<std::complex<double>*>(out) = (*handle.vec)(idx); }

inline void eigenVectorRSSet(EigenVectorHandleRS handle, int idx, float value) {
  (*handle.vec)(idx) = value;
}

inline void eigenVectorRDSet(EigenVectorHandleRD handle, int idx, double value) {
  (*handle.vec)(idx) = value;
}

inline void eigenVectorCSSet(
  EigenVectorHandleCS handle, int idx, const void* value
) { (*handle.vec)(idx) = *static_cast<const std::complex<float>*>(value); }

inline void eigenVectorCDSet(
  EigenVectorHandleCD handle, int idx, const void* value
) { (*handle.vec)(idx) = *static_cast<const std::complex<double>*>(value); }

/* algebra */

inline void eigenVectorRSAdd(
  const EigenVectorHandleRS a, const EigenVectorHandleRS b, EigenVectorHandleRS out
) { *out.vec = *a.vec + *b.vec; }

inline void eigenVectorRDAdd(
  const EigenVectorHandleRD a, const EigenVectorHandleRD b, EigenVectorHandleRD out
) { *out.vec = *a.vec + *b.vec; }

inline void eigenVectorCSAdd(
  const EigenVectorHandleCS a, const EigenVectorHandleCS b, EigenVectorHandleCS out
) { *out.vec = *a.vec + *b.vec; }

inline void eigenVectorCDAdd(
  const EigenVectorHandleCD a, const EigenVectorHandleCD b, EigenVectorHandleCD out
) { *out.vec = *a.vec + *b.vec; }

inline void eigenVectorRSSub(
  const EigenVectorHandleRS a, const EigenVectorHandleRS b, EigenVectorHandleRS out
) { *out.vec = *a.vec - *b.vec; }

inline void eigenVectorRDSub(
  const EigenVectorHandleRD a, const EigenVectorHandleRD b, EigenVectorHandleRD out
) { *out.vec = *a.vec - *b.vec; }

inline void eigenVectorCSSub(
  const EigenVectorHandleCS a, const EigenVectorHandleCS b, EigenVectorHandleCS out
) { *out.vec = *a.vec - *b.vec; }

inline void eigenVectorCDSub(
  const EigenVectorHandleCD a, const EigenVectorHandleCD b, EigenVectorHandleCD out
) { *out.vec = *a.vec - *b.vec; }

/* element-wise multiply */

inline void eigenVectorRSMul(
  const EigenVectorHandleRS a, const EigenVectorHandleRS b, EigenVectorHandleRS out
) { *out.vec = a.vec->cwiseProduct(*b.vec); }

inline void eigenVectorRDMul(
  const EigenVectorHandleRD a, const EigenVectorHandleRD b, EigenVectorHandleRD out
) { *out.vec = a.vec->cwiseProduct(*b.vec); }

inline void eigenVectorCSMul(
  const EigenVectorHandleCS a, const EigenVectorHandleCS b, EigenVectorHandleCS out
) { *out.vec = a.vec->cwiseProduct(*b.vec); }

inline void eigenVectorCDMul(
  const EigenVectorHandleCD a, const EigenVectorHandleCD b, EigenVectorHandleCD out
) { *out.vec = a.vec->cwiseProduct(*b.vec); }

/* dot product */

inline float eigenVectorRSDot(
  const EigenVectorHandleRS a, const EigenVectorHandleRS b
) { return a.vec->dot(*b.vec); }

inline double eigenVectorRDDot(
  const EigenVectorHandleRD a, const EigenVectorHandleRD b
) { return a.vec->dot(*b.vec); }

inline void eigenVectorCSDot(
  const EigenVectorHandleCS a, const EigenVectorHandleCS b, void* out
) { *static_cast<std::complex<float>*>(out) = a.vec->dot(*b.vec); }

inline void eigenVectorCDDot(
  const EigenVectorHandleCD a, const EigenVectorHandleCD b, void* out
) { *static_cast<std::complex<double>*>(out) = a.vec->dot(*b.vec); }

/* norm */

inline float eigenVectorRSNorm(const EigenVectorHandleRS handle) {
  return handle.vec->norm();
}

inline double eigenVectorRDNorm(const EigenVectorHandleRD handle) {
  return handle.vec->norm();
}

inline float eigenVectorCSNorm(const EigenVectorHandleCS handle) {
  return handle.vec->norm();
}

inline double eigenVectorCDNorm(const EigenVectorHandleCD handle) {
  return handle.vec->norm();
}

/* compound assignment */

inline void eigenVectorRSAddAssign(
  EigenVectorHandleRS a, const EigenVectorHandleRS b
) { *a.vec += *b.vec; }

inline void eigenVectorRDAddAssign(
  EigenVectorHandleRD a, const EigenVectorHandleRD b
) { *a.vec += *b.vec; }

inline void eigenVectorCSAddAssign(
  EigenVectorHandleCS a, const EigenVectorHandleCS b
) { *a.vec += *b.vec; }

inline void eigenVectorCDAddAssign(
  EigenVectorHandleCD a, const EigenVectorHandleCD b
) { *a.vec += *b.vec; }

inline void eigenVectorRSSubAssign(
  EigenVectorHandleRS a, const EigenVectorHandleRS b
) { *a.vec -= *b.vec; }

inline void eigenVectorRDSubAssign(
  EigenVectorHandleRD a, const EigenVectorHandleRD b
) { *a.vec -= *b.vec; }

inline void eigenVectorCSSubAssign(
  EigenVectorHandleCS a, const EigenVectorHandleCS b
) { *a.vec -= *b.vec; }

inline void eigenVectorCDSubAssign(
  EigenVectorHandleCD a, const EigenVectorHandleCD b
) { *a.vec -= *b.vec; }

/* scalar multiply */

inline void eigenVectorRSScalarMul(
  const EigenVectorHandleRS a, float s, EigenVectorHandleRS out
) { *out.vec = *a.vec * s; }

inline void eigenVectorRDScalarMul(
  const EigenVectorHandleRD a, double s, EigenVectorHandleRD out
) { *out.vec = *a.vec * s; }

inline void eigenVectorCSScalarMul(
  const EigenVectorHandleCS a, const void* s, EigenVectorHandleCS out
) { *out.vec = *a.vec * *static_cast<const std::complex<float>*>(s); }

inline void eigenVectorCDScalarMul(
  const EigenVectorHandleCD a, const void* s, EigenVectorHandleCD out
) { *out.vec = *a.vec * *static_cast<const std::complex<double>*>(s); }

inline void eigenVectorRSScalarMulAssign(EigenVectorHandleRS a, float s) {
  *a.vec *= s;
}

inline void eigenVectorRDScalarMulAssign(EigenVectorHandleRD a, double s) {
  *a.vec *= s;
}

inline void eigenVectorCSScalarMulAssign(EigenVectorHandleCS a, const void* s) {
  *a.vec *= *static_cast<const std::complex<float>*>(s);
}

inline void eigenVectorCDScalarMulAssign(EigenVectorHandleCD a, const void* s) {
  *a.vec *= *static_cast<const std::complex<double>*>(s);
}

/* conjugate */

inline void eigenVectorRSConjugate(
  const EigenVectorHandleRS a, EigenVectorHandleRS out
) { *out.vec = a.vec->conjugate(); }

inline void eigenVectorRDConjugate(
  const EigenVectorHandleRD a, EigenVectorHandleRD out
) { *out.vec = a.vec->conjugate(); }

inline void eigenVectorCSConjugate(
  const EigenVectorHandleCS a, EigenVectorHandleCS out
) { *out.vec = a.vec->conjugate(); }

inline void eigenVectorCDConjugate(
  const EigenVectorHandleCD a, EigenVectorHandleCD out
) { *out.vec = a.vec->conjugate(); }

/* negate */

inline void eigenVectorRSNegate(
  const EigenVectorHandleRS a, EigenVectorHandleRS out
) { *out.vec = -(*a.vec); }

inline void eigenVectorRDNegate(
  const EigenVectorHandleRD a, EigenVectorHandleRD out
) { *out.vec = -(*a.vec); }

inline void eigenVectorCSNegate(
  const EigenVectorHandleCS a, EigenVectorHandleCS out
) { *out.vec = -(*a.vec); }

inline void eigenVectorCDNegate(
  const EigenVectorHandleCD a, EigenVectorHandleCD out
) { *out.vec = -(*a.vec); }

/* normalized (returns unit vector copy) */

inline void eigenVectorRSNormalized(
  const EigenVectorHandleRS a, EigenVectorHandleRS out
) { *out.vec = a.vec->normalized(); }

inline void eigenVectorRDNormalized(
  const EigenVectorHandleRD a, EigenVectorHandleRD out
) { *out.vec = a.vec->normalized(); }

inline void eigenVectorCSNormalized(
  const EigenVectorHandleCS a, EigenVectorHandleCS out
) { *out.vec = a.vec->normalized(); }

inline void eigenVectorCDNormalized(
  const EigenVectorHandleCD a, EigenVectorHandleCD out
) { *out.vec = a.vec->normalized(); }

/* normalize (in-place) */

inline void eigenVectorRSNormalize(EigenVectorHandleRS a) {
  a.vec->normalize();
}

inline void eigenVectorRDNormalize(EigenVectorHandleRD a) {
  a.vec->normalize();
}

inline void eigenVectorCSNormalize(EigenVectorHandleCS a) {
  a.vec->normalize();
}

inline void eigenVectorCDNormalize(EigenVectorHandleCD a) {
  a.vec->normalize();
}

/* squaredNorm */

inline float eigenVectorRSSquaredNorm(const EigenVectorHandleRS handle) {
  return handle.vec->squaredNorm();
}

inline double eigenVectorRDSquaredNorm(const EigenVectorHandleRD handle) {
  return handle.vec->squaredNorm();
}

inline float eigenVectorCSSquaredNorm(const EigenVectorHandleCS handle) {
  return handle.vec->squaredNorm();
}

inline double eigenVectorCDSquaredNorm(const EigenVectorHandleCD handle) {
  return handle.vec->squaredNorm();
}

/* cross product (3D only) — copy to Vector3 for compile-time size check */

inline void eigenVectorRSCross(
  const EigenVectorHandleRS a, const EigenVectorHandleRS b, EigenVectorHandleRS out
) {
  Eigen::Vector3f va = *a.vec;
  Eigen::Vector3f vb = *b.vec;
  Eigen::Vector3f vc = va.cross(vb);
  *out.vec = vc;
}

inline void eigenVectorRDCross(
  const EigenVectorHandleRD a, const EigenVectorHandleRD b, EigenVectorHandleRD out
) {
  Eigen::Vector3d va = *a.vec;
  Eigen::Vector3d vb = *b.vec;
  Eigen::Vector3d vc = va.cross(vb);
  *out.vec = vc;
}

inline void eigenVectorCSCross(
  const EigenVectorHandleCS a, const EigenVectorHandleCS b, EigenVectorHandleCS out
) {
  Eigen::Vector3cf va = *a.vec;
  Eigen::Vector3cf vb = *b.vec;
  Eigen::Vector3cf vc = va.cross(vb);
  *out.vec = vc;
}

inline void eigenVectorCDCross(
  const EigenVectorHandleCD a, const EigenVectorHandleCD b, EigenVectorHandleCD out
) {
  Eigen::Vector3cd va = *a.vec;
  Eigen::Vector3cd vb = *b.vec;
  Eigen::Vector3cd vc = va.cross(vb);
  *out.vec = vc;
}

/* copy-from — write all elements of src into dst through the view */

inline void eigenVectorRSCopyFrom(EigenVectorHandleRS dst, const EigenVectorHandleRS src) {
  *dst.vec = *src.vec;
}
inline void eigenVectorRDCopyFrom(EigenVectorHandleRD dst, const EigenVectorHandleRD src) {
  *dst.vec = *src.vec;
}
inline void eigenVectorCSCopyFrom(EigenVectorHandleCS dst, const EigenVectorHandleCS src) {
  *dst.vec = *src.vec;
}
inline void eigenVectorCDCopyFrom(EigenVectorHandleCD dst, const EigenVectorHandleCD src) {
  *dst.vec = *src.vec;
}

/* fill — set all elements to a scalar value */

inline void eigenVectorRSFill(EigenVectorHandleRS handle, float value) {
  handle.vec->fill(value);
}
inline void eigenVectorRDFill(EigenVectorHandleRD handle, double value) {
  handle.vec->fill(value);
}
inline void eigenVectorCSFill(EigenVectorHandleCS handle, const void* value) {
  handle.vec->fill(*static_cast<const std::complex<float>*>(value));
}
inline void eigenVectorCDFill(EigenVectorHandleCD handle, const void* value) {
  handle.vec->fill(*static_cast<const std::complex<double>*>(value));
}