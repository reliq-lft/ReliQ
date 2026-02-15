#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/eigen/eigenvector.nim
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
  auto* data = new float[size]();
  auto* vec = new EigenMapVectorRS(data, size, VectorStride(stride));
  return EigenVectorHandleRS{vec};
}

inline EigenVectorHandleRD createTempEigenVectorRD(int size, int stride) {
  auto* data = new double[size]();
  auto* vec = new EigenMapVectorRD(data, size, VectorStride(stride));
  return EigenVectorHandleRD{vec};
}

inline EigenVectorHandleCS createTempEigenVectorCS(int size, int stride) {
  auto* data = new std::complex<float>[size]();
  auto* vec = new EigenMapVectorCS(data, size, VectorStride(stride));
  return EigenVectorHandleCS{vec};
}

inline EigenVectorHandleCD createTempEigenVectorCD(int size, int stride) {
  auto* data = new std::complex<double>[size]();
  auto* vec = new EigenMapVectorCD(data, size, VectorStride(stride));
  return EigenVectorHandleCD{vec};
}

/* destructors */

inline void destroyEigenVectorRS(EigenVectorHandleRS handle, bool ownsData) {
  if (ownsData) delete[] handle.vec->data();
  delete handle.vec;
}
inline void destroyEigenVectorRD(EigenVectorHandleRD handle, bool ownsData) {
  if (ownsData) delete[] handle.vec->data();
  delete handle.vec;
}
inline void destroyEigenVectorCS(EigenVectorHandleCS handle, bool ownsData) {
  if (ownsData) delete[] handle.vec->data();
  delete handle.vec;
}
inline void destroyEigenVectorCD(EigenVectorHandleCD handle, bool ownsData) {
  if (ownsData) delete[] handle.vec->data();
  delete handle.vec;
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
""".}

type
  EigenVectorHandleRS* {.importcpp: "EigenVectorHandleRS".} = object
  EigenVectorHandleRD* {.importcpp: "EigenVectorHandleRD".} = object
  EigenVectorHandleCS* {.importcpp: "EigenVectorHandleCS".} = object
  EigenVectorHandleCD* {.importcpp: "EigenVectorHandleCD".} = object

record EigenVector*[T]:
  var size: int
  var stride: int
  var ownsData: bool
  when isReal32(T):
    var data*: EigenVectorHandleRS
  elif isReal64(T):
    var data*: EigenVectorHandleRD
  elif isComplex32(T):
    var data*: EigenVectorHandleCS
  elif isComplex64(T):
    var data*: EigenVectorHandleCD

#[ eigen wrapper ]#

# constructors

proc createEigenVectorRS(
  data: ptr float32; size, stride: int
): EigenVectorHandleRS {.importcpp: "createEigenVectorRS(@)".}

proc createEigenVectorRD(
  data: ptr float64; size, stride: int
): EigenVectorHandleRD {.importcpp: "createEigenVectorRD(@)".}

proc createEigenVectorCS(
  data: pointer; size, stride: int
): EigenVectorHandleCS {.importcpp: "createEigenVectorCS(@)".}

proc createEigenVectorCD(
  data: pointer; size, stride: int
): EigenVectorHandleCD {.importcpp: "createEigenVectorCD(@)".}

# temp constructors

proc createTempEigenVectorRS(
  size, stride: int
): EigenVectorHandleRS {.importcpp: "createTempEigenVectorRS(@)".}

proc createTempEigenVectorRD(
  size, stride: int
): EigenVectorHandleRD {.importcpp: "createTempEigenVectorRD(@)".}

proc createTempEigenVectorCS(
  size, stride: int
): EigenVectorHandleCS {.importcpp: "createTempEigenVectorCS(@)".}

proc createTempEigenVectorCD(
  size, stride: int
): EigenVectorHandleCD {.importcpp: "createTempEigenVectorCD(@)".}

# destructors

proc destroyEigenVectorRS(handle: EigenVectorHandleRS, ownsData: bool) 
  {.importcpp: "destroyEigenVectorRS(@)".}

proc destroyEigenVectorRD(handle: EigenVectorHandleRD, ownsData: bool) 
  {.importcpp: "destroyEigenVectorRD(@)".}

proc destroyEigenVectorCS(handle: EigenVectorHandleCS, ownsData: bool) 
  {.importcpp: "destroyEigenVectorCS(@)".}

proc destroyEigenVectorCD(handle: EigenVectorHandleCD, ownsData: bool) 
  {.importcpp: "destroyEigenVectorCD(@)".}

# accessors

proc eigenVectorRSGet(handle: EigenVectorHandleRS, idx: int): float32 
  {.importcpp: "eigenVectorRSGet(@)".}

proc eigenVectorRDGet(handle: EigenVectorHandleRD, idx: int): float64 
  {.importcpp: "eigenVectorRDGet(@)".}

proc eigenVectorCSGet(handle: EigenVectorHandleCS, idx: int, outVal: pointer) 
  {.importcpp: "eigenVectorCSGet(@)".}

proc eigenVectorCDGet(handle: EigenVectorHandleCD, idx: int, outVal: pointer) 
  {.importcpp: "eigenVectorCDGet(@)".}

proc eigenVectorRSSet(handle: EigenVectorHandleRS, idx: int, value: float32) 
  {.importcpp: "eigenVectorRSSet(@)".}

proc eigenVectorRDSet(handle: EigenVectorHandleRD, idx: int, value: float64) 
  {.importcpp: "eigenVectorRDSet(@)".}

proc eigenVectorCSSet(handle: EigenVectorHandleCS, idx: int, value: pointer) 
  {.importcpp: "eigenVectorCSSet(@)".}

proc eigenVectorCDSet(handle: EigenVectorHandleCD, idx: int, value: pointer) 
  {.importcpp: "eigenVectorCDSet(@)".}

# algebra

proc eigenVectorRSAdd(a, b: EigenVectorHandleRS; c: EigenVectorHandleRS) 
  {.importcpp: "eigenVectorRSAdd(@)".}

proc eigenVectorRDAdd(a, b: EigenVectorHandleRD; c: EigenVectorHandleRD) 
  {.importcpp: "eigenVectorRDAdd(@)".}

proc eigenVectorCSAdd(a, b: EigenVectorHandleCS; c: EigenVectorHandleCS) 
  {.importcpp: "eigenVectorCSAdd(@)".}

proc eigenVectorCDAdd(a, b: EigenVectorHandleCD; c: EigenVectorHandleCD) 
  {.importcpp: "eigenVectorCDAdd(@)".}

proc eigenVectorRSSub(a, b: EigenVectorHandleRS; c: EigenVectorHandleRS) 
  {.importcpp: "eigenVectorRSSub(@)".}

proc eigenVectorRDSub(a, b: EigenVectorHandleRD; c: EigenVectorHandleRD) 
  {.importcpp: "eigenVectorRDSub(@)".}

proc eigenVectorCSSub(a, b: EigenVectorHandleCS; c: EigenVectorHandleCS) 
  {.importcpp: "eigenVectorCSSub(@)".}

proc eigenVectorCDSub(a, b: EigenVectorHandleCD; c: EigenVectorHandleCD) 
  {.importcpp: "eigenVectorCDSub(@)".}

proc eigenVectorRSMul(a, b: EigenVectorHandleRS; c: EigenVectorHandleRS) 
  {.importcpp: "eigenVectorRSMul(@)".}

proc eigenVectorRDMul(a, b: EigenVectorHandleRD; c: EigenVectorHandleRD) 
  {.importcpp: "eigenVectorRDMul(@)".}

proc eigenVectorCSMul(a, b: EigenVectorHandleCS; c: EigenVectorHandleCS) 
  {.importcpp: "eigenVectorCSMul(@)".}

proc eigenVectorCDMul(a, b: EigenVectorHandleCD; c: EigenVectorHandleCD) 
  {.importcpp: "eigenVectorCDMul(@)".}

# dot product

proc eigenVectorRSDot(a, b: EigenVectorHandleRS): float32 
  {.importcpp: "eigenVectorRSDot(@)".}

proc eigenVectorRDDot(a, b: EigenVectorHandleRD): float64 
  {.importcpp: "eigenVectorRDDot(@)".}

proc eigenVectorCSDot(a, b: EigenVectorHandleCS, outVal: pointer) 
  {.importcpp: "eigenVectorCSDot(@)".}

proc eigenVectorCDDot(a, b: EigenVectorHandleCD, outVal: pointer) 
  {.importcpp: "eigenVectorCDDot(@)".}

# norm

proc eigenVectorRSNorm(handle: EigenVectorHandleRS): float32 
  {.importcpp: "eigenVectorRSNorm(@)".}

proc eigenVectorRDNorm(handle: EigenVectorHandleRD): float64 
  {.importcpp: "eigenVectorRDNorm(@)".}

proc eigenVectorCSNorm(handle: EigenVectorHandleCS): float32 
  {.importcpp: "eigenVectorCSNorm(@)".}

proc eigenVectorCDNorm(handle: EigenVectorHandleCD): float64 
  {.importcpp: "eigenVectorCDNorm(@)".}

# compound assignment

proc eigenVectorRSAddAssign(a, b: EigenVectorHandleRS) 
  {.importcpp: "eigenVectorRSAddAssign(@)".}

proc eigenVectorRDAddAssign(a, b: EigenVectorHandleRD) 
  {.importcpp: "eigenVectorRDAddAssign(@)".}

proc eigenVectorCSAddAssign(a, b: EigenVectorHandleCS) 
  {.importcpp: "eigenVectorCSAddAssign(@)".}

proc eigenVectorCDAddAssign(a, b: EigenVectorHandleCD) 
  {.importcpp: "eigenVectorCDAddAssign(@)".}

proc eigenVectorRSSubAssign(a, b: EigenVectorHandleRS) 
  {.importcpp: "eigenVectorRSSubAssign(@)".}

proc eigenVectorRDSubAssign(a, b: EigenVectorHandleRD) 
  {.importcpp: "eigenVectorRDSubAssign(@)".}

proc eigenVectorCSSubAssign(a, b: EigenVectorHandleCS) 
  {.importcpp: "eigenVectorCSSubAssign(@)".}

proc eigenVectorCDSubAssign(a, b: EigenVectorHandleCD) 
  {.importcpp: "eigenVectorCDSubAssign(@)".}

# scalar multiply

proc eigenVectorRSScalarMul(a: EigenVectorHandleRS, s: float32, c: EigenVectorHandleRS) 
  {.importcpp: "eigenVectorRSScalarMul(@)".}

proc eigenVectorRDScalarMul(a: EigenVectorHandleRD, s: float64, c: EigenVectorHandleRD) 
  {.importcpp: "eigenVectorRDScalarMul(@)".}

proc eigenVectorCSScalarMul(a: EigenVectorHandleCS, s: pointer, c: EigenVectorHandleCS) 
  {.importcpp: "eigenVectorCSScalarMul(@)".}

proc eigenVectorCDScalarMul(a: EigenVectorHandleCD, s: pointer, c: EigenVectorHandleCD) 
  {.importcpp: "eigenVectorCDScalarMul(@)".}

proc eigenVectorRSScalarMulAssign(a: EigenVectorHandleRS, s: float32) 
  {.importcpp: "eigenVectorRSScalarMulAssign(@)".}

proc eigenVectorRDScalarMulAssign(a: EigenVectorHandleRD, s: float64) 
  {.importcpp: "eigenVectorRDScalarMulAssign(@)".}

proc eigenVectorCSScalarMulAssign(a: EigenVectorHandleCS, s: pointer) 
  {.importcpp: "eigenVectorCSScalarMulAssign(@)".}

proc eigenVectorCDScalarMulAssign(a: EigenVectorHandleCD, s: pointer) 
  {.importcpp: "eigenVectorCDScalarMulAssign(@)".}

# conjugate

proc eigenVectorRSConjugate(a: EigenVectorHandleRS, c: EigenVectorHandleRS) 
  {.importcpp: "eigenVectorRSConjugate(@)".}

proc eigenVectorRDConjugate(a: EigenVectorHandleRD, c: EigenVectorHandleRD) 
  {.importcpp: "eigenVectorRDConjugate(@)".}

proc eigenVectorCSConjugate(a: EigenVectorHandleCS, c: EigenVectorHandleCS) 
  {.importcpp: "eigenVectorCSConjugate(@)".}

proc eigenVectorCDConjugate(a: EigenVectorHandleCD, c: EigenVectorHandleCD) 
  {.importcpp: "eigenVectorCDConjugate(@)".}

# negate

proc eigenVectorRSNegate(a: EigenVectorHandleRS, c: EigenVectorHandleRS) 
  {.importcpp: "eigenVectorRSNegate(@)".}

proc eigenVectorRDNegate(a: EigenVectorHandleRD, c: EigenVectorHandleRD) 
  {.importcpp: "eigenVectorRDNegate(@)".}

proc eigenVectorCSNegate(a: EigenVectorHandleCS, c: EigenVectorHandleCS) 
  {.importcpp: "eigenVectorCSNegate(@)".}

proc eigenVectorCDNegate(a: EigenVectorHandleCD, c: EigenVectorHandleCD) 
  {.importcpp: "eigenVectorCDNegate(@)".}

# normalized

proc eigenVectorRSNormalized(a: EigenVectorHandleRS, c: EigenVectorHandleRS) 
  {.importcpp: "eigenVectorRSNormalized(@)".}

proc eigenVectorRDNormalized(a: EigenVectorHandleRD, c: EigenVectorHandleRD) 
  {.importcpp: "eigenVectorRDNormalized(@)".}

proc eigenVectorCSNormalized(a: EigenVectorHandleCS, c: EigenVectorHandleCS) 
  {.importcpp: "eigenVectorCSNormalized(@)".}

proc eigenVectorCDNormalized(a: EigenVectorHandleCD, c: EigenVectorHandleCD) 
  {.importcpp: "eigenVectorCDNormalized(@)".}

# normalize (in-place)

proc eigenVectorRSNormalize(a: EigenVectorHandleRS) 
  {.importcpp: "eigenVectorRSNormalize(@)".}

proc eigenVectorRDNormalize(a: EigenVectorHandleRD) 
  {.importcpp: "eigenVectorRDNormalize(@)".}

proc eigenVectorCSNormalize(a: EigenVectorHandleCS) 
  {.importcpp: "eigenVectorCSNormalize(@)".}

proc eigenVectorCDNormalize(a: EigenVectorHandleCD) 
  {.importcpp: "eigenVectorCDNormalize(@)".}

# squaredNorm

proc eigenVectorRSSquaredNorm(handle: EigenVectorHandleRS): float32 
  {.importcpp: "eigenVectorRSSquaredNorm(@)".}

proc eigenVectorRDSquaredNorm(handle: EigenVectorHandleRD): float64 
  {.importcpp: "eigenVectorRDSquaredNorm(@)".}

proc eigenVectorCSSquaredNorm(handle: EigenVectorHandleCS): float32 
  {.importcpp: "eigenVectorCSSquaredNorm(@)".}

proc eigenVectorCDSquaredNorm(handle: EigenVectorHandleCD): float64 
  {.importcpp: "eigenVectorCDSquaredNorm(@)".}

# cross product (3D only)

proc eigenVectorRSCross(a, b: EigenVectorHandleRS; c: EigenVectorHandleRS) 
  {.importcpp: "eigenVectorRSCross(@)".}

proc eigenVectorRDCross(a, b: EigenVectorHandleRD; c: EigenVectorHandleRD) 
  {.importcpp: "eigenVectorRDCross(@)".}

proc eigenVectorCSCross(a, b: EigenVectorHandleCS; c: EigenVectorHandleCS) 
  {.importcpp: "eigenVectorCSCross(@)".}

proc eigenVectorCDCross(a, b: EigenVectorHandleCD; c: EigenVectorHandleCD) 
  {.importcpp: "eigenVectorCDCross(@)".}

#[ EigenVector implementation ]#

impl EigenVector:
  #[ constructor/destructor ]#

  method init(rawData: ptr T; numSize, numStride: int) =
    this.size = numSize
    this.stride = numStride
    this.ownsData = false

    when isReal32(T):
      this.data = createEigenVectorRS(rawData, numSize, numStride)
    elif isReal64(T):
      this.data = createEigenVectorRD(rawData, numSize, numStride)
    elif isComplex32(T):
      this.data = createEigenVectorCS(cast[pointer](rawData), numSize, numStride)
    elif isComplex64(T):
      this.data = createEigenVectorCD(cast[pointer](rawData), numSize, numStride)

  method init(numSize, numStride: int) =
    this.size = numSize
    this.stride = numStride
    this.ownsData = true

    when isReal32(T):
      this.data = createTempEigenVectorRS(numSize, numStride)
    elif isReal64(T):
      this.data = createTempEigenVectorRD(numSize, numStride)
    elif isComplex32(T):
      this.data = createTempEigenVectorCS(numSize, numStride)
    elif isComplex64(T):
      this.data = createTempEigenVectorCD(numSize, numStride)
  
  method deinit() =
    when isReal32(T): destroyEigenVectorRS(this.data, this.ownsData)
    elif isReal64(T): destroyEigenVectorRD(this.data, this.ownsData)
    elif isComplex32(T): destroyEigenVectorCS(this.data, this.ownsData)
    elif isComplex64(T): destroyEigenVectorCD(this.data, this.ownsData)
  
  #[ accessors ]#

  method `[]`(idx: int): T =
    when isReal32(T):
      return eigenVectorRSGet(this.data, idx)
    elif isReal64(T):
      return eigenVectorRDGet(this.data, idx)
    elif isComplex32(T):
      var res: Complex32
      eigenVectorCSGet(this.data, idx, addr res)
      return res
    elif isComplex64(T):
      var res: Complex64
      eigenVectorCDGet(this.data, idx, addr res)
      return res
  
  method `[]=`(idx: int, value: T) =
    when isReal32(T):
      eigenVectorRSSet(this.data, idx, value)
    elif isReal64(T):
      eigenVectorRDSet(this.data, idx, value)
    elif isComplex32(T):
      var v = value
      eigenVectorCSSet(this.data, idx, addr v)
    elif isComplex64(T):
      var v = value
      eigenVectorCDSet(this.data, idx, addr v)
  
  #[ algebra ]#

  method `+`(other: EigenVector[T]): EigenVector[T] =
    assert this.size == other.size, "size mismatch"
    result = newEigenVector[T](this.size, this.stride)
    when isReal32(T): eigenVectorRSAdd(this.data, other.data, result.data)
    elif isReal64(T): eigenVectorRDAdd(this.data, other.data, result.data)
    elif isComplex32(T): eigenVectorCSAdd(this.data, other.data, result.data)
    elif isComplex64(T): eigenVectorCDAdd(this.data, other.data, result.data)
  
  method `-`(other: EigenVector[T]): EigenVector[T] =
    assert this.size == other.size, "size mismatch"
    result = newEigenVector[T](this.size, this.stride)
    when isReal32(T): eigenVectorRSSub(this.data, other.data, result.data)
    elif isReal64(T): eigenVectorRDSub(this.data, other.data, result.data)
    elif isComplex32(T): eigenVectorCSSub(this.data, other.data, result.data)
    elif isComplex64(T): eigenVectorCDSub(this.data, other.data, result.data)
  
  method `*`(other: EigenVector[T]): EigenVector[T] =
    assert this.size == other.size, "size mismatch"
    result = newEigenVector[T](this.size, this.stride)
    when isReal32(T): eigenVectorRSMul(this.data, other.data, result.data)
    elif isReal64(T): eigenVectorRDMul(this.data, other.data, result.data)
    elif isComplex32(T): eigenVectorCSMul(this.data, other.data, result.data)
    elif isComplex64(T): eigenVectorCDMul(this.data, other.data, result.data)
  
  method `*`(scalar: T): EigenVector[T] =
    result = newEigenVector[T](this.size, this.stride)
    when isReal32(T): eigenVectorRSScalarMul(this.data, scalar, result.data)
    elif isReal64(T): eigenVectorRDScalarMul(this.data, scalar, result.data)
    elif isComplex32(T):
      var s = scalar
      eigenVectorCSScalarMul(this.data, addr s, result.data)
    elif isComplex64(T):
      var s = scalar
      eigenVectorCDScalarMul(this.data, addr s, result.data)

  method dot(other: EigenVector[T]): T =
    assert this.size == other.size, "size mismatch"
    when isReal32(T):
      return eigenVectorRSDot(this.data, other.data)
    elif isReal64(T):
      return eigenVectorRDDot(this.data, other.data)
    elif isComplex32(T):
      var res: Complex32
      eigenVectorCSDot(this.data, other.data, addr res)
      return res
    elif isComplex64(T):
      var res: Complex64
      eigenVectorCDDot(this.data, other.data, addr res)
      return res
  
  method norm(): auto =
    when isReal32(T):
      return eigenVectorRSNorm(this.data)
    elif isReal64(T):
      return eigenVectorRDNorm(this.data)
    elif isComplex32(T):
      return eigenVectorCSNorm(this.data)
    elif isComplex64(T):
      return eigenVectorCDNorm(this.data)

  method `+=`(other: EigenVector[T]) =
    assert this.size == other.size, "size mismatch"
    when isReal32(T): eigenVectorRSAddAssign(this.data, other.data)
    elif isReal64(T): eigenVectorRDAddAssign(this.data, other.data)
    elif isComplex32(T): eigenVectorCSAddAssign(this.data, other.data)
    elif isComplex64(T): eigenVectorCDAddAssign(this.data, other.data)
  
  method `-=`(other: EigenVector[T]) =
    assert this.size == other.size, "size mismatch"
    when isReal32(T): eigenVectorRSSubAssign(this.data, other.data)
    elif isReal64(T): eigenVectorRDSubAssign(this.data, other.data)
    elif isComplex32(T): eigenVectorCSSubAssign(this.data, other.data)
    elif isComplex64(T): eigenVectorCDSubAssign(this.data, other.data)
  
  method `*=`(scalar: T) =
    when isReal32(T): eigenVectorRSScalarMulAssign(this.data, scalar)
    elif isReal64(T): eigenVectorRDScalarMulAssign(this.data, scalar)
    elif isComplex32(T):
      var s = scalar
      eigenVectorCSScalarMulAssign(this.data, addr s)
    elif isComplex64(T):
      var s = scalar
      eigenVectorCDScalarMulAssign(this.data, addr s)
  
  #[ unary operations ]#

  method conjugate(): EigenVector[T] =
    result = newEigenVector[T](this.size, this.stride)
    when isReal32(T): eigenVectorRSConjugate(this.data, result.data)
    elif isReal64(T): eigenVectorRDConjugate(this.data, result.data)
    elif isComplex32(T): eigenVectorCSConjugate(this.data, result.data)
    elif isComplex64(T): eigenVectorCDConjugate(this.data, result.data)
  
  method `-`(): EigenVector[T] =
    result = newEigenVector[T](this.size, this.stride)
    when isReal32(T): eigenVectorRSNegate(this.data, result.data)
    elif isReal64(T): eigenVectorRDNegate(this.data, result.data)
    elif isComplex32(T): eigenVectorCSNegate(this.data, result.data)
    elif isComplex64(T): eigenVectorCDNegate(this.data, result.data)
  
  method normalized(): EigenVector[T] =
    result = newEigenVector[T](this.size, this.stride)
    when isReal32(T): eigenVectorRSNormalized(this.data, result.data)
    elif isReal64(T): eigenVectorRDNormalized(this.data, result.data)
    elif isComplex32(T): eigenVectorCSNormalized(this.data, result.data)
    elif isComplex64(T): eigenVectorCDNormalized(this.data, result.data)
  
  method normalize() =
    when isReal32(T): eigenVectorRSNormalize(this.data)
    elif isReal64(T): eigenVectorRDNormalize(this.data)
    elif isComplex32(T): eigenVectorCSNormalize(this.data)
    elif isComplex64(T): eigenVectorCDNormalize(this.data)
  
  method squaredNorm(): auto =
    when isReal32(T):
      return eigenVectorRSSquaredNorm(this.data)
    elif isReal64(T):
      return eigenVectorRDSquaredNorm(this.data)
    elif isComplex32(T):
      return eigenVectorCSSquaredNorm(this.data)
    elif isComplex64(T):
      return eigenVectorCDSquaredNorm(this.data)
  
  method cross(other: EigenVector[T]): EigenVector[T] =
    assert this.size == 3 and other.size == 3, "cross product requires 3D vectors"
    result = newEigenVector[T](3, this.stride)
    when isReal32(T): eigenVectorRSCross(this.data, other.data, result.data)
    elif isReal64(T): eigenVectorRDCross(this.data, other.data, result.data)
    elif isComplex32(T): eigenVectorCSCross(this.data, other.data, result.data)
    elif isComplex64(T): eigenVectorCDCross(this.data, other.data, result.data)

when isMainModule:
  import std/[unittest, math]

  suite "EigenVector tests":
    test "real creation, access, and write-through":
      var vdata = [1.0, 2.0, 3.0, 4.0]
      var v = newEigenVector(addr vdata[0], 4, 1)
      check v.ownsData == false

      for i in 0..<4: check v[i] == float64(i + 1)

      # write through vector → underlying buffer
      v[2] = 99.0
      check vdata[2] == 99.0

      # write through buffer → visible via vector
      vdata[0] = 42.0
      check v[0] == 42.0

    test "complex creation and access":
      var cdata = [complex(1.0, 2.0), complex(3.0, 4.0), complex(5.0, 6.0)]
      var v = newEigenVector(addr cdata[0], 3, 1)

      check v[0] == complex(1.0, 2.0)
      check v[1] == complex(3.0, 4.0)
      check v[2] == complex(5.0, 6.0)

      v[1] = complex(77.0, 88.0)
      check cdata[1].re == 77.0
      check cdata[1].im == 88.0

    test "temporary vector creation":
      var tv = newEigenVector[float64](5, 1)
      check tv.ownsData == true

      for i in 0..<5: check tv[i] == 0.0

      tv[3] = 42.0
      check tv[3] == 42.0

    test "vector addition and subtraction":
      var adata = [1.0, 2.0, 3.0]
      var bdata = [4.0, 5.0, 6.0]
      var a = newEigenVector(addr adata[0], 3, 1)
      var b = newEigenVector(addr bdata[0], 3, 1)

      var s = a + b
      check s[0] == 5.0
      check s[1] == 7.0
      check s[2] == 9.0

      var d = b - a
      check d[0] == 3.0
      check d[1] == 3.0
      check d[2] == 3.0

    test "element-wise multiply":
      var adata = [2.0, 3.0, 4.0]
      var bdata = [5.0, 6.0, 7.0]
      var a = newEigenVector(addr adata[0], 3, 1)
      var b = newEigenVector(addr bdata[0], 3, 1)

      var p = a * b
      check p[0] == 10.0
      check p[1] == 18.0
      check p[2] == 28.0

    test "dot product":
      # [1, 2, 3] · [4, 5, 6] = 4 + 10 + 18 = 32
      var adata = [1.0, 2.0, 3.0]
      var bdata = [4.0, 5.0, 6.0]
      var a = newEigenVector(addr adata[0], 3, 1)
      var b = newEigenVector(addr bdata[0], 3, 1)

      check a.dot(b) == 32.0

    test "complex dot product":
      # Eigen dot is conjugate-linear in first arg:
      # dot(a, b) = conj(a) · b
      var ca = [complex(1.0, 1.0), complex(2.0, 0.0)]
      var cb = [complex(1.0, 0.0), complex(0.0, 1.0)]
      var a = newEigenVector(addr ca[0], 2, 1)
      var b = newEigenVector(addr cb[0], 2, 1)

      # conj(1+i)(1+0i) + conj(2+0i)(0+i) = (1-i)(1) + (2)(i) = 1-i+2i = 1+i
      var d = a.dot(b)
      check d.re == 1.0
      check d.im == 1.0

    test "norm":
      # ||[3, 4]|| = 5
      var vdata = [3.0, 4.0]
      var v = newEigenVector(addr vdata[0], 2, 1)
      check abs(v.norm() - 5.0) < 1e-12

    test "scalar multiply":
      var vdata = [1.0, 2.0, 3.0]
      var v = newEigenVector(addr vdata[0], 3, 1)
      var s = v * 3.0
      check s[0] == 3.0
      check s[1] == 6.0
      check s[2] == 9.0

    test "compound assignment (+=, -=, *=)":
      var adata = [1.0, 2.0, 3.0]
      var bdata = [4.0, 5.0, 6.0]
      var a = newEigenVector(addr adata[0], 3, 1)
      var b = newEigenVector(addr bdata[0], 3, 1)

      a += b
      check a[0] == 5.0
      check a[1] == 7.0
      check a[2] == 9.0
      check adata[0] == 5.0  # write-through

      a -= b
      check a[0] == 1.0
      check a[1] == 2.0
      check a[2] == 3.0

      a *= 10.0
      check a[0] == 10.0
      check a[1] == 20.0
      check a[2] == 30.0

    test "strided access":
      # data = [1, _, 2, _, 3] with stride=2 → view [1, 2, 3]
      var vdata = [1.0, 0.0, 2.0, 0.0, 3.0]
      var v = newEigenVector(addr vdata[0], 3, 2)
      check v[0] == 1.0
      check v[1] == 2.0
      check v[2] == 3.0

    test "unary negate":
      var vdata = [1.0, -2.0, 3.0]
      var v = newEigenVector(addr vdata[0], 3, 1)
      var n = -v
      check n[0] == -1.0
      check n[1] == 2.0
      check n[2] == -3.0

    test "conjugate":
      var cdata = [complex(1.0, 2.0), complex(3.0, -4.0), complex(-5.0, 6.0)]
      var v = newEigenVector(addr cdata[0], 3, 1)
      var c = v.conjugate()
      check c[0] == complex(1.0, -2.0)
      check c[1] == complex(3.0, 4.0)
      check c[2] == complex(-5.0, -6.0)

    test "normalized (returns unit vector)":
      var vdata = [3.0, 4.0]
      var v = newEigenVector(addr vdata[0], 2, 1)
      var u = v.normalized()
      check abs(u[0] - 0.6) < 1e-12
      check abs(u[1] - 0.8) < 1e-12
      check abs(u.norm() - 1.0) < 1e-12

    test "normalize (in-place)":
      var vdata = [3.0, 4.0]
      var v = newEigenVector(addr vdata[0], 2, 1)
      v.normalize()
      check abs(v[0] - 0.6) < 1e-12
      check abs(v[1] - 0.8) < 1e-12
      check abs(vdata[0] - 0.6) < 1e-12  # write-through

    test "squaredNorm":
      # ||[3, 4]||^2 = 9 + 16 = 25
      var vdata = [3.0, 4.0]
      var v = newEigenVector(addr vdata[0], 2, 1)
      check abs(v.squaredNorm() - 25.0) < 1e-12

    test "cross product (3D)":
      # [1, 0, 0] x [0, 1, 0] = [0, 0, 1]
      var adata = [1.0, 0.0, 0.0]
      var bdata = [0.0, 1.0, 0.0]
      var a = newEigenVector(addr adata[0], 3, 1)
      var b = newEigenVector(addr bdata[0], 3, 1)
      var c = a.cross(b)
      check c[0] == 0.0
      check c[1] == 0.0
      check c[2] == 1.0

      # [1, 2, 3] x [4, 5, 6] = [2*6-3*5, 3*4-1*6, 1*5-2*4] = [-3, 6, -3]
      var a2data = [1.0, 2.0, 3.0]
      var b2data = [4.0, 5.0, 6.0]
      var a2 = newEigenVector(addr a2data[0], 3, 1)
      var b2 = newEigenVector(addr b2data[0], 3, 1)
      var c2 = a2.cross(b2)
      check c2[0] == -3.0
      check c2[1] == 6.0
      check c2[2] == -3.0
