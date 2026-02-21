/** 
 * ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
 * Source file: src/eigen/eigenscalar.h
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

/* scalar handle — a pointer to a single value, optionally owned */

struct EigenScalarHandleRS { float* val; };
struct EigenScalarHandleRD { double* val; };
struct EigenScalarHandleCS { std::complex<float>* val; };
struct EigenScalarHandleCD { std::complex<double>* val; };

/* constructors — map over existing data */

inline EigenScalarHandleRS createEigenScalarRS(float* data) {
  return EigenScalarHandleRS{data};
}

inline EigenScalarHandleRD createEigenScalarRD(double* data) {
  return EigenScalarHandleRD{data};
}

inline EigenScalarHandleCS createEigenScalarCS(void* data) {
  return EigenScalarHandleCS{static_cast<std::complex<float>*>(data)};
}

inline EigenScalarHandleCD createEigenScalarCD(void* data) {
  return EigenScalarHandleCD{static_cast<std::complex<double>*>(data)};
}

/* temporary constructors — allocate owned data */

inline EigenScalarHandleRS createTempEigenScalarRS() {
  auto* data = new float(0);
  return EigenScalarHandleRS{data};
}

inline EigenScalarHandleRD createTempEigenScalarRD() {
  auto* data = new double(0);
  return EigenScalarHandleRD{data};
}

inline EigenScalarHandleCS createTempEigenScalarCS() {
  auto* data = new std::complex<float>(0, 0);
  return EigenScalarHandleCS{data};
}

inline EigenScalarHandleCD createTempEigenScalarCD() {
  auto* data = new std::complex<double>(0, 0);
  return EigenScalarHandleCD{data};
}

/* destructors */

inline void destroyEigenScalarRS(EigenScalarHandleRS handle, bool ownsData) {
  if (!handle.val) return;
  if (ownsData) delete handle.val;
}
inline void destroyEigenScalarRD(EigenScalarHandleRD handle, bool ownsData) {
  if (!handle.val) return;
  if (ownsData) delete handle.val;
}
inline void destroyEigenScalarCS(EigenScalarHandleCS handle, bool ownsData) {
  if (!handle.val) return;
  if (ownsData) delete handle.val;
}
inline void destroyEigenScalarCD(EigenScalarHandleCD handle, bool ownsData) {
  if (!handle.val) return;
  if (ownsData) delete handle.val;
}

/* accessors */

inline float eigenScalarRSGet(const EigenScalarHandleRS handle) {
  return *handle.val;
}

inline double eigenScalarRDGet(const EigenScalarHandleRD handle) {
  return *handle.val;
}

inline void eigenScalarCSGet(const EigenScalarHandleCS handle, void* out) {
  *static_cast<std::complex<float>*>(out) = *handle.val;
}

inline void eigenScalarCDGet(const EigenScalarHandleCD handle, void* out) {
  *static_cast<std::complex<double>*>(out) = *handle.val;
}

inline void eigenScalarRSSet(EigenScalarHandleRS handle, float value) {
  *handle.val = value;
}

inline void eigenScalarRDSet(EigenScalarHandleRD handle, double value) {
  *handle.val = value;
}

inline void eigenScalarCSSet(EigenScalarHandleCS handle, const void* value) {
  *handle.val = *static_cast<const std::complex<float>*>(value);
}

inline void eigenScalarCDSet(EigenScalarHandleCD handle, const void* value) {
  *handle.val = *static_cast<const std::complex<double>*>(value);
}

/* algebra */

inline void eigenScalarRSAdd(
  const EigenScalarHandleRS a, const EigenScalarHandleRS b, EigenScalarHandleRS out
) { *out.val = *a.val + *b.val; }

inline void eigenScalarRDAdd(
  const EigenScalarHandleRD a, const EigenScalarHandleRD b, EigenScalarHandleRD out
) { *out.val = *a.val + *b.val; }

inline void eigenScalarCSAdd(
  const EigenScalarHandleCS a, const EigenScalarHandleCS b, EigenScalarHandleCS out
) { *out.val = *a.val + *b.val; }

inline void eigenScalarCDAdd(
  const EigenScalarHandleCD a, const EigenScalarHandleCD b, EigenScalarHandleCD out
) { *out.val = *a.val + *b.val; }

inline void eigenScalarRSSub(
  const EigenScalarHandleRS a, const EigenScalarHandleRS b, EigenScalarHandleRS out
) { *out.val = *a.val - *b.val; }

inline void eigenScalarRDSub(
  const EigenScalarHandleRD a, const EigenScalarHandleRD b, EigenScalarHandleRD out
) { *out.val = *a.val - *b.val; }

inline void eigenScalarCSSub(
  const EigenScalarHandleCS a, const EigenScalarHandleCS b, EigenScalarHandleCS out
) { *out.val = *a.val - *b.val; }

inline void eigenScalarCDSub(
  const EigenScalarHandleCD a, const EigenScalarHandleCD b, EigenScalarHandleCD out
) { *out.val = *a.val - *b.val; }

inline void eigenScalarRSMul(
  const EigenScalarHandleRS a, const EigenScalarHandleRS b, EigenScalarHandleRS out
) { *out.val = *a.val * *b.val; }

inline void eigenScalarRDMul(
  const EigenScalarHandleRD a, const EigenScalarHandleRD b, EigenScalarHandleRD out
) { *out.val = *a.val * *b.val; }

inline void eigenScalarCSMul(
  const EigenScalarHandleCS a, const EigenScalarHandleCS b, EigenScalarHandleCS out
) { *out.val = *a.val * *b.val; }

inline void eigenScalarCDMul(
  const EigenScalarHandleCD a, const EigenScalarHandleCD b, EigenScalarHandleCD out
) { *out.val = *a.val * *b.val; }

inline void eigenScalarRSDiv(
  const EigenScalarHandleRS a, const EigenScalarHandleRS b, EigenScalarHandleRS out
) { *out.val = *a.val / *b.val; }

inline void eigenScalarRDDiv(
  const EigenScalarHandleRD a, const EigenScalarHandleRD b, EigenScalarHandleRD out
) { *out.val = *a.val / *b.val; }

inline void eigenScalarCSDiv(
  const EigenScalarHandleCS a, const EigenScalarHandleCS b, EigenScalarHandleCS out
) { *out.val = *a.val / *b.val; }

inline void eigenScalarCDDiv(
  const EigenScalarHandleCD a, const EigenScalarHandleCD b, EigenScalarHandleCD out
) { *out.val = *a.val / *b.val; }

/* compound assignment */

inline void eigenScalarRSAddAssign(EigenScalarHandleRS a, const EigenScalarHandleRS b) {
  *a.val += *b.val;
}
inline void eigenScalarRDAddAssign(EigenScalarHandleRD a, const EigenScalarHandleRD b) {
  *a.val += *b.val;
}
inline void eigenScalarCSAddAssign(EigenScalarHandleCS a, const EigenScalarHandleCS b) {
  *a.val += *b.val;
}
inline void eigenScalarCDAddAssign(EigenScalarHandleCD a, const EigenScalarHandleCD b) {
  *a.val += *b.val;
}

inline void eigenScalarRSSubAssign(EigenScalarHandleRS a, const EigenScalarHandleRS b) {
  *a.val -= *b.val;
}
inline void eigenScalarRDSubAssign(EigenScalarHandleRD a, const EigenScalarHandleRD b) {
  *a.val -= *b.val;
}
inline void eigenScalarCSSubAssign(EigenScalarHandleCS a, const EigenScalarHandleCS b) {
  *a.val -= *b.val;
}
inline void eigenScalarCDSubAssign(EigenScalarHandleCD a, const EigenScalarHandleCD b) {
  *a.val -= *b.val;
}

inline void eigenScalarRSMulAssign(EigenScalarHandleRS a, const EigenScalarHandleRS b) {
  *a.val *= *b.val;
}
inline void eigenScalarRDMulAssign(EigenScalarHandleRD a, const EigenScalarHandleRD b) {
  *a.val *= *b.val;
}
inline void eigenScalarCSMulAssign(EigenScalarHandleCS a, const EigenScalarHandleCS b) {
  *a.val *= *b.val;
}
inline void eigenScalarCDMulAssign(EigenScalarHandleCD a, const EigenScalarHandleCD b) {
  *a.val *= *b.val;
}

inline void eigenScalarRSDivAssign(EigenScalarHandleRS a, const EigenScalarHandleRS b) {
  *a.val /= *b.val;
}
inline void eigenScalarRDDivAssign(EigenScalarHandleRD a, const EigenScalarHandleRD b) {
  *a.val /= *b.val;
}
inline void eigenScalarCSDivAssign(EigenScalarHandleCS a, const EigenScalarHandleCS b) {
  *a.val /= *b.val;
}
inline void eigenScalarCDDivAssign(EigenScalarHandleCD a, const EigenScalarHandleCD b) {
  *a.val /= *b.val;
}

/* abs / conj */

inline float eigenScalarRSAbs(const EigenScalarHandleRS handle) {
  return std::abs(*handle.val);
}
inline double eigenScalarRDAbs(const EigenScalarHandleRD handle) {
  return std::abs(*handle.val);
}
inline float eigenScalarCSAbs(const EigenScalarHandleCS handle) {
  return std::abs(*handle.val);
}
inline double eigenScalarCDAbs(const EigenScalarHandleCD handle) {
  return std::abs(*handle.val);
}

inline void eigenScalarCSConj(const EigenScalarHandleCS handle, void* out) {
  *static_cast<std::complex<float>*>(out) = std::conj(*handle.val);
}
inline void eigenScalarCDConj(const EigenScalarHandleCD handle, void* out) {
  *static_cast<std::complex<double>*>(out) = std::conj(*handle.val);
}