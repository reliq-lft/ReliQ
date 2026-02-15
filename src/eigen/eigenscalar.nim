#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/eigen/eigenscalar.nim
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
  if (ownsData) delete handle.val;
}
inline void destroyEigenScalarRD(EigenScalarHandleRD handle, bool ownsData) {
  if (ownsData) delete handle.val;
}
inline void destroyEigenScalarCS(EigenScalarHandleCS handle, bool ownsData) {
  if (ownsData) delete handle.val;
}
inline void destroyEigenScalarCD(EigenScalarHandleCD handle, bool ownsData) {
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
""".}

type
  EigenScalarHandleRS* {.importcpp: "EigenScalarHandleRS".} = object
  EigenScalarHandleRD* {.importcpp: "EigenScalarHandleRD".} = object
  EigenScalarHandleCS* {.importcpp: "EigenScalarHandleCS".} = object
  EigenScalarHandleCD* {.importcpp: "EigenScalarHandleCD".} = object

record EigenScalar*[T]:
  var ownsData: bool
  when isReal32(T):
    var data*: EigenScalarHandleRS
  elif isReal64(T):
    var data*: EigenScalarHandleRD
  elif isComplex32(T):
    var data*: EigenScalarHandleCS
  elif isComplex64(T):
    var data*: EigenScalarHandleCD

#[ eigen wrapper ]#

# constructors

proc createEigenScalarRS(data: ptr float32): EigenScalarHandleRS 
  {.importcpp: "createEigenScalarRS(@)".}

proc createEigenScalarRD(data: ptr float64): EigenScalarHandleRD 
  {.importcpp: "createEigenScalarRD(@)".}

proc createEigenScalarCS(data: pointer): EigenScalarHandleCS 
  {.importcpp: "createEigenScalarCS(@)".}

proc createEigenScalarCD(data: pointer): EigenScalarHandleCD 
  {.importcpp: "createEigenScalarCD(@)".}

# temp constructors

proc createTempEigenScalarRS(): EigenScalarHandleRS 
  {.importcpp: "createTempEigenScalarRS(@)".}

proc createTempEigenScalarRD(): EigenScalarHandleRD 
  {.importcpp: "createTempEigenScalarRD(@)".}

proc createTempEigenScalarCS(): EigenScalarHandleCS 
  {.importcpp: "createTempEigenScalarCS(@)".}

proc createTempEigenScalarCD(): EigenScalarHandleCD 
  {.importcpp: "createTempEigenScalarCD(@)".}

# destructors

proc destroyEigenScalarRS(handle: EigenScalarHandleRS, ownsData: bool) 
  {.importcpp: "destroyEigenScalarRS(@)".}

proc destroyEigenScalarRD(handle: EigenScalarHandleRD, ownsData: bool) 
  {.importcpp: "destroyEigenScalarRD(@)".}

proc destroyEigenScalarCS(handle: EigenScalarHandleCS, ownsData: bool) 
  {.importcpp: "destroyEigenScalarCS(@)".}

proc destroyEigenScalarCD(handle: EigenScalarHandleCD, ownsData: bool) 
  {.importcpp: "destroyEigenScalarCD(@)".}

# accessors

proc eigenScalarRSGet(handle: EigenScalarHandleRS): float32 
  {.importcpp: "eigenScalarRSGet(@)".}

proc eigenScalarRDGet(handle: EigenScalarHandleRD): float64 
  {.importcpp: "eigenScalarRDGet(@)".}

proc eigenScalarCSGet(handle: EigenScalarHandleCS, outVal: pointer) 
  {.importcpp: "eigenScalarCSGet(@)".}

proc eigenScalarCDGet(handle: EigenScalarHandleCD, outVal: pointer) 
  {.importcpp: "eigenScalarCDGet(@)".}

proc eigenScalarRSSet(handle: EigenScalarHandleRS, value: float32) 
  {.importcpp: "eigenScalarRSSet(@)".}

proc eigenScalarRDSet(handle: EigenScalarHandleRD, value: float64) 
  {.importcpp: "eigenScalarRDSet(@)".}

proc eigenScalarCSSet(handle: EigenScalarHandleCS, value: pointer) 
  {.importcpp: "eigenScalarCSSet(@)".}

proc eigenScalarCDSet(handle: EigenScalarHandleCD, value: pointer) 
  {.importcpp: "eigenScalarCDSet(@)".}

# algebra

proc eigenScalarRSAdd(a, b: EigenScalarHandleRS; c: EigenScalarHandleRS) 
  {.importcpp: "eigenScalarRSAdd(@)".}

proc eigenScalarRDAdd(a, b: EigenScalarHandleRD; c: EigenScalarHandleRD) 
  {.importcpp: "eigenScalarRDAdd(@)".}

proc eigenScalarCSAdd(a, b: EigenScalarHandleCS; c: EigenScalarHandleCS) 
  {.importcpp: "eigenScalarCSAdd(@)".}

proc eigenScalarCDAdd(a, b: EigenScalarHandleCD; c: EigenScalarHandleCD) 
  {.importcpp: "eigenScalarCDAdd(@)".}

proc eigenScalarRSSub(a, b: EigenScalarHandleRS; c: EigenScalarHandleRS) 
  {.importcpp: "eigenScalarRSSub(@)".}

proc eigenScalarRDSub(a, b: EigenScalarHandleRD; c: EigenScalarHandleRD) 
  {.importcpp: "eigenScalarRDSub(@)".}

proc eigenScalarCSSub(a, b: EigenScalarHandleCS; c: EigenScalarHandleCS) 
  {.importcpp: "eigenScalarCSSub(@)".}

proc eigenScalarCDSub(a, b: EigenScalarHandleCD; c: EigenScalarHandleCD) 
  {.importcpp: "eigenScalarCDSub(@)".}

proc eigenScalarRSMul(a, b: EigenScalarHandleRS; c: EigenScalarHandleRS) 
  {.importcpp: "eigenScalarRSMul(@)".}

proc eigenScalarRDMul(a, b: EigenScalarHandleRD; c: EigenScalarHandleRD) 
  {.importcpp: "eigenScalarRDMul(@)".}

proc eigenScalarCSMul(a, b: EigenScalarHandleCS; c: EigenScalarHandleCS) 
  {.importcpp: "eigenScalarCSMul(@)".}

proc eigenScalarCDMul(a, b: EigenScalarHandleCD; c: EigenScalarHandleCD) 
  {.importcpp: "eigenScalarCDMul(@)".}

proc eigenScalarRSDiv(a, b: EigenScalarHandleRS; c: EigenScalarHandleRS) 
  {.importcpp: "eigenScalarRSDiv(@)".}

proc eigenScalarRDDiv(a, b: EigenScalarHandleRD; c: EigenScalarHandleRD) 
  {.importcpp: "eigenScalarRDDiv(@)".}

proc eigenScalarCSDiv(a, b: EigenScalarHandleCS; c: EigenScalarHandleCS) 
  {.importcpp: "eigenScalarCSDiv(@)".}

proc eigenScalarCDDiv(a, b: EigenScalarHandleCD; c: EigenScalarHandleCD) 
  {.importcpp: "eigenScalarCDDiv(@)".}

# compound assignment

proc eigenScalarRSAddAssign(a, b: EigenScalarHandleRS) 
  {.importcpp: "eigenScalarRSAddAssign(@)".}

proc eigenScalarRDAddAssign(a, b: EigenScalarHandleRD) 
  {.importcpp: "eigenScalarRDAddAssign(@)".}

proc eigenScalarCSAddAssign(a, b: EigenScalarHandleCS) 
  {.importcpp: "eigenScalarCSAddAssign(@)".}

proc eigenScalarCDAddAssign(a, b: EigenScalarHandleCD) 
  {.importcpp: "eigenScalarCDAddAssign(@)".}

proc eigenScalarRSSubAssign(a, b: EigenScalarHandleRS) 
  {.importcpp: "eigenScalarRSSubAssign(@)".}

proc eigenScalarRDSubAssign(a, b: EigenScalarHandleRD) 
  {.importcpp: "eigenScalarRDSubAssign(@)".}

proc eigenScalarCSSubAssign(a, b: EigenScalarHandleCS) 
  {.importcpp: "eigenScalarCSSubAssign(@)".}

proc eigenScalarCDSubAssign(a, b: EigenScalarHandleCD) 
  {.importcpp: "eigenScalarCDSubAssign(@)".}

proc eigenScalarRSMulAssign(a, b: EigenScalarHandleRS) 
  {.importcpp: "eigenScalarRSMulAssign(@)".}

proc eigenScalarRDMulAssign(a, b: EigenScalarHandleRD) 
  {.importcpp: "eigenScalarRDMulAssign(@)".}

proc eigenScalarCSMulAssign(a, b: EigenScalarHandleCS) 
  {.importcpp: "eigenScalarCSMulAssign(@)".}

proc eigenScalarCDMulAssign(a, b: EigenScalarHandleCD) 
  {.importcpp: "eigenScalarCDMulAssign(@)".}

proc eigenScalarRSDivAssign(a, b: EigenScalarHandleRS) 
  {.importcpp: "eigenScalarRSDivAssign(@)".}

proc eigenScalarRDDivAssign(a, b: EigenScalarHandleRD) 
  {.importcpp: "eigenScalarRDDivAssign(@)".}

proc eigenScalarCSDivAssign(a, b: EigenScalarHandleCS) 
  {.importcpp: "eigenScalarCSDivAssign(@)".}

proc eigenScalarCDDivAssign(a, b: EigenScalarHandleCD) 
  {.importcpp: "eigenScalarCDDivAssign(@)".}

# abs / conj

proc eigenScalarRSAbs(handle: EigenScalarHandleRS): float32 
  {.importcpp: "eigenScalarRSAbs(@)".}

proc eigenScalarRDAbs(handle: EigenScalarHandleRD): float64 
  {.importcpp: "eigenScalarRDAbs(@)".}

proc eigenScalarCSAbs(handle: EigenScalarHandleCS): float32 
  {.importcpp: "eigenScalarCSAbs(@)".}

proc eigenScalarCDAbs(handle: EigenScalarHandleCD): float64 
  {.importcpp: "eigenScalarCDAbs(@)".}

proc eigenScalarCSConj(handle: EigenScalarHandleCS, outVal: pointer) 
  {.importcpp: "eigenScalarCSConj(@)".}

proc eigenScalarCDConj(handle: EigenScalarHandleCD, outVal: pointer) 
  {.importcpp: "eigenScalarCDConj(@)".}

#[ EigenScalar implementation ]#

impl EigenScalar:
  #[ constructor/destructor ]#

  method init(rawData: ptr T) =
    this.ownsData = false
    when isReal32(T):
      this.data = createEigenScalarRS(rawData)
    elif isReal64(T):
      this.data = createEigenScalarRD(rawData)
    elif isComplex32(T):
      this.data = createEigenScalarCS(cast[pointer](rawData))
    elif isComplex64(T):
      this.data = createEigenScalarCD(cast[pointer](rawData))

  method init(value: T) =
    this.ownsData = true
    when isReal32(T):
      this.data = createTempEigenScalarRS()
      eigenScalarRSSet(this.data, value)
    elif isReal64(T):
      this.data = createTempEigenScalarRD()
      eigenScalarRDSet(this.data, value)
    elif isComplex32(T):
      this.data = createTempEigenScalarCS()
      var v = value
      eigenScalarCSSet(this.data, addr v)
    elif isComplex64(T):
      this.data = createTempEigenScalarCD()
      var v = value
      eigenScalarCDSet(this.data, addr v)
  
  method deinit() =
    when isReal32(T): destroyEigenScalarRS(this.data, this.ownsData)
    elif isReal64(T): destroyEigenScalarRD(this.data, this.ownsData)
    elif isComplex32(T): destroyEigenScalarCS(this.data, this.ownsData)
    elif isComplex64(T): destroyEigenScalarCD(this.data, this.ownsData)
  
  #[ accessors ]#

  method get(): T =
    when isReal32(T):
      return eigenScalarRSGet(this.data)
    elif isReal64(T):
      return eigenScalarRDGet(this.data)
    elif isComplex32(T):
      var res: Complex32
      eigenScalarCSGet(this.data, addr res)
      return res
    elif isComplex64(T):
      var res: Complex64
      eigenScalarCDGet(this.data, addr res)
      return res
  
  method set(value: T) =
    when isReal32(T):
      eigenScalarRSSet(this.data, value)
    elif isReal64(T):
      eigenScalarRDSet(this.data, value)
    elif isComplex32(T):
      var v = value
      eigenScalarCSSet(this.data, addr v)
    elif isComplex64(T):
      var v = value
      eigenScalarCDSet(this.data, addr v)
  
  #[ algebra ]#

  method `+`(other: EigenScalar[T]): EigenScalar[T] =
    result = newEigenScalar[T](this.get())
    when isReal32(T): eigenScalarRSAdd(this.data, other.data, result.data)
    elif isReal64(T): eigenScalarRDAdd(this.data, other.data, result.data)
    elif isComplex32(T): eigenScalarCSAdd(this.data, other.data, result.data)
    elif isComplex64(T): eigenScalarCDAdd(this.data, other.data, result.data)
  
  method `-`(other: EigenScalar[T]): EigenScalar[T] =
    result = newEigenScalar[T](this.get())
    when isReal32(T): eigenScalarRSSub(this.data, other.data, result.data)
    elif isReal64(T): eigenScalarRDSub(this.data, other.data, result.data)
    elif isComplex32(T): eigenScalarCSSub(this.data, other.data, result.data)
    elif isComplex64(T): eigenScalarCDSub(this.data, other.data, result.data)
  
  method `*`(other: EigenScalar[T]): EigenScalar[T] =
    result = newEigenScalar[T](this.get())
    when isReal32(T): eigenScalarRSMul(this.data, other.data, result.data)
    elif isReal64(T): eigenScalarRDMul(this.data, other.data, result.data)
    elif isComplex32(T): eigenScalarCSMul(this.data, other.data, result.data)
    elif isComplex64(T): eigenScalarCDMul(this.data, other.data, result.data)
  
  method `/`(other: EigenScalar[T]): EigenScalar[T] =
    result = newEigenScalar[T](this.get())
    when isReal32(T): eigenScalarRSDiv(this.data, other.data, result.data)
    elif isReal64(T): eigenScalarRDDiv(this.data, other.data, result.data)
    elif isComplex32(T): eigenScalarCSDiv(this.data, other.data, result.data)
    elif isComplex64(T): eigenScalarCDDiv(this.data, other.data, result.data)

  method `+=`(other: EigenScalar[T]) =
    when isReal32(T): eigenScalarRSAddAssign(this.data, other.data)
    elif isReal64(T): eigenScalarRDAddAssign(this.data, other.data)
    elif isComplex32(T): eigenScalarCSAddAssign(this.data, other.data)
    elif isComplex64(T): eigenScalarCDAddAssign(this.data, other.data)
  
  method `-=`(other: EigenScalar[T]) =
    when isReal32(T): eigenScalarRSSubAssign(this.data, other.data)
    elif isReal64(T): eigenScalarRDSubAssign(this.data, other.data)
    elif isComplex32(T): eigenScalarCSSubAssign(this.data, other.data)
    elif isComplex64(T): eigenScalarCDSubAssign(this.data, other.data)
  
  method `*=`(other: EigenScalar[T]) =
    when isReal32(T): eigenScalarRSMulAssign(this.data, other.data)
    elif isReal64(T): eigenScalarRDMulAssign(this.data, other.data)
    elif isComplex32(T): eigenScalarCSMulAssign(this.data, other.data)
    elif isComplex64(T): eigenScalarCDMulAssign(this.data, other.data)
  
  method `/=`(other: EigenScalar[T]) =
    when isReal32(T): eigenScalarRSDivAssign(this.data, other.data)
    elif isReal64(T): eigenScalarRDDivAssign(this.data, other.data)
    elif isComplex32(T): eigenScalarCSDivAssign(this.data, other.data)
    elif isComplex64(T): eigenScalarCDDivAssign(this.data, other.data)

  method abs(): auto =
    when isReal32(T):
      return eigenScalarRSAbs(this.data)
    elif isReal64(T):
      return eigenScalarRDAbs(this.data)
    elif isComplex32(T):
      return eigenScalarCSAbs(this.data)
    elif isComplex64(T):
      return eigenScalarCDAbs(this.data)

  method conj(): EigenScalar[T] =
    when isComplex32(T):
      var res: Complex32
      eigenScalarCSConj(this.data, addr res)
      result = newEigenScalar[T](res)
    elif isComplex64(T):
      var res: Complex64
      eigenScalarCDConj(this.data, addr res)
      result = newEigenScalar[T](res)
    else:
      # conj of a real is itself
      result = newEigenScalar[T](this.get())

when isMainModule:
  import std/[unittest, math]

  suite "EigenScalar tests":
    test "real creation, access, and write-through":
      var val = 42.0
      var s = newEigenScalar(addr val)
      check s.ownsData == false
      check s.get() == 42.0

      # write through scalar → underlying value
      s.set(99.0)
      check val == 99.0

      # write through value → visible via scalar
      val = 7.0
      check s.get() == 7.0

    test "complex creation and access":
      var cval = complex(3.0, 4.0)
      var s = newEigenScalar(addr cval)
      var g = s.get()
      check g.re == 3.0
      check g.im == 4.0

      s.set(complex(10.0, 20.0))
      check cval.re == 10.0
      check cval.im == 20.0

    test "temporary scalar":
      var s = newEigenScalar[float64](0.0)
      check s.ownsData == true
      check s.get() == 0.0

      s.set(123.0)
      check s.get() == 123.0

    test "real arithmetic (+, -, *, /)":
      var aval = 10.0
      var bval = 3.0
      var a = newEigenScalar(addr aval)
      var b = newEigenScalar(addr bval)

      var s = a + b
      check s.get() == 13.0

      var d = a - b
      check d.get() == 7.0

      var p = a * b
      check p.get() == 30.0

      var q = a / b
      check abs(q.get() - 10.0 / 3.0) < 1e-12

    test "real compound assignment (+=, -=, *=, /=)":
      var aval = 10.0
      var bval = 3.0
      var a = newEigenScalar(addr aval)
      var b = newEigenScalar(addr bval)

      a += b
      check a.get() == 13.0
      check aval == 13.0  # write-through

      a -= b
      check a.get() == 10.0

      a *= b
      check a.get() == 30.0

      a /= b
      check abs(a.get() - 10.0) < 1e-12

    test "complex arithmetic":
      var ca = complex(1.0, 2.0)
      var cb = complex(3.0, 4.0)
      var a = newEigenScalar(addr ca)
      var b = newEigenScalar(addr cb)

      var s = a + b
      check s.get() == complex(4.0, 6.0)

      var d = a - b
      check d.get() == complex(-2.0, -2.0)

      # (1+2i)(3+4i) = 3+4i+6i+8i² = -5+10i
      var p = a * b
      check p.get() == complex(-5.0, 10.0)

    test "abs":
      var rval = -5.0
      var rs = newEigenScalar(addr rval)
      check rs.abs() == 5.0

      # |3+4i| = 5
      var cval = complex(3.0, 4.0)
      var cs = newEigenScalar(addr cval)
      check abs(cs.abs() - 5.0) < 1e-12

    test "conj":
      var cval = complex(3.0, 4.0)
      var s = newEigenScalar(addr cval)
      var c = s.conj()
      var g = c.get()
      check g.re == 3.0
      check g.im == -4.0

      # conj of real is itself
      var rval = 7.0
      var rs = newEigenScalar(addr rval)
      var rc = rs.conj()
      check rc.get() == 7.0
