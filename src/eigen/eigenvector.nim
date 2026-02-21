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

template eigenVectorHeader*: untyped =
  {.pragma: vector, header: "eigenvector.h".}

eigenVectorHeader()

type
  EigenVectorHandleRS* {.importcpp: "EigenVectorHandleRS", vector.} = object
  EigenVectorHandleRD* {.importcpp: "EigenVectorHandleRD", vector.} = object
  EigenVectorHandleCS* {.importcpp: "EigenVectorHandleCS", vector.} = object
  EigenVectorHandleCD* {.importcpp: "EigenVectorHandleCD", vector.} = object

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
): EigenVectorHandleRS {.importcpp: "createEigenVectorRS(@)", vector.}

proc createEigenVectorRD(
  data: ptr float64; size, stride: int
): EigenVectorHandleRD {.importcpp: "createEigenVectorRD(@)", vector.}

proc createEigenVectorCS(
  data: pointer; size, stride: int
): EigenVectorHandleCS {.importcpp: "createEigenVectorCS(@)", vector.}
proc createEigenVectorCD(
  data: pointer; size, stride: int
): EigenVectorHandleCD {.importcpp: "createEigenVectorCD(@)", vector.}

# temp constructors

proc createTempEigenVectorRS(
  size, stride: int
): EigenVectorHandleRS {.importcpp: "createTempEigenVectorRS(@)", vector.}

proc createTempEigenVectorRD(
  size, stride: int
): EigenVectorHandleRD {.importcpp: "createTempEigenVectorRD(@)", vector.}
proc createTempEigenVectorCS(
  size, stride: int
): EigenVectorHandleCS {.importcpp: "createTempEigenVectorCS(@)", vector.}

proc createTempEigenVectorCD(
  size, stride: int
): EigenVectorHandleCD {.importcpp: "createTempEigenVectorCD(@)", vector.}
# destructors

proc destroyEigenVectorRS(handle: EigenVectorHandleRS, ownsData: bool) 
  {.importcpp: "destroyEigenVectorRS(@)", vector.}

proc destroyEigenVectorRD(handle: EigenVectorHandleRD, ownsData: bool) 
  {.importcpp: "destroyEigenVectorRD(@)", vector.}

proc destroyEigenVectorCS(handle: EigenVectorHandleCS, ownsData: bool) 
  {.importcpp: "destroyEigenVectorCS(@)", vector.}

proc destroyEigenVectorCD(handle: EigenVectorHandleCD, ownsData: bool) 
  {.importcpp: "destroyEigenVectorCD(@)", vector.}

# clone — for =copy hook (new Map pointing to same data, ownsData=false)

proc cloneEigenVectorRS(handle: EigenVectorHandleRS): EigenVectorHandleRS
  {.importcpp: "cloneEigenVectorRS(@)", vector.}
proc cloneEigenVectorRD(handle: EigenVectorHandleRD): EigenVectorHandleRD
  {.importcpp: "cloneEigenVectorRD(@)", vector.}
proc cloneEigenVectorCS(handle: EigenVectorHandleCS): EigenVectorHandleCS
  {.importcpp: "cloneEigenVectorCS(@)", vector.}
proc cloneEigenVectorCD(handle: EigenVectorHandleCD): EigenVectorHandleCD
  {.importcpp: "cloneEigenVectorCD(@)", vector.}

# copy-from and fill — for := operator

proc eigenVectorRSCopyFrom(dst, src: EigenVectorHandleRS)
  {.importcpp: "eigenVectorRSCopyFrom(@)", vector.}
proc eigenVectorRDCopyFrom(dst, src: EigenVectorHandleRD)
  {.importcpp: "eigenVectorRDCopyFrom(@)", vector.}
proc eigenVectorCSCopyFrom(dst, src: EigenVectorHandleCS)
  {.importcpp: "eigenVectorCSCopyFrom(@)", vector.}
proc eigenVectorCDCopyFrom(dst, src: EigenVectorHandleCD)
  {.importcpp: "eigenVectorCDCopyFrom(@)", vector.}

proc eigenVectorRSFill(handle: EigenVectorHandleRS, value: float32)
  {.importcpp: "eigenVectorRSFill(@)", vector.}
proc eigenVectorRDFill(handle: EigenVectorHandleRD, value: float64)
  {.importcpp: "eigenVectorRDFill(@)", vector.}
proc eigenVectorCSFill(handle: EigenVectorHandleCS, value: pointer)
  {.importcpp: "eigenVectorCSFill(@)", vector.}
proc eigenVectorCDFill(handle: EigenVectorHandleCD, value: pointer)
  {.importcpp: "eigenVectorCDFill(@)", vector.}

# accessors

proc eigenVectorRSGet(handle: EigenVectorHandleRS, idx: int): float32 
  {.importcpp: "eigenVectorRSGet(@)", vector.}

proc eigenVectorRDGet(handle: EigenVectorHandleRD, idx: int): float64 
  {.importcpp: "eigenVectorRDGet(@)", vector.}

proc eigenVectorCSGet(handle: EigenVectorHandleCS, idx: int, outVal: pointer) 
  {.importcpp: "eigenVectorCSGet(@)", vector.}

proc eigenVectorCDGet(handle: EigenVectorHandleCD, idx: int, outVal: pointer) 
  {.importcpp: "eigenVectorCDGet(@)", vector.}

proc eigenVectorRSSet(handle: EigenVectorHandleRS, idx: int, value: float32) 
  {.importcpp: "eigenVectorRSSet(@)", vector.}

proc eigenVectorRDSet(handle: EigenVectorHandleRD, idx: int, value: float64) 
  {.importcpp: "eigenVectorRDSet(@)", vector.}

proc eigenVectorCSSet(handle: EigenVectorHandleCS, idx: int, value: pointer) 
  {.importcpp: "eigenVectorCSSet(@)", vector.}

proc eigenVectorCDSet(handle: EigenVectorHandleCD, idx: int, value: pointer) 
  {.importcpp: "eigenVectorCDSet(@)", vector.}

# algebra

proc eigenVectorRSAdd(a, b: EigenVectorHandleRS; c: EigenVectorHandleRS) 
  {.importcpp: "eigenVectorRSAdd(@)", vector.}

proc eigenVectorRDAdd(a, b: EigenVectorHandleRD; c: EigenVectorHandleRD) 
  {.importcpp: "eigenVectorRDAdd(@)", vector.}

proc eigenVectorCSAdd(a, b: EigenVectorHandleCS; c: EigenVectorHandleCS) 
  {.importcpp: "eigenVectorCSAdd(@)", vector.}

proc eigenVectorCDAdd(a, b: EigenVectorHandleCD; c: EigenVectorHandleCD) 
  {.importcpp: "eigenVectorCDAdd(@)", vector.}

proc eigenVectorRSSub(a, b: EigenVectorHandleRS; c: EigenVectorHandleRS) 
  {.importcpp: "eigenVectorRSSub(@)", vector.}

proc eigenVectorRDSub(a, b: EigenVectorHandleRD; c: EigenVectorHandleRD) 
  {.importcpp: "eigenVectorRDSub(@)", vector.}

proc eigenVectorCSSub(a, b: EigenVectorHandleCS; c: EigenVectorHandleCS) 
  {.importcpp: "eigenVectorCSSub(@)", vector.}

proc eigenVectorCDSub(a, b: EigenVectorHandleCD; c: EigenVectorHandleCD) 
  {.importcpp: "eigenVectorCDSub(@)", vector.}

proc eigenVectorRSMul(a, b: EigenVectorHandleRS; c: EigenVectorHandleRS) 
  {.importcpp: "eigenVectorRSMul(@)", vector.}

proc eigenVectorRDMul(a, b: EigenVectorHandleRD; c: EigenVectorHandleRD) 
  {.importcpp: "eigenVectorRDMul(@)", vector.}

proc eigenVectorCSMul(a, b: EigenVectorHandleCS; c: EigenVectorHandleCS) 
  {.importcpp: "eigenVectorCSMul(@)", vector.}

proc eigenVectorCDMul(a, b: EigenVectorHandleCD; c: EigenVectorHandleCD) 
  {.importcpp: "eigenVectorCDMul(@)", vector.}

# dot product

proc eigenVectorRSDot(a, b: EigenVectorHandleRS): float32 
  {.importcpp: "eigenVectorRSDot(@)", vector.}

proc eigenVectorRDDot(a, b: EigenVectorHandleRD): float64 
  {.importcpp: "eigenVectorRDDot(@)", vector.}

proc eigenVectorCSDot(a, b: EigenVectorHandleCS, outVal: pointer) 
  {.importcpp: "eigenVectorCSDot(@)", vector.}

proc eigenVectorCDDot(a, b: EigenVectorHandleCD, outVal: pointer) 
  {.importcpp: "eigenVectorCDDot(@)", vector.}

# norm

proc eigenVectorRSNorm(handle: EigenVectorHandleRS): float32 
  {.importcpp: "eigenVectorRSNorm(@)", vector.}

proc eigenVectorRDNorm(handle: EigenVectorHandleRD): float64 
  {.importcpp: "eigenVectorRDNorm(@)", vector.}

proc eigenVectorCSNorm(handle: EigenVectorHandleCS): float32 
  {.importcpp: "eigenVectorCSNorm(@)", vector.}

proc eigenVectorCDNorm(handle: EigenVectorHandleCD): float64 
  {.importcpp: "eigenVectorCDNorm(@)", vector.}

# compound assignment

proc eigenVectorRSAddAssign(a, b: EigenVectorHandleRS) 
  {.importcpp: "eigenVectorRSAddAssign(@)", vector.}

proc eigenVectorRDAddAssign(a, b: EigenVectorHandleRD) 
  {.importcpp: "eigenVectorRDAddAssign(@)", vector.}

proc eigenVectorCSAddAssign(a, b: EigenVectorHandleCS) 
  {.importcpp: "eigenVectorCSAddAssign(@)", vector.}

proc eigenVectorCDAddAssign(a, b: EigenVectorHandleCD) 
  {.importcpp: "eigenVectorCDAddAssign(@)", vector.}

proc eigenVectorRSSubAssign(a, b: EigenVectorHandleRS) 
  {.importcpp: "eigenVectorRSSubAssign(@)", vector.}

proc eigenVectorRDSubAssign(a, b: EigenVectorHandleRD) 
  {.importcpp: "eigenVectorRDSubAssign(@)", vector.}

proc eigenVectorCSSubAssign(a, b: EigenVectorHandleCS) 
  {.importcpp: "eigenVectorCSSubAssign(@)", vector.}

proc eigenVectorCDSubAssign(a, b: EigenVectorHandleCD) 
  {.importcpp: "eigenVectorCDSubAssign(@)", vector.}

# scalar multiply

proc eigenVectorRSScalarMul(a: EigenVectorHandleRS, s: float32, c: EigenVectorHandleRS) 
  {.importcpp: "eigenVectorRSScalarMul(@)", vector.}

proc eigenVectorRDScalarMul(a: EigenVectorHandleRD, s: float64, c: EigenVectorHandleRD) 
  {.importcpp: "eigenVectorRDScalarMul(@)", vector.}

proc eigenVectorCSScalarMul(a: EigenVectorHandleCS, s: pointer, c: EigenVectorHandleCS) 
  {.importcpp: "eigenVectorCSScalarMul(@)", vector.}

proc eigenVectorCDScalarMul(a: EigenVectorHandleCD, s: pointer, c: EigenVectorHandleCD) 
  {.importcpp: "eigenVectorCDScalarMul(@)", vector.}

proc eigenVectorRSScalarMulAssign(a: EigenVectorHandleRS, s: float32) 
  {.importcpp: "eigenVectorRSScalarMulAssign(@)", vector.}

proc eigenVectorRDScalarMulAssign(a: EigenVectorHandleRD, s: float64) 
  {.importcpp: "eigenVectorRDScalarMulAssign(@)", vector.}

proc eigenVectorCSScalarMulAssign(a: EigenVectorHandleCS, s: pointer) 
  {.importcpp: "eigenVectorCSScalarMulAssign(@)", vector.}

proc eigenVectorCDScalarMulAssign(a: EigenVectorHandleCD, s: pointer) 
  {.importcpp: "eigenVectorCDScalarMulAssign(@)", vector.}

# conjugate

proc eigenVectorRSConjugate(a: EigenVectorHandleRS, c: EigenVectorHandleRS) 
  {.importcpp: "eigenVectorRSConjugate(@)", vector.}

proc eigenVectorRDConjugate(a: EigenVectorHandleRD, c: EigenVectorHandleRD) 
  {.importcpp: "eigenVectorRDConjugate(@)", vector.}

proc eigenVectorCSConjugate(a: EigenVectorHandleCS, c: EigenVectorHandleCS) 
  {.importcpp: "eigenVectorCSConjugate(@)", vector.}

proc eigenVectorCDConjugate(a: EigenVectorHandleCD, c: EigenVectorHandleCD) 
  {.importcpp: "eigenVectorCDConjugate(@)", vector.}

# negate

proc eigenVectorRSNegate(a: EigenVectorHandleRS, c: EigenVectorHandleRS) 
  {.importcpp: "eigenVectorRSNegate(@)", vector.}

proc eigenVectorRDNegate(a: EigenVectorHandleRD, c: EigenVectorHandleRD) 
  {.importcpp: "eigenVectorRDNegate(@)", vector.}

proc eigenVectorCSNegate(a: EigenVectorHandleCS, c: EigenVectorHandleCS) 
  {.importcpp: "eigenVectorCSNegate(@)", vector.}

proc eigenVectorCDNegate(a: EigenVectorHandleCD, c: EigenVectorHandleCD) 
  {.importcpp: "eigenVectorCDNegate(@)", vector.}

# normalized

proc eigenVectorRSNormalized(a: EigenVectorHandleRS, c: EigenVectorHandleRS) 
  {.importcpp: "eigenVectorRSNormalized(@)", vector.}

proc eigenVectorRDNormalized(a: EigenVectorHandleRD, c: EigenVectorHandleRD) 
  {.importcpp: "eigenVectorRDNormalized(@)", vector.}

proc eigenVectorCSNormalized(a: EigenVectorHandleCS, c: EigenVectorHandleCS) 
  {.importcpp: "eigenVectorCSNormalized(@)", vector.}

proc eigenVectorCDNormalized(a: EigenVectorHandleCD, c: EigenVectorHandleCD) 
  {.importcpp: "eigenVectorCDNormalized(@)", vector.}

# normalize (in-place)

proc eigenVectorRSNormalize(a: EigenVectorHandleRS) 
  {.importcpp: "eigenVectorRSNormalize(@)", vector.}

proc eigenVectorRDNormalize(a: EigenVectorHandleRD) 
  {.importcpp: "eigenVectorRDNormalize(@)", vector.}

proc eigenVectorCSNormalize(a: EigenVectorHandleCS) 
  {.importcpp: "eigenVectorCSNormalize(@)", vector.}

proc eigenVectorCDNormalize(a: EigenVectorHandleCD) 
  {.importcpp: "eigenVectorCDNormalize(@)", vector.}

# squaredNorm

proc eigenVectorRSSquaredNorm(handle: EigenVectorHandleRS): float32 
  {.importcpp: "eigenVectorRSSquaredNorm(@)", vector.}

proc eigenVectorRDSquaredNorm(handle: EigenVectorHandleRD): float64 
  {.importcpp: "eigenVectorRDSquaredNorm(@)", vector.}

proc eigenVectorCSSquaredNorm(handle: EigenVectorHandleCS): float32 
  {.importcpp: "eigenVectorCSSquaredNorm(@)", vector.}

proc eigenVectorCDSquaredNorm(handle: EigenVectorHandleCD): float64 
  {.importcpp: "eigenVectorCDSquaredNorm(@)", vector.}

# cross product (3D only)

proc eigenVectorRSCross(a, b: EigenVectorHandleRS; c: EigenVectorHandleRS) 
  {.importcpp: "eigenVectorRSCross(@)", vector.}

proc eigenVectorRDCross(a, b: EigenVectorHandleRD; c: EigenVectorHandleRD) 
  {.importcpp: "eigenVectorRDCross(@)", vector.}

proc eigenVectorCSCross(a, b: EigenVectorHandleCS; c: EigenVectorHandleCS) 
  {.importcpp: "eigenVectorCSCross(@)", vector.}

proc eigenVectorCDCross(a, b: EigenVectorHandleCD; c: EigenVectorHandleCD) 
  {.importcpp: "eigenVectorCDCross(@)", vector.}

#[ EigenVector implementation ]#

recordImpl EigenVector:
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

  method `[]`(idx: int): T {.immutable.} =
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

  method `+`(other: EigenVector[T]): EigenVector[T] {.immutable.} =
    assert this.size == other.size, "size mismatch"
    result = newEigenVector[T](this.size, this.stride)
    when isReal32(T): eigenVectorRSAdd(this.data, other.data, result.data)
    elif isReal64(T): eigenVectorRDAdd(this.data, other.data, result.data)
    elif isComplex32(T): eigenVectorCSAdd(this.data, other.data, result.data)
    elif isComplex64(T): eigenVectorCDAdd(this.data, other.data, result.data)
  
  method `-`(other: EigenVector[T]): EigenVector[T] {.immutable.} =
    assert this.size == other.size, "size mismatch"
    result = newEigenVector[T](this.size, this.stride)
    when isReal32(T): eigenVectorRSSub(this.data, other.data, result.data)
    elif isReal64(T): eigenVectorRDSub(this.data, other.data, result.data)
    elif isComplex32(T): eigenVectorCSSub(this.data, other.data, result.data)
    elif isComplex64(T): eigenVectorCDSub(this.data, other.data, result.data)
  
  method `*`(other: EigenVector[T]): EigenVector[T] {.immutable.} =
    assert this.size == other.size, "size mismatch"
    result = newEigenVector[T](this.size, this.stride)
    when isReal32(T): eigenVectorRSMul(this.data, other.data, result.data)
    elif isReal64(T): eigenVectorRDMul(this.data, other.data, result.data)
    elif isComplex32(T): eigenVectorCSMul(this.data, other.data, result.data)
    elif isComplex64(T): eigenVectorCDMul(this.data, other.data, result.data)
  
  method `*`(scalar: T): EigenVector[T] {.immutable.} =
    result = newEigenVector[T](this.size, this.stride)
    when isReal32(T): eigenVectorRSScalarMul(this.data, scalar, result.data)
    elif isReal64(T): eigenVectorRDScalarMul(this.data, scalar, result.data)
    elif isComplex32(T):
      var s = scalar
      eigenVectorCSScalarMul(this.data, addr s, result.data)
    elif isComplex64(T):
      var s = scalar
      eigenVectorCDScalarMul(this.data, addr s, result.data)

  method dot(other: EigenVector[T]): T {.immutable.} =
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
  
  method norm(): auto {.immutable.} =
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

  method conjugate(): EigenVector[T] {.immutable.} =
    result = newEigenVector[T](this.size, this.stride)
    when isReal32(T): eigenVectorRSConjugate(this.data, result.data)
    elif isReal64(T): eigenVectorRDConjugate(this.data, result.data)
    elif isComplex32(T): eigenVectorCSConjugate(this.data, result.data)
    elif isComplex64(T): eigenVectorCDConjugate(this.data, result.data)
  
  method `-`(): EigenVector[T] {.immutable.} =
    result = newEigenVector[T](this.size, this.stride)
    when isReal32(T): eigenVectorRSNegate(this.data, result.data)
    elif isReal64(T): eigenVectorRDNegate(this.data, result.data)
    elif isComplex32(T): eigenVectorCSNegate(this.data, result.data)
    elif isComplex64(T): eigenVectorCDNegate(this.data, result.data)
  
  method normalized(): EigenVector[T] {.immutable.} =
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

# =copy hook: when an EigenVector is copied (e.g. passed by value to an
# {.immutable.} method), create a fresh Map pointing to the same underlying
# data with ownsData=false so that the copy's =destroy only frees the Map
# wrapper — not the data — leaving the original intact.
proc `=copy`*[T](dst: var EigenVector[T]; src: EigenVector[T]) =
  if addr(dst) == unsafeAddr(src): return
  dst.size = src.size
  dst.stride = src.stride
  dst.ownsData = false
  when isReal32(T):    dst.data = cloneEigenVectorRS(src.data)
  elif isReal64(T):    dst.data = cloneEigenVectorRD(src.data)
  elif isComplex32(T): dst.data = cloneEigenVectorCS(src.data)
  elif isComplex64(T): dst.data = cloneEigenVectorCD(src.data)

proc `=sink`*[T](dst: var EigenVector[T]; src: EigenVector[T]) =
  # Transfer ownership from src to dst without cloning.
  # =destroy will NOT be called on src by the compiler after this.
  `=destroy`(dst)
  dst.size = src.size
  dst.stride = src.stride
  dst.ownsData = src.ownsData
  when isReal32(T):    dst.data = src.data
  elif isReal64(T):    dst.data = src.data
  elif isComplex32(T): dst.data = src.data
  elif isComplex64(T): dst.data = src.data

proc `:=`*[T](dst: EigenVector[T]; src: EigenVector[T]) =
  ## Write-through: copies src's elements into dst's underlying buffer.
  ## dst need not be `var` — writes go through the C++ handle, not the struct.
  assert dst.size == src.size, "size mismatch"
  when isReal32(T):    eigenVectorRSCopyFrom(dst.data, src.data)
  elif isReal64(T):    eigenVectorRDCopyFrom(dst.data, src.data)
  elif isComplex32(T): eigenVectorCSCopyFrom(dst.data, src.data)
  elif isComplex64(T): eigenVectorCDCopyFrom(dst.data, src.data)

proc `:=`*[T](dst: EigenVector[T]; val: T) =
  ## Fill: set every element of the vector to val through the view.
  ## dst need not be `var` — writes go through the C++ handle, not the struct.
  when isReal32(T):    eigenVectorRSFill(dst.data, val)
  elif isReal64(T):    eigenVectorRDFill(dst.data, val)
  elif isComplex32(T):
    var v = val
    eigenVectorCSFill(dst.data, addr v)
  elif isComplex64(T):
    var v = val
    eigenVectorCDFill(dst.data, addr v)

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

    test ":= copy-from (write-through from another vector)":
      var adata = [1.0, 2.0, 3.0]
      var bdata = [10.0, 20.0, 30.0]
      var a = newEigenVector(addr adata[0], 3, 1)
      var b = newEigenVector(addr bdata[0], 3, 1)

      # := writes b's values into a's underlying buffer
      a := b
      check adata[0] == 10.0
      check adata[1] == 20.0
      check adata[2] == 30.0
      check a[0] == 10.0
      check a[1] == 20.0
      check a[2] == 30.0

      # b's buffer is untouched
      check bdata[0] == 10.0

    test ":= fill (set all elements to a scalar)":
      var vdata = [1.0, 2.0, 3.0, 4.0]
      var v = newEigenVector(addr vdata[0], 4, 1)

      v := 99.0
      for i in 0..<4:
        check vdata[i] == 99.0
        check v[i] == 99.0

    test ":= complex fill":
      var cdata = [complex(0.0, 0.0), complex(0.0, 0.0), complex(0.0, 0.0)]
      var v = newEigenVector(addr cdata[0], 3, 1)

      v := complex(3.0, -4.0)
      for i in 0..<3:
        check cdata[i] == complex(3.0, -4.0)
        check v[i] == complex(3.0, -4.0)