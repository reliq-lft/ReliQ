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

template eigenMatrixHeader*: untyped =
  {.pragma: matrix, header: "eigenmatrix.h".}

eigenMatrixHeader()

type
  EigenMatrixHandleRS* {.importcpp: "EigenMatrixHandleRS", matrix.} = object
  EigenMatrixHandleRD* {.importcpp: "EigenMatrixHandleRD", matrix.} = object
  EigenMatrixHandleCS* {.importcpp: "EigenMatrixHandleCS", matrix.} = object
  EigenMatrixHandleCD* {.importcpp: "EigenMatrixHandleCD", matrix.} = object

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
): EigenMatrixHandleRS {.importcpp: "createEigenMatrixRS(@)", matrix.}

proc createEigenMatrixRD(
  data: ptr float64; 
  rows, cols, outer, inner: int
): EigenMatrixHandleRD {.importcpp: "createEigenMatrixRD(@)", matrix.}

proc createEigenMatrixCS(
  data: pointer; 
  rows, cols, outer, inner: int
): EigenMatrixHandleCS {.importcpp: "createEigenMatrixCS(@)", matrix.}

proc createEigenMatrixCD(
  data: pointer; 
  rows, cols, outer, inner: int
): EigenMatrixHandleCD {.importcpp: "createEigenMatrixCD(@)", matrix.}
  
# temp constructors

proc createTempEigenMatrixRS(
  rows, cols, outer, inner: int
): EigenMatrixHandleRS {.importcpp: "createTempEigenMatrixRS(@)", matrix.}

proc createTempEigenMatrixRD(
  rows, cols, outer, inner: int
): EigenMatrixHandleRD {.importcpp: "createTempEigenMatrixRD(@)", matrix.}
proc createTempEigenMatrixCS(
  rows, cols, outer, inner: int
): EigenMatrixHandleCS {.importcpp: "createTempEigenMatrixCS(@)", matrix.}

proc createTempEigenMatrixCD(
  rows, cols, outer, inner: int
): EigenMatrixHandleCD {.importcpp: "createTempEigenMatrixCD(@)", matrix.}
# destructors

proc destroyEigenMatrixRS(handle: EigenMatrixHandleRS, ownsData: bool) 
  {.importcpp: "destroyEigenMatrixRS(@)", matrix.}

proc destroyEigenMatrixRD(handle: EigenMatrixHandleRD, ownsData: bool) 
  {.importcpp: "destroyEigenMatrixRD(@)", matrix.}

proc destroyEigenMatrixCS(handle: EigenMatrixHandleCS, ownsData: bool) 
  {.importcpp: "destroyEigenMatrixCS(@)", matrix.}

proc destroyEigenMatrixCD(handle: EigenMatrixHandleCD, ownsData: bool) 
  {.importcpp: "destroyEigenMatrixCD(@)", matrix.}

# clone — for =copy hook (new Map pointing to same data, ownsData=false)

proc cloneEigenMatrixRS(handle: EigenMatrixHandleRS): EigenMatrixHandleRS
  {.importcpp: "cloneEigenMatrixRS(@)", matrix.}
proc cloneEigenMatrixRD(handle: EigenMatrixHandleRD): EigenMatrixHandleRD
  {.importcpp: "cloneEigenMatrixRD(@)", matrix.}
proc cloneEigenMatrixCS(handle: EigenMatrixHandleCS): EigenMatrixHandleCS
  {.importcpp: "cloneEigenMatrixCS(@)", matrix.}
proc cloneEigenMatrixCD(handle: EigenMatrixHandleCD): EigenMatrixHandleCD
  {.importcpp: "cloneEigenMatrixCD(@)", matrix.}

# copy-from and fill — for := operator

proc eigenMatrixRSCopyFrom(dst, src: EigenMatrixHandleRS)
  {.importcpp: "eigenMatrixRSCopyFrom(@)", matrix.}
proc eigenMatrixRDCopyFrom(dst, src: EigenMatrixHandleRD)
  {.importcpp: "eigenMatrixRDCopyFrom(@)", matrix.}
proc eigenMatrixCSCopyFrom(dst, src: EigenMatrixHandleCS)
  {.importcpp: "eigenMatrixCSCopyFrom(@)", matrix.}
proc eigenMatrixCDCopyFrom(dst, src: EigenMatrixHandleCD)
  {.importcpp: "eigenMatrixCDCopyFrom(@)", matrix.}

proc eigenMatrixRSFill(handle: EigenMatrixHandleRS, value: float32)
  {.importcpp: "eigenMatrixRSFill(@)", matrix.}
proc eigenMatrixRDFill(handle: EigenMatrixHandleRD, value: float64)
  {.importcpp: "eigenMatrixRDFill(@)", matrix.}
proc eigenMatrixCSFill(handle: EigenMatrixHandleCS, value: pointer)
  {.importcpp: "eigenMatrixCSFill(@)", matrix.}
proc eigenMatrixCDFill(handle: EigenMatrixHandleCD, value: pointer)
  {.importcpp: "eigenMatrixCDFill(@)", matrix.}

# accessors

proc eigenMatrixRSGet(handle: EigenMatrixHandleRS, row, col: int): float32 
  {.importcpp: "eigenMatrixRSGet(@)", matrix.}

proc eigenMatrixRDGet(handle: EigenMatrixHandleRD, row, col: int): float64 
  {.importcpp: "eigenMatrixRDGet(@)", matrix.}

proc eigenMatrixCSGet(handle: EigenMatrixHandleCS, row, col: int, outVal: pointer) 
  {.importcpp: "eigenMatrixCSGet(@)", matrix.}

proc eigenMatrixCDGet(handle: EigenMatrixHandleCD, row, col: int, outVal: pointer) 
  {.importcpp: "eigenMatrixCDGet(@)", matrix.}

proc eigenMatrixRSSet(handle: EigenMatrixHandleRS, row, col: int, value: float32) 
  {.importcpp: "eigenMatrixRSSet(@)", matrix.}

proc eigenMatrixRDSet(handle: EigenMatrixHandleRD, row, col: int, value: float64) 
  {.importcpp: "eigenMatrixRDSet(@)", matrix.}

proc eigenMatrixCSSet(handle: EigenMatrixHandleCS, row, col: int, value: pointer) 
  {.importcpp: "eigenMatrixCSSet(@)", matrix.}

proc eigenMatrixCDSet(handle: EigenMatrixHandleCD, row, col: int, value: pointer) 
  {.importcpp: "eigenMatrixCDSet(@)", matrix.}

# algebra

proc eigenMatrixRSAdd(a, b: EigenMatrixHandleRS; c: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSAdd(@)", matrix.}

proc eigenMatrixRDAdd(a, b: EigenMatrixHandleRD; c: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDAdd(@)", matrix.}

proc eigenMatrixCSAdd(a, b: EigenMatrixHandleCS; c: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSAdd(@)", matrix.}

proc eigenMatrixCDAdd(a, b: EigenMatrixHandleCD; c: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDAdd(@)", matrix.}

proc eigenMatrixRSSub(a, b: EigenMatrixHandleRS; c: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSSub(@)", matrix.}

proc eigenMatrixRDSub(a, b: EigenMatrixHandleRD; c: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDSub(@)", matrix.}

proc eigenMatrixCSSub(a, b: EigenMatrixHandleCS; c: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSSub(@)", matrix.}

proc eigenMatrixCDSub(a, b: EigenMatrixHandleCD; c: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDSub(@)", matrix.}

proc eigenMatrixRSMul(a, b: EigenMatrixHandleRS; c: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSMul(@)", matrix.}

proc eigenMatrixRDMul(a, b: EigenMatrixHandleRD; c: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDMul(@)", matrix.}

proc eigenMatrixCSMul(a, b: EigenMatrixHandleCS; c: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSMul(@)", matrix.}

proc eigenMatrixCDMul(a, b: EigenMatrixHandleCD; c: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDMul(@)", matrix.}

proc eigenMatrixRSAddAssign(a, b: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSAddAssign(@)", matrix.}

proc eigenMatrixRDAddAssign(a, b: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDAddAssign(@)", matrix.}

proc eigenMatrixCSAddAssign(a, b: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSAddAssign(@)", matrix.}

proc eigenMatrixCDAddAssign(a, b: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDAddAssign(@)", matrix.}

proc eigenMatrixRSSubAssign(a, b: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSSubAssign(@)", matrix.}

proc eigenMatrixRDSubAssign(a, b: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDSubAssign(@)", matrix.}

proc eigenMatrixCSSubAssign(a, b: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSSubAssign(@)", matrix.}

proc eigenMatrixCDSubAssign(a, b: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDSubAssign(@)", matrix.}

proc eigenMatrixRSMulAssign(a, b: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSMulAssign(@)", matrix.}

proc eigenMatrixRDMulAssign(a, b: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDMulAssign(@)", matrix.}

proc eigenMatrixCSMulAssign(a, b: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSMulAssign(@)", matrix.}

proc eigenMatrixCDMulAssign(a, b: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDMulAssign(@)", matrix.}

# transpose

proc eigenMatrixRSTranspose(a: EigenMatrixHandleRS, c: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSTranspose(@)", matrix.}

proc eigenMatrixRDTranspose(a: EigenMatrixHandleRD, c: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDTranspose(@)", matrix.}

proc eigenMatrixCSTranspose(a: EigenMatrixHandleCS, c: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSTranspose(@)", matrix.}

proc eigenMatrixCDTranspose(a: EigenMatrixHandleCD, c: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDTranspose(@)", matrix.}

# adjoint

proc eigenMatrixRSAdjoint(a: EigenMatrixHandleRS, c: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSAdjoint(@)", matrix.}

proc eigenMatrixRDAdjoint(a: EigenMatrixHandleRD, c: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDAdjoint(@)", matrix.}

proc eigenMatrixCSAdjoint(a: EigenMatrixHandleCS, c: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSAdjoint(@)", matrix.}

proc eigenMatrixCDAdjoint(a: EigenMatrixHandleCD, c: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDAdjoint(@)", matrix.}

# conjugate

proc eigenMatrixRSConjugate(a: EigenMatrixHandleRS, c: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSConjugate(@)", matrix.}

proc eigenMatrixRDConjugate(a: EigenMatrixHandleRD, c: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDConjugate(@)", matrix.}

proc eigenMatrixCSConjugate(a: EigenMatrixHandleCS, c: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSConjugate(@)", matrix.}

proc eigenMatrixCDConjugate(a: EigenMatrixHandleCD, c: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDConjugate(@)", matrix.}

# trace

proc eigenMatrixRSTrace(handle: EigenMatrixHandleRS): float32 
  {.importcpp: "eigenMatrixRSTrace(@)", matrix.}

proc eigenMatrixRDTrace(handle: EigenMatrixHandleRD): float64 
  {.importcpp: "eigenMatrixRDTrace(@)", matrix.}

proc eigenMatrixCSTrace(handle: EigenMatrixHandleCS, outVal: pointer) 
  {.importcpp: "eigenMatrixCSTrace(@)", matrix.}

proc eigenMatrixCDTrace(handle: EigenMatrixHandleCD, outVal: pointer) 
  {.importcpp: "eigenMatrixCDTrace(@)", matrix.}

# determinant

proc eigenMatrixRSDet(handle: EigenMatrixHandleRS): float32 
  {.importcpp: "eigenMatrixRSDet(@)", matrix.}

proc eigenMatrixRDDet(handle: EigenMatrixHandleRD): float64 
  {.importcpp: "eigenMatrixRDDet(@)", matrix.}

proc eigenMatrixCSDet(handle: EigenMatrixHandleCS, outVal: pointer) 
  {.importcpp: "eigenMatrixCSDet(@)", matrix.}

proc eigenMatrixCDDet(handle: EigenMatrixHandleCD, outVal: pointer) 
  {.importcpp: "eigenMatrixCDDet(@)", matrix.}

# inverse

proc eigenMatrixRSInverse(a: EigenMatrixHandleRS, c: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSInverse(@)", matrix.}

proc eigenMatrixRDInverse(a: EigenMatrixHandleRD, c: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDInverse(@)", matrix.}

proc eigenMatrixCSInverse(a: EigenMatrixHandleCS, c: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSInverse(@)", matrix.}

proc eigenMatrixCDInverse(a: EigenMatrixHandleCD, c: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDInverse(@)", matrix.}

# scalar multiply

proc eigenMatrixRSScalarMul(a: EigenMatrixHandleRS, s: float32, c: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSScalarMul(@)", matrix.}

proc eigenMatrixRDScalarMul(a: EigenMatrixHandleRD, s: float64, c: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDScalarMul(@)", matrix.}

proc eigenMatrixCSScalarMul(a: EigenMatrixHandleCS, s: pointer, c: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSScalarMul(@)", matrix.}

proc eigenMatrixCDScalarMul(a: EigenMatrixHandleCD, s: pointer, c: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDScalarMul(@)", matrix.}

proc eigenMatrixRSScalarMulAssign(a: EigenMatrixHandleRS, s: float32) 
  {.importcpp: "eigenMatrixRSScalarMulAssign(@)", matrix.}

proc eigenMatrixRDScalarMulAssign(a: EigenMatrixHandleRD, s: float64) 
  {.importcpp: "eigenMatrixRDScalarMulAssign(@)", matrix.}

proc eigenMatrixCSScalarMulAssign(a: EigenMatrixHandleCS, s: pointer) 
  {.importcpp: "eigenMatrixCSScalarMulAssign(@)", matrix.}

proc eigenMatrixCDScalarMulAssign(a: EigenMatrixHandleCD, s: pointer) 
  {.importcpp: "eigenMatrixCDScalarMulAssign(@)", matrix.}

# negate

proc eigenMatrixRSNegate(a: EigenMatrixHandleRS, c: EigenMatrixHandleRS) 
  {.importcpp: "eigenMatrixRSNegate(@)", matrix.}

proc eigenMatrixRDNegate(a: EigenMatrixHandleRD, c: EigenMatrixHandleRD) 
  {.importcpp: "eigenMatrixRDNegate(@)", matrix.}

proc eigenMatrixCSNegate(a: EigenMatrixHandleCS, c: EigenMatrixHandleCS) 
  {.importcpp: "eigenMatrixCSNegate(@)", matrix.}

proc eigenMatrixCDNegate(a: EigenMatrixHandleCD, c: EigenMatrixHandleCD) 
  {.importcpp: "eigenMatrixCDNegate(@)", matrix.}

# norm (Frobenius)

proc eigenMatrixRSNorm(handle: EigenMatrixHandleRS): float32 
  {.importcpp: "eigenMatrixRSNorm(@)", matrix.}

proc eigenMatrixRDNorm(handle: EigenMatrixHandleRD): float64 
  {.importcpp: "eigenMatrixRDNorm(@)", matrix.}

proc eigenMatrixCSNorm(handle: EigenMatrixHandleCS): float32 
  {.importcpp: "eigenMatrixCSNorm(@)", matrix.}

proc eigenMatrixCDNorm(handle: EigenMatrixHandleCD): float64 
  {.importcpp: "eigenMatrixCDNorm(@)", matrix.}

#[ EigenMatrix implementation ]#

recordImpl EigenMatrix:
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

  method `[]`(row, col: int): T {.immutable.} =
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

  method `+`(other: EigenMatrix[R, T]): EigenMatrix[R, T] {.immutable.} =
    assert this.cols == other.cols and this.rows == other.rows, "shape mismatch"
    assert this.inner == other.inner and this.outer == other.outer, "stride mismatch"
    result = newEigenMatrix[R, T](this.rows, this.cols, this.inner, this.outer)
    when isReal32(T): eigenMatrixRSAdd(this.data, other.data, result.data)
    elif isReal64(T): eigenMatrixRDAdd(this.data, other.data, result.data)
    elif isComplex32(T): eigenMatrixCSAdd(this.data, other.data, result.data)
    elif isComplex64(T): eigenMatrixCDAdd(this.data, other.data, result.data)
  
  method `-`(other: EigenMatrix[R, T]): EigenMatrix[R, T] {.immutable.} =
    assert this.cols == other.cols and this.rows == other.rows, "shape mismatch"
    assert this.inner == other.inner and this.outer == other.outer, "stride mismatch"
    result = newEigenMatrix[R, T](this.rows, this.cols, this.inner, this.outer)
    when isReal32(T): eigenMatrixRSSub(this.data, other.data, result.data)
    elif isReal64(T): eigenMatrixRDSub(this.data, other.data, result.data)
    elif isComplex32(T): eigenMatrixCSSub(this.data, other.data, result.data)
    elif isComplex64(T): eigenMatrixCDSub(this.data, other.data, result.data)
  
  method `*`(other: EigenMatrix[R, T]): EigenMatrix[R, T] {.immutable.} =
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

  method transpose: EigenMatrix[R, T] {.immutable.} =
    result = newEigenMatrix[R, T](this.cols, this.rows, this.inner, this.outer)
    when isReal32(T): eigenMatrixRSTranspose(this.data, result.data)
    elif isReal64(T): eigenMatrixRDTranspose(this.data, result.data)
    elif isComplex32(T): eigenMatrixCSTranspose(this.data, result.data)
    elif isComplex64(T): eigenMatrixCDTranspose(this.data, result.data)
  
  method adjoint: EigenMatrix[R, T] {.immutable.} =
    result = newEigenMatrix[R, T](this.cols, this.rows, this.inner, this.outer)
    when isReal32(T): eigenMatrixRSAdjoint(this.data, result.data)
    elif isReal64(T): eigenMatrixRDAdjoint(this.data, result.data)
    elif isComplex32(T): eigenMatrixCSAdjoint(this.data, result.data)
    elif isComplex64(T): eigenMatrixCDAdjoint(this.data, result.data)
  
  method conjugate: EigenMatrix[R, T] {.immutable.} =
    result = newEigenMatrix[R, T](this.rows, this.cols, this.inner, this.outer)
    when isReal32(T): eigenMatrixRSConjugate(this.data, result.data)
    elif isReal64(T): eigenMatrixRDConjugate(this.data, result.data)
    elif isComplex32(T): eigenMatrixCSConjugate(this.data, result.data)
    elif isComplex64(T): eigenMatrixCDConjugate(this.data, result.data)
  
  method trace: T {.immutable.} =
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
  
  method determinant: T {.immutable.} =
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
  
  method inverse: EigenMatrix[R, T] {.immutable.} =
    assert this.rows == this.cols, "inverse requires square matrix"
    result = newEigenMatrix[R, T](this.rows, this.cols, this.inner, this.outer)
    when isReal32(T): eigenMatrixRSInverse(this.data, result.data)
    elif isReal64(T): eigenMatrixRDInverse(this.data, result.data)
    elif isComplex32(T): eigenMatrixCSInverse(this.data, result.data)
    elif isComplex64(T): eigenMatrixCDInverse(this.data, result.data)
  
  method `*`(scalar: T): EigenMatrix[R, T] {.immutable.} =
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
  
  method `-`: EigenMatrix[R, T] {.immutable.} =
    result = newEigenMatrix[R, T](this.rows, this.cols, this.inner, this.outer)
    when isReal32(T): eigenMatrixRSNegate(this.data, result.data)
    elif isReal64(T): eigenMatrixRDNegate(this.data, result.data)
    elif isComplex32(T): eigenMatrixCSNegate(this.data, result.data)
    elif isComplex64(T): eigenMatrixCDNegate(this.data, result.data)
  
  method norm: auto {.immutable.} =
    when isReal32(T):
      return eigenMatrixRSNorm(this.data)
    elif isReal64(T):
      return eigenMatrixRDNorm(this.data)
    elif isComplex32(T):
      return eigenMatrixCSNorm(this.data)
    elif isComplex64(T):
      return eigenMatrixCDNorm(this.data)

# =copy hook: when an EigenMatrix is copied (e.g. passed by value to an
# {.immutable.} method), create a fresh Map pointing to the same underlying
# data with ownsData=false so that the copy's =destroy only frees the Map
# wrapper — not the data — leaving the original intact.
proc `=copy`*[R: static[int], T](dst: var EigenMatrix[R, T]; src: EigenMatrix[R, T]) =
  if addr(dst) == unsafeAddr(src): return
  dst.rows = src.rows
  dst.cols = src.cols
  dst.inner = src.inner
  dst.outer = src.outer
  dst.ownsData = false
  when isReal32(T):    dst.data = cloneEigenMatrixRS(src.data)
  elif isReal64(T):    dst.data = cloneEigenMatrixRD(src.data)
  elif isComplex32(T): dst.data = cloneEigenMatrixCS(src.data)
  elif isComplex64(T): dst.data = cloneEigenMatrixCD(src.data)

proc `=sink`*[R: static[int], T](dst: var EigenMatrix[R, T]; src: EigenMatrix[R, T]) =
  # Transfer ownership from src to dst without cloning.
  # =destroy will NOT be called on src by the compiler after this.
  `=destroy`(dst)
  dst.rows = src.rows
  dst.cols = src.cols
  dst.inner = src.inner
  dst.outer = src.outer
  dst.ownsData = src.ownsData
  when isReal32(T):    dst.data = src.data
  elif isReal64(T):    dst.data = src.data
  elif isComplex32(T): dst.data = src.data
  elif isComplex64(T): dst.data = src.data

proc `:=`*[R: static[int], T](dst: EigenMatrix[R, T]; src: EigenMatrix[R, T]) =
  ## Write-through: copies src's elements into dst's underlying buffer.
  ## dst need not be `var` — writes go through the C++ handle, not the struct.
  assert dst.rows == src.rows and dst.cols == src.cols, "shape mismatch"
  when isReal32(T):    eigenMatrixRSCopyFrom(dst.data, src.data)
  elif isReal64(T):    eigenMatrixRDCopyFrom(dst.data, src.data)
  elif isComplex32(T): eigenMatrixCSCopyFrom(dst.data, src.data)
  elif isComplex64(T): eigenMatrixCDCopyFrom(dst.data, src.data)

proc `:=`*[R: static[int], T](dst: EigenMatrix[R, T]; val: T) =
  ## Fill: set every element of the matrix to val through the view.
  ## dst need not be `var` — writes go through the C++ handle, not the struct.
  when isReal32(T):    eigenMatrixRSFill(dst.data, val)
  elif isReal64(T):    eigenMatrixRDFill(dst.data, val)
  elif isComplex32(T):
    var v = val
    eigenMatrixCSFill(dst.data, addr v)
  elif isComplex64(T):
    var v = val
    eigenMatrixCDFill(dst.data, addr v)

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

    test ":= copy-from (write-through from another matrix)":
      var adata = [1.0, 2.0, 3.0, 4.0]
      var bdata = [10.0, 20.0, 30.0, 40.0]
      var a = newEigenMatrix(addr adata[0], [2, 2], 1, 2)
      var b = newEigenMatrix(addr bdata[0], [2, 2], 1, 2)

      # := writes b's values into a's underlying buffer
      a := b
      check adata[0] == 10.0
      check adata[1] == 20.0
      check adata[2] == 30.0
      check adata[3] == 40.0
      check a[0, 0] == 10.0
      check a[0, 1] == 20.0
      check a[1, 0] == 30.0
      check a[1, 1] == 40.0

      # b's buffer is untouched
      check bdata[0] == 10.0

    test ":= fill (set all elements to a scalar)":
      var adata = [1.0, 2.0, 3.0, 4.0]
      var a = newEigenMatrix(addr adata[0], [2, 2], 1, 2)

      a := 7.0
      for i in 0..<4:
        check adata[i] == 7.0
      check a[0, 0] == 7.0
      check a[0, 1] == 7.0
      check a[1, 0] == 7.0
      check a[1, 1] == 7.0

    test ":= complex fill":
      var cdata = [complex(0.0, 0.0), complex(0.0, 0.0),
                   complex(0.0, 0.0), complex(0.0, 0.0)]
      var a = newEigenMatrix(addr cdata[0], [2, 2], 1, 2)

      a := complex(2.0, -1.0)
      for i in 0..<4:
        check cdata[i] == complex(2.0, -1.0)
      check a[0, 0] == complex(2.0, -1.0)
      check a[1, 1] == complex(2.0, -1.0)