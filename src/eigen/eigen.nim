#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/eigen/eigen.nim
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

import eigenscalar
import eigenvector
import eigenmatrix
import types/[complex]

export eigenscalar
export eigenvector
export eigenmatrix

#[ Nim wrappers for cross-type C++ functions ]#

template eigenHeaders*: untyped =
  eigenScalarHeader()
  eigenVectorHeader()
  eigenMatrixHeader()
  {.pragma: eigen, header: "eigen.h".}

eigenHeaders()

#[ extra constructors ]#

proc newLocalScalar*(T: typedesc): EigenScalar[T] = newEigenScalar(default(T))

proc newLocalVector*(size: int; T: typedesc): EigenVector[T] =
  const ghostWidth = 1
  when isComplex(T): 
    let stride = (2 + 2 * ghostWidth) div 2
  else: 
    let stride = 1 + 2 * ghostWidth
  return newEigenVector[T](size, stride)

proc newLocalMatrix*[R: static[int]](shape: array[R, int]; T: typedesc): EigenMatrix[R, T] =
  const ghostWidth = 1
  when isComplex(T): 
    let inner = (2 + 2 * ghostWidth) div 2
  else: 
    let inner = 1 + 2 * ghostWidth
  let outer = (shape[1] + 2 * ghostWidth) * inner
  return newEigenMatrix[R, T](shape[0], shape[1], inner, outer)

proc newLocalTensor*[R: static[int]](shape: array[R, int]; T: typedesc): auto =
  when R == 0: return newLocalScalar(T)
  elif R == 1: return newLocalVector(shape[0], T)
  elif R == 2: return newLocalMatrix(shape, T)
  else: raise newException(ValueError, "Unsupported tensor rank")

# matrix-vector multiply

proc eigenMatVecRSMul(m: EigenMatrixHandleRS, v: EigenVectorHandleRS, o: EigenVectorHandleRS)
  {.importcpp: "eigenMatVecRSMul(@)", eigen.}

proc eigenMatVecRDMul(m: EigenMatrixHandleRD, v: EigenVectorHandleRD, o: EigenVectorHandleRD)
  {.importcpp: "eigenMatVecRDMul(@)", eigen.}

proc eigenMatVecCSMul(m: EigenMatrixHandleCS, v: EigenVectorHandleCS, o: EigenVectorHandleCS)
  {.importcpp: "eigenMatVecCSMul(@)", eigen.}

proc eigenMatVecCDMul(m: EigenMatrixHandleCD, v: EigenVectorHandleCD, o: EigenVectorHandleCD)
  {.importcpp: "eigenMatVecCDMul(@)", eigen.}

# outer product

proc eigenOuterProductRS(v, w: EigenVectorHandleRS, o: EigenMatrixHandleRS)
  {.importcpp: "eigenOuterProductRS(@)", eigen.}

proc eigenOuterProductRD(v, w: EigenVectorHandleRD, o: EigenMatrixHandleRD)
  {.importcpp: "eigenOuterProductRD(@)", eigen.}

proc eigenOuterProductCS(v, w: EigenVectorHandleCS, o: EigenMatrixHandleCS)
  {.importcpp: "eigenOuterProductCS(@)", eigen.}

proc eigenOuterProductCD(v, w: EigenVectorHandleCD, o: EigenMatrixHandleCD)
  {.importcpp: "eigenOuterProductCD(@)", eigen.}

# scalar * matrix

proc eigenScalarMatRSMul(s: EigenScalarHandleRS, m: EigenMatrixHandleRS, o: EigenMatrixHandleRS)
  {.importcpp: "eigenScalarMatRSMul(@)", eigen.}

proc eigenScalarMatRDMul(s: EigenScalarHandleRD, m: EigenMatrixHandleRD, o: EigenMatrixHandleRD)
  {.importcpp: "eigenScalarMatRDMul(@)", eigen.}

proc eigenScalarMatCSMul(s: EigenScalarHandleCS, m: EigenMatrixHandleCS, o: EigenMatrixHandleCS)
  {.importcpp: "eigenScalarMatCSMul(@)", eigen.}

proc eigenScalarMatCDMul(s: EigenScalarHandleCD, m: EigenMatrixHandleCD, o: EigenMatrixHandleCD)
  {.importcpp: "eigenScalarMatCDMul(@)", eigen.}

# scalar * vector

proc eigenScalarVecRSMul(s: EigenScalarHandleRS, v: EigenVectorHandleRS, o: EigenVectorHandleRS)
  {.importcpp: "eigenScalarVecRSMul(@)", eigen.}

proc eigenScalarVecRDMul(s: EigenScalarHandleRD, v: EigenVectorHandleRD, o: EigenVectorHandleRD)
  {.importcpp: "eigenScalarVecRDMul(@)", eigen.}

proc eigenScalarVecCSMul(s: EigenScalarHandleCS, v: EigenVectorHandleCS, o: EigenVectorHandleCS)
  {.importcpp: "eigenScalarVecCSMul(@)", eigen.}

proc eigenScalarVecCDMul(s: EigenScalarHandleCD, v: EigenVectorHandleCD, o: EigenVectorHandleCD)
  {.importcpp: "eigenScalarVecCDMul(@)", eigen.}

#[ exported cross-type operations ]#

# matrix * vector → vector

proc `*`*[R: static[int], T](m: EigenMatrix[R, T], v: EigenVector[T]): EigenVector[T] =
  assert R == 2, "only rank-2 supported"
  assert m.cols == v.size, "matrix cols must equal vector size"
  result = newEigenVector[T](m.rows, 1)
  when isReal32(T): eigenMatVecRSMul(m.data, v.data, result.data)
  elif isReal64(T): eigenMatVecRDMul(m.data, v.data, result.data)
  elif isComplex32(T): eigenMatVecCSMul(m.data, v.data, result.data)
  elif isComplex64(T): eigenMatVecCDMul(m.data, v.data, result.data)

# outer product:  v ⊗ w = v * w^T → matrix

proc outerProduct*[T](v, w: EigenVector[T]): EigenMatrix[2, T] =
  result = newEigenMatrix[2, T](v.size, w.size, 1, w.size)
  when isReal32(T): eigenOuterProductRS(v.data, w.data, result.data)
  elif isReal64(T): eigenOuterProductRD(v.data, w.data, result.data)
  elif isComplex32(T): eigenOuterProductCS(v.data, w.data, result.data)
  elif isComplex64(T): eigenOuterProductCD(v.data, w.data, result.data)

# scalar * matrix → matrix

proc `*`*[R: static[int], T](s: EigenScalar[T], m: EigenMatrix[R, T]): EigenMatrix[R, T] =
  result = newEigenMatrix[R, T](m.rows, m.cols, m.inner, m.outer)
  when isReal32(T): eigenScalarMatRSMul(s.data, m.data, result.data)
  elif isReal64(T): eigenScalarMatRDMul(s.data, m.data, result.data)
  elif isComplex32(T): eigenScalarMatCSMul(s.data, m.data, result.data)
  elif isComplex64(T): eigenScalarMatCDMul(s.data, m.data, result.data)

# matrix * scalar → matrix  (commutative)

proc `*`*[R: static[int], T](m: EigenMatrix[R, T], s: EigenScalar[T]): EigenMatrix[R, T] =
  result = s * m

# scalar * vector → vector

proc `*`*[T](s: EigenScalar[T], v: EigenVector[T]): EigenVector[T] =
  result = newEigenVector[T](v.size, v.stride)
  when isReal32(T): eigenScalarVecRSMul(s.data, v.data, result.data)
  elif isReal64(T): eigenScalarVecRDMul(s.data, v.data, result.data)
  elif isComplex32(T): eigenScalarVecCSMul(s.data, v.data, result.data)
  elif isComplex64(T): eigenScalarVecCDMul(s.data, v.data, result.data)

# vector * scalar → vector  (commutative)

proc `*`*[T](v: EigenVector[T], s: EigenScalar[T]): EigenVector[T] =
  result = s * v

when isMainModule:
  import std/[unittest, math]

  suite "Eigen cross-type operations":

    test "matrix * vector (real)":
      # A = [[1, 2], [3, 4]],  v = [5, 6]
      # A*v = [1*5+2*6, 3*5+4*6] = [17, 39]
      var mdata = [1.0, 2.0, 3.0, 4.0]
      var vdata = [5.0, 6.0]
      var m = newEigenMatrix(addr mdata[0], [2, 2], 1, 2)
      var v = newEigenVector(addr vdata[0], 2, 1)
      var r = m * v
      check r[0] == 17.0
      check r[1] == 39.0

    test "matrix * vector (non-square)":
      # A = [[1, 2, 3], [4, 5, 6]],  v = [1, 1, 1]
      # A*v = [6, 15]
      var mdata = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
      var vdata = [1.0, 1.0, 1.0]
      var m = newEigenMatrix(addr mdata[0], [2, 3], 1, 3)
      var v = newEigenVector(addr vdata[0], 3, 1)
      var r = m * v
      check r.size == 2
      check r[0] == 6.0
      check r[1] == 15.0

    test "matrix * vector (complex)":
      # A = [[1+i, 0], [0, 2-i]],  v = [1, 1+i]
      # A*v = [(1+i)*1, (2-i)*(1+i)] = [1+i, 2+2i-i-i² = 3+i]
      var mdata = [complex(1.0, 1.0), complex(0.0, 0.0),
                   complex(0.0, 0.0), complex(2.0, -1.0)]
      var vdata = [complex(1.0, 0.0), complex(1.0, 1.0)]
      var m = newEigenMatrix(addr mdata[0], [2, 2], 1, 2)
      var v = newEigenVector(addr vdata[0], 2, 1)
      var r = m * v
      check r[0] == complex(1.0, 1.0)
      check r[1] == complex(3.0, 1.0)

    test "outer product (real)":
      # v = [1, 2, 3],  w = [4, 5]
      # v ⊗ w = [[4, 5], [8, 10], [12, 15]]
      var vdata = [1.0, 2.0, 3.0]
      var wdata = [4.0, 5.0]
      var v = newEigenVector(addr vdata[0], 3, 1)
      var w = newEigenVector(addr wdata[0], 2, 1)
      var op = outerProduct(v, w)
      check op.rows == 3
      check op.cols == 2
      check op[0, 0] == 4.0
      check op[0, 1] == 5.0
      check op[1, 0] == 8.0
      check op[1, 1] == 10.0
      check op[2, 0] == 12.0
      check op[2, 1] == 15.0

    test "outer product (complex)":
      # v = [1+i, 2],  w = [1, i]
      # v * w^T = [[1+i, (1+i)*i], [2, 2i]]
      #         = [[1+i, -1+i], [2, 2i]]
      var vdata = [complex(1.0, 1.0), complex(2.0, 0.0)]
      var wdata = [complex(1.0, 0.0), complex(0.0, 1.0)]
      var v = newEigenVector(addr vdata[0], 2, 1)
      var w = newEigenVector(addr wdata[0], 2, 1)
      var op = outerProduct(v, w)
      check op[0, 0] == complex(1.0, 1.0)
      check op[0, 1] == complex(-1.0, 1.0)
      check op[1, 0] == complex(2.0, 0.0)
      check op[1, 1] == complex(0.0, 2.0)

    test "scalar * matrix (real)":
      var sval = 3.0
      var mdata = [1.0, 2.0, 3.0, 4.0]
      var s = newEigenScalar(addr sval)
      var m = newEigenMatrix(addr mdata[0], [2, 2], 1, 2)
      var r = s * m
      check r[0, 0] == 3.0
      check r[0, 1] == 6.0
      check r[1, 0] == 9.0
      check r[1, 1] == 12.0

      # commutative: matrix * scalar
      var r2 = m * s
      check r2[0, 0] == 3.0
      check r2[1, 1] == 12.0

    test "scalar * matrix (complex)":
      var sval = complex(0.0, 1.0)  # i
      var mdata = [complex(1.0, 0.0), complex(0.0, 1.0),
                   complex(2.0, 0.0), complex(0.0, -1.0)]
      var s = newEigenScalar(addr sval)
      var m = newEigenMatrix(addr mdata[0], [2, 2], 1, 2)
      # i * [[1, i], [2, -i]] = [[i, i²], [2i, -i²]] = [[i, -1], [2i, 1]]
      var r = s * m
      check r[0, 0] == complex(0.0, 1.0)
      check r[0, 1] == complex(-1.0, 0.0)
      check r[1, 0] == complex(0.0, 2.0)
      check r[1, 1] == complex(1.0, 0.0)

    test "scalar * vector (real)":
      var sval = 5.0
      var vdata = [1.0, 2.0, 3.0]
      var s = newEigenScalar(addr sval)
      var v = newEigenVector(addr vdata[0], 3, 1)
      var r = s * v
      check r[0] == 5.0
      check r[1] == 10.0
      check r[2] == 15.0

      # commutative: vector * scalar
      var r2 = v * s
      check r2[0] == 5.0
      check r2[2] == 15.0

    test "scalar * vector (complex)":
      var sval = complex(0.0, 1.0)  # i
      var vdata = [complex(1.0, 0.0), complex(0.0, 1.0), complex(2.0, -1.0)]
      var s = newEigenScalar(addr sval)
      var v = newEigenVector(addr vdata[0], 3, 1)
      # i * [1, i, 2-i] = [i, -1, 1+2i]
      var r = s * v
      check r[0] == complex(0.0, 1.0)
      check r[1] == complex(-1.0, 0.0)
      check r[2] == complex(1.0, 2.0)

    test "A*A^-1 * v = v (round-trip)":
      var mdata = [2.0, 1.0, 1.0, 3.0]
      var vdata = [7.0, 11.0]
      var m = newEigenMatrix(addr mdata[0], [2, 2], 1, 2)
      var v = newEigenVector(addr vdata[0], 2, 1)
      var mi = m.inverse()
      var w = m * v
      var v2 = mi * w
      check abs(v2[0] - 7.0) < 1e-12
      check abs(v2[1] - 11.0) < 1e-12

    test "outer product rank-1 structure":
      # v ⊗ w has rank 1, so det of square outer product = 0
      var vdata = [1.0, 2.0]
      var wdata = [3.0, 4.0]
      var v = newEigenVector(addr vdata[0], 2, 1)
      var w = newEigenVector(addr wdata[0], 2, 1)
      var op = outerProduct(v, w)
      check abs(op.determinant()) < 1e-12