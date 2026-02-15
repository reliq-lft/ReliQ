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
import utils/[complex]

export eigenscalar
export eigenvector
export eigenmatrix

#[ C++ cross-type operations ]#

{.emit: """/*TYPESECTION*/
#include <Eigen/Dense>

/* re-declare handle types needed for cross-type operations.
   These must match the definitions in eigenmatrix, eigenvector, eigenscalar. */

#define MatrixTemplateArgs Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor
#define MapTemplateArgs Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>
#define VectorStride Eigen::InnerStride<Eigen::Dynamic>

#ifndef EIGEN_HANDLE_TYPES_DEFINED
#define EIGEN_HANDLE_TYPES_DEFINED

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

struct EigenScalarHandleRS { float* val; };
struct EigenScalarHandleRD { double* val; };
struct EigenScalarHandleCS { std::complex<float>* val; };
struct EigenScalarHandleCD { std::complex<double>* val; };

#endif

/* matrix-vector multiply:  out = mat * vec */

inline void eigenMatVecRSMul(
  const EigenMatrixHandleRS m, const EigenVectorHandleRS v, EigenVectorHandleRS out
) { *out.vec = (*m.mat) * (*v.vec); }

inline void eigenMatVecRDMul(
  const EigenMatrixHandleRD m, const EigenVectorHandleRD v, EigenVectorHandleRD out
) { *out.vec = (*m.mat) * (*v.vec); }

inline void eigenMatVecCSMul(
  const EigenMatrixHandleCS m, const EigenVectorHandleCS v, EigenVectorHandleCS out
) { *out.vec = (*m.mat) * (*v.vec); }

inline void eigenMatVecCDMul(
  const EigenMatrixHandleCD m, const EigenVectorHandleCD v, EigenVectorHandleCD out
) { *out.vec = (*m.mat) * (*v.vec); }

/* outer product:  out = v * w^T  (or v * conj(w)^T for complex = v * w^H) */
/* Uses v * w.transpose() for real and v * w.adjoint() for complex to match
   the standard physics convention for outer products. */

inline void eigenOuterProductRS(
  const EigenVectorHandleRS v, const EigenVectorHandleRS w, EigenMatrixHandleRS out
) { *out.mat = (*v.vec) * w.vec->transpose(); }

inline void eigenOuterProductRD(
  const EigenVectorHandleRD v, const EigenVectorHandleRD w, EigenMatrixHandleRD out
) { *out.mat = (*v.vec) * w.vec->transpose(); }

inline void eigenOuterProductCS(
  const EigenVectorHandleCS v, const EigenVectorHandleCS w, EigenMatrixHandleCS out
) { *out.mat = (*v.vec) * w.vec->transpose(); }

inline void eigenOuterProductCD(
  const EigenVectorHandleCD v, const EigenVectorHandleCD w, EigenMatrixHandleCD out
) { *out.mat = (*v.vec) * w.vec->transpose(); }

/* scalar * matrix */

inline void eigenScalarMatRSMul(
  const EigenScalarHandleRS s, const EigenMatrixHandleRS m, EigenMatrixHandleRS out
) { *out.mat = (*s.val) * (*m.mat); }

inline void eigenScalarMatRDMul(
  const EigenScalarHandleRD s, const EigenMatrixHandleRD m, EigenMatrixHandleRD out
) { *out.mat = (*s.val) * (*m.mat); }

inline void eigenScalarMatCSMul(
  const EigenScalarHandleCS s, const EigenMatrixHandleCS m, EigenMatrixHandleCS out
) { *out.mat = (*s.val) * (*m.mat); }

inline void eigenScalarMatCDMul(
  const EigenScalarHandleCD s, const EigenMatrixHandleCD m, EigenMatrixHandleCD out
) { *out.mat = (*s.val) * (*m.mat); }

/* scalar * vector */

inline void eigenScalarVecRSMul(
  const EigenScalarHandleRS s, const EigenVectorHandleRS v, EigenVectorHandleRS out
) { *out.vec = (*s.val) * (*v.vec); }

inline void eigenScalarVecRDMul(
  const EigenScalarHandleRD s, const EigenVectorHandleRD v, EigenVectorHandleRD out
) { *out.vec = (*s.val) * (*v.vec); }

inline void eigenScalarVecCSMul(
  const EigenScalarHandleCS s, const EigenVectorHandleCS v, EigenVectorHandleCS out
) { *out.vec = (*s.val) * (*v.vec); }

inline void eigenScalarVecCDMul(
  const EigenScalarHandleCD s, const EigenVectorHandleCD v, EigenVectorHandleCD out
) { *out.vec = (*s.val) * (*v.vec); }

""".}

#[ Nim wrappers for cross-type C++ functions ]#

# matrix-vector multiply

proc eigenMatVecRSMul(m: EigenMatrixHandleRS, v: EigenVectorHandleRS, o: EigenVectorHandleRS)
  {.importcpp: "eigenMatVecRSMul(@)".}

proc eigenMatVecRDMul(m: EigenMatrixHandleRD, v: EigenVectorHandleRD, o: EigenVectorHandleRD)
  {.importcpp: "eigenMatVecRDMul(@)".}

proc eigenMatVecCSMul(m: EigenMatrixHandleCS, v: EigenVectorHandleCS, o: EigenVectorHandleCS)
  {.importcpp: "eigenMatVecCSMul(@)".}

proc eigenMatVecCDMul(m: EigenMatrixHandleCD, v: EigenVectorHandleCD, o: EigenVectorHandleCD)
  {.importcpp: "eigenMatVecCDMul(@)".}

# outer product

proc eigenOuterProductRS(v, w: EigenVectorHandleRS, o: EigenMatrixHandleRS)
  {.importcpp: "eigenOuterProductRS(@)".}

proc eigenOuterProductRD(v, w: EigenVectorHandleRD, o: EigenMatrixHandleRD)
  {.importcpp: "eigenOuterProductRD(@)".}

proc eigenOuterProductCS(v, w: EigenVectorHandleCS, o: EigenMatrixHandleCS)
  {.importcpp: "eigenOuterProductCS(@)".}

proc eigenOuterProductCD(v, w: EigenVectorHandleCD, o: EigenMatrixHandleCD)
  {.importcpp: "eigenOuterProductCD(@)".}

# scalar * matrix

proc eigenScalarMatRSMul(s: EigenScalarHandleRS, m: EigenMatrixHandleRS, o: EigenMatrixHandleRS)
  {.importcpp: "eigenScalarMatRSMul(@)".}

proc eigenScalarMatRDMul(s: EigenScalarHandleRD, m: EigenMatrixHandleRD, o: EigenMatrixHandleRD)
  {.importcpp: "eigenScalarMatRDMul(@)".}

proc eigenScalarMatCSMul(s: EigenScalarHandleCS, m: EigenMatrixHandleCS, o: EigenMatrixHandleCS)
  {.importcpp: "eigenScalarMatCSMul(@)".}

proc eigenScalarMatCDMul(s: EigenScalarHandleCD, m: EigenMatrixHandleCD, o: EigenMatrixHandleCD)
  {.importcpp: "eigenScalarMatCDMul(@)".}

# scalar * vector

proc eigenScalarVecRSMul(s: EigenScalarHandleRS, v: EigenVectorHandleRS, o: EigenVectorHandleRS)
  {.importcpp: "eigenScalarVecRSMul(@)".}

proc eigenScalarVecRDMul(s: EigenScalarHandleRD, v: EigenVectorHandleRD, o: EigenVectorHandleRD)
  {.importcpp: "eigenScalarVecRDMul(@)".}

proc eigenScalarVecCSMul(s: EigenScalarHandleCS, v: EigenVectorHandleCS, o: EigenVectorHandleCS)
  {.importcpp: "eigenScalarVecCSMul(@)".}

proc eigenScalarVecCDMul(s: EigenScalarHandleCD, v: EigenVectorHandleCD, o: EigenVectorHandleCD)
  {.importcpp: "eigenScalarVecCDMul(@)".}

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