/** 
 * ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
 * Source file: src/eigen/eigen.h
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

#include "eigenscalar.h"
#include "eigenmatrix.h"
#include "eigenvector.h"

#pragma once

#define MatrixTemplateArgs Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor
#define MapTemplateArgs Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>
#define VectorStride Eigen::InnerStride<Eigen::Dynamic>

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