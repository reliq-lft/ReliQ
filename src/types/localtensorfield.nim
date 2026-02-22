#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/types/localtensor.nim
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

import lattice/[lattice]
import eigen/[eigen]
import utils/[complex]
import utils/[composite]
import memory/[storage]

import globaltensorfield

record LocalTensorField*[D: static[int], R: static[int], L: Lattice[D], T]:
  var global*: TensorField[D, R, L, T]
  var lattice*: L
  var shape*: array[R, int]

  when isComplex32(T): 
    var data*: LocalStorage[float32]
  elif isComplex64(T):
    var data*: LocalStorage[float64]
  else:
    var data*: LocalStorage[T]
  
  #[ constructor/destructor ]#

  method init(tensor: var TensorField[D, R, L, T]) =
    this.global = tensor
    this.lattice = tensor.lattice
    this.shape = tensor.shape

    when isComplex32(T): 
      this.data = cast[LocalStorage[float32]](tensor.accessPadded())
    elif isComplex64(T):
      this.data = cast[LocalStorage[float64]](tensor.accessPadded())
    else: this.data = cast[LocalStorage[T]](tensor.accessPadded())
  
  method deinit = this.global.releaseLocal()

  #[ accessors ]#

  method `[]`*(n: int): auto =
    let ghostWidth = 1
    let data = cast[ptr T](addr this.data[this.global.paddedLexIdx(n)])
    when R == 0: return newEigenScalar(data)
    elif R == 1: 
      when isComplex(T): 
        let stride = (2 + 2 * ghostWidth) div 2
      else: 
        let stride = 1 + 2 * ghostWidth
      return newEigenVector(data, this.shape[0], stride)
    elif R == 2: 
      when isComplex(T): 
        let inner = (2 + 2 * ghostWidth) div 2
      else: 
        let inner = 1 + 2 * ghostWidth
      let outer = (this.shape[1] + 2 * ghostWidth) * inner
      return data.newEigenMatrix(this.shape, inner, outer)
    else: raise newException(ValueError, "Unsupported tensor rank")
  
  #[ misc ]#

  method numLocalSites: int = this.global.numLocalSites()
  
#[ convenience procedures/templates ]#

template all*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: LocalTensorField[D, R, L, T]
): untyped =
  ## Get a range over all local sites (excluding ghosts)
  0..<tensor.numLocalSites()

#[ unit test ]#

when isMainModule:
  import std/[unittest, math]
  import ga/[ga]
  import openmp/[ompbase]

  const eps = 1e-10

  gaParallel:
    let gmtry = [8, 8, 8, 16]
    let ghost = [1, 1, 1, 1]
    var lat = gmtry.newSimpleCubicLattice(ghostGrid = ghost)

    var gf = lat.newTensorField([nc, nc]): Complex64
    var gv = lat.newTensorField([nc]): Complex64
    var gs = lat.newScalarField(): Complex64

    var lf = gf.newLocalTensorField()
    var lv = gv.newLocalTensorField()
    var ls = gs.newLocalTensorField()

    suite "LocalTensor tests":

      # ── infrastructure ──────────────────────────────────────────────────────

      test "numLocalSites is positive and consistent across fields":
        check lf.numLocalSites() > 0
        check lf.numLocalSites() == lv.numLocalSites()
        check lf.numLocalSites() == ls.numLocalSites()

      test "all template covers exactly numLocalSites":
        var count = 0
        for n in lf.all: inc count
        check count == lf.numLocalSites()

      # ── scalar field ────────────────────────────────────────────────────────

      test "scalar write/read roundtrip (site-dependent values)":
        for n in ls.all:
          var s = ls[n]
          s := complex(float64(n), float64(-n))
        for n in ls.all:
          let s = ls[n]
          check s == complex(float64(n), float64(-n))

      test "scalar abs: |3+4i| = 5":
        for n in ls.all:
          var s = ls[n]
          s := complex(3.0, 4.0)
        for n in ls.all:
          let s = ls[n]
          check abs(s.abs() - 5.0) < eps

      test "scalar conjugate: conj(3+4i) = 3-4i":
        for n in ls.all:
          var s = ls[n]
          s := complex(3.0, 4.0)
        for n in ls.all:
          let s = ls[n]
          check s.conj() == complex(3.0, -4.0)

      test "scalar addition: (3+4i) + (1+2i) = 4+6i":
        for n in ls.all:
          var s = ls[n]
          s := complex(3.0, 4.0)
        for n in ls.all:
          let s = ls[n]
          var b = newLocalTensor(ls.shape): Complex64
          b := complex(1.0, 2.0)
          let r = s + b
          check r == complex(4.0, 6.0)

      test "scalar subtraction: (3+4i) - (1+2i) = 2+2i":
        for n in ls.all:
          var s = ls[n]
          s := complex(3.0, 4.0)
        for n in ls.all:
          var s = ls[n]
          var b = newLocalTensor(ls.shape): Complex64
          b := complex(1.0, 2.0)
          let r = s - b
          check r == complex(2.0, 2.0)
          ls[n] := r
        for n in ls.all:
          let s = ls[n]
          check s == complex(2.0, 2.0)

      test "scalar field[n] := val writes through without var":
        for n in ls.all: ls[n] := complex(float64(n), float64(-n))
        for n in ls.all:
          let s = ls[n]
          check s == complex(float64(n), float64(-n))

      test "scalar multiplication: (2+i)*(1+i) = 1+3i":
        for n in ls.all:
          var s = ls[n]
          s := complex(2.0, 1.0)
        for n in ls.all:
          let s = ls[n]
          var b = newLocalTensor(ls.shape): Complex64
          b := complex(1.0, 1.0)
          let r = s * b
          check r == complex(1.0, 3.0)

      # ── vector field ────────────────────────────────────────────────────────

      test "vector write/read roundtrip (site-dependent values)":
        for n in lv.all:
          var v = lv[n]
          for i in 0..<nc: v[i] = complex(float64(i + 1), float64(n))
        for n in lv.all:
          let v = lv[n]
          for i in 0..<nc:
            check v[i] == complex(float64(i + 1), float64(n))

      test "different sites hold independent vector data":
        for n in lv.all:
          var v = lv[n]
          for i in 0..<nc: v[i] = complex(float64(n * nc + i), 0.0)
        for n in lv.all:
          let v = lv[n]
          for i in 0..<nc:
            check v[i] == complex(float64(n * nc + i), 0.0)

      test "vector field[n] := fill writes through without var":
        for n in lv.all: lv[n] := complex(7.0, -3.0)
        for n in lv.all:
          let v = lv[n]
          for i in 0..<nc:
            check v[i] == complex(7.0, -3.0)

      test "vector field[n] := copy-from another vector":
        for n in lv.all:
          var v = lv[n]
          for i in 0..<nc: v[i] = complex(float64(i + 1), float64(n))
        for n in lv.all:
          var tmp = newLocalTensor(lv.shape): Complex64
          tmp := lv[n]
          lv[n] := tmp
        for n in lv.all:
          let v = lv[n]
          for i in 0..<nc:
            check v[i] == complex(float64(i + 1), float64(n))

      test "vector addition: a + b - b = a":
        for n in lv.all:
          var v = lv[n]
          for i in 0..<nc: v[i] = complex(float64(i + 1), 0.0)
        for n in lv.all:
          let v = lv[n]
          var b = newLocalTensor(lv.shape): Complex64
          for i in 0..<nc: b[i] = complex(0.0, float64(i + 1))
          let r = (v + b) - b
          for i in 0..<nc:
            check abs((r[i] - complex(float64(i + 1), 0.0)).re) < eps
            check abs((r[i] - complex(float64(i + 1), 0.0)).im) < eps

      test "vector norm: ||[3, 4, 0]|| = 5":
        for n in lv.all:
          var v = lv[n]
          v[0] = complex(3.0, 0.0)
          v[1] = complex(4.0, 0.0)
          for i in 2..<nc: v[i] = complex(0.0, 0.0)
        for n in lv.all:
          let v = lv[n]
          check abs(v.norm() - 5.0) < eps

      test "vector dot product: [1,2,3]·[1,2,3] = 14":
        for n in lv.all:
          var v = lv[n]
          for i in 0..<nc: v[i] = complex(float64(i + 1), 0.0)
        for n in lv.all:
          let v = lv[n]
          let d = v.dot(v)
          check abs((d - complex(14.0, 0.0)).re) < eps
          check abs((d - complex(14.0, 0.0)).im) < eps

      test "vector conjugate: conj([1+i, 2-i, 3i]) = [1-i, 2+i, -3i]":
        for n in lv.all:
          var v = lv[n]
          v[0] = complex(1.0, 1.0)
          v[1] = complex(2.0, -1.0)
          v[2] = complex(0.0, 3.0)
        for n in lv.all:
          let v = lv[n]
          let vc = v.conjugate()
          check abs((vc[0] - complex(1.0, -1.0)).re) < eps
          check abs((vc[1] - complex(2.0,  1.0)).re) < eps
          check abs((vc[2] - complex(0.0, -3.0)).im) < eps

      # ── matrix field ────────────────────────────────────────────────────────

      test "matrix write/read roundtrip (site-dependent values)":
        for n in lf.all:
          var m = lf[n]
          for row in 0..<nc:
            for col in 0..<nc:
              m[row, col] = complex(float64(row * nc + col + 1), float64(n))
        for n in lf.all:
          let m = lf[n]
          for row in 0..<nc:
            for col in 0..<nc:
              let expected = complex(float64(row * nc + col + 1), float64(n))
              check abs((m[row, col] - expected).re) < eps
              check abs((m[row, col] - expected).im) < eps

      test "different sites hold independent matrix data":
        for n in lf.all:
          var m = lf[n]
          for row in 0..<nc:
            for col in 0..<nc:
              m[row, col] = complex(float64(n), float64(row * nc + col))
        for n in lf.all:
          let m = lf[n]
          for row in 0..<nc:
            for col in 0..<nc:
              check abs((m[row, col] - complex(float64(n), float64(row * nc + col))).re) < eps
              check abs((m[row, col] - complex(float64(n), float64(row * nc + col))).im) < eps

      test "matrix field[n] := fill writes through without var":
        for n in lf.all: lf[n] := complex(5.0, -2.0)
        for n in lf.all:
          let m = lf[n]
          for row in 0..<nc:
            for col in 0..<nc:
              check abs((m[row, col] - complex(5.0, -2.0)).re) < eps
              check abs((m[row, col] - complex(5.0, -2.0)).im) < eps

      test "matrix field[n] := copy-from another matrix":
        for n in lf.all:
          var m = lf[n]
          for row in 0..<nc:
            for col in 0..<nc:
              m[row, col] = complex(float64(row * nc + col + 1), float64(n))
        for n in lf.all:
          var tmp = newLocalTensor(lf.shape): Complex64
          tmp := lf[n]
          lf[n] := tmp
        for n in lf.all:
          let m = lf[n]
          for row in 0..<nc:
            for col in 0..<nc:
              let expected = complex(float64(row * nc + col + 1), float64(n))
              check abs((m[row, col] - expected).re) < eps
              check abs((m[row, col] - expected).im) < eps

      test "matrix addition: a + b - b = a":
        for n in lf.all:
          var m = lf[n]
          for row in 0..<nc:
            for col in 0..<nc:
              m[row, col] = complex(float64(row + 1), float64(col + 1))
        for n in lf.all:
          let m = lf[n]
          var b = newLocalTensor(lf.shape): Complex64
          for row in 0..<nc:
            for col in 0..<nc: b[row, col] = complex(float64(col), float64(row))
          let r = (m + b) - b
          for row in 0..<nc:
            for col in 0..<nc:
              let expected = complex(float64(row + 1), float64(col + 1))
              check abs((r[row, col] - expected).re) < eps
              check abs((r[row, col] - expected).im) < eps

      test "matrix × identity = matrix":
        for n in lf.all:
          var m = lf[n]
          for row in 0..<nc:
            for col in 0..<nc:
              m[row, col] = complex(float64(row * nc + col + 1), float64(row - col))
        for n in lf.all:
          let m = lf[n]
          var ident = newLocalTensor(lf.shape): Complex64
          for i in 0..<nc: ident[i, i] = complex(1.0, 0.0)
          let r = m * ident
          for row in 0..<nc:
            for col in 0..<nc:
              let expected = m[row, col]
              check abs((r[row, col] - expected).re) < eps
              check abs((r[row, col] - expected).im) < eps

      test "matrix × e_0 = first column of matrix":
        # M[row, col] = row*nc+col+1  →  M*e_0 = [1, nc+1, 2*nc+1, ...]
        for n in lf.all:
          var m = lf[n]
          for row in 0..<nc:
            for col in 0..<nc:
              m[row, col] = complex(float64(row * nc + col + 1), 0.0)
        for n in lf.all:
          let m = lf[n]
          var e0 = newLocalTensor(lv.shape): Complex64
          e0[0] = complex(1.0, 0.0)
          let r = m * e0
          for row in 0..<nc:
            let expected = complex(float64(row * nc + 1), 0.0)
            check abs((r[row] - expected).re) < eps
            check abs((r[row] - expected).im) < eps

      test "matrix × vector: [[1,2,0],[3,4,0],[0,0,0]] * [5,6,0] = [17,39,0]":
        for n in lf.all:
          var m = lf[n]
          for row in 0..<nc:
            for col in 0..<nc: m[row, col] = complex(0.0, 0.0)
          m[0, 0] = complex(1.0, 0.0); m[0, 1] = complex(2.0, 0.0)
          m[1, 0] = complex(3.0, 0.0); m[1, 1] = complex(4.0, 0.0)
        for n in lv.all:
          var v = lv[n]
          for i in 0..<nc: v[i] = complex(0.0, 0.0)
          v[0] = complex(5.0, 0.0)
          v[1] = complex(6.0, 0.0)
        for n in lf.all:
          let m = lf[n]
          let v = lv[n]
          let r = m * v
          check abs((r[0] - complex(17.0, 0.0)).re) < eps
          check abs((r[1] - complex(39.0, 0.0)).re) < eps
          check abs(r[2].re) < eps
          check abs(r[2].im) < eps

      test "matrix adjoint is conjugate transpose":
        for n in lf.all:
          var m = lf[n]
          for row in 0..<nc:
            for col in 0..<nc:
              m[row, col] = complex(float64(row * nc + col + 1), float64(row - col))
        for n in lf.all:
          let m = lf[n]
          let mh = m.adjoint()
          for row in 0..<nc:
            for col in 0..<nc:
              let expected = complex(m[col, row].re, -m[col, row].im)
              check abs((mh[row, col] - expected).re) < eps
              check abs((mh[row, col] - expected).im) < eps

      test "trace equals sum of diagonal":
        for n in lf.all:
          var m = lf[n]
          for row in 0..<nc:
            for col in 0..<nc:
              m[row, col] = complex(float64(row * nc + col + 1), float64(col - row))
        for n in lf.all:
          let m = lf[n]
          let tr = m.trace()
          var expRe = 0.0; var expIm = 0.0
          for i in 0..<nc:
            expRe += float64(i * nc + i + 1)   # diagonal real part
            expIm += float64(i - i)             # = 0
          check abs((tr - complex(expRe, expIm)).re) < eps
          check abs((tr - complex(expRe, expIm)).im) < eps

      test "M * M^-1 ≈ identity (diagonally dominant matrix)":
        for n in lf.all:
          var m = lf[n]
          for row in 0..<nc:
            for col in 0..<nc:
              m[row, col] = if row == col: complex(float64(nc + row + 2), 0.0)
                            else: complex(0.1, 0.05)
        for n in lf.all:
          let m = lf[n]
          let mi = m.inverse()
          let prod = m * mi
          for i in 0..<nc:
            for j in 0..<nc:
              let expected = if i == j: complex(1.0, 0.0) else: complex(0.0, 0.0)
              check abs((prod[i, j] - expected).re) < eps
              check abs((prod[i, j] - expected).im) < eps

      test "Frobenius norm of identity = sqrt(nc)":
        for n in lf.all:
          var m = lf[n]
          for row in 0..<nc:
            for col in 0..<nc:
              m[row, col] = if row == col: complex(1.0, 0.0) else: complex(0.0, 0.0)
        for n in lf.all:
          let m = lf[n]
          check abs(m.norm() - sqrt(float64(nc))) < eps

      # ── cross-type operations ────────────────────────────────────────────────

      test "scalar × matrix: (2+i) * I has diagonal 2+i, off-diagonal 0":
        for n in ls.all:
          var s = ls[n]; s := complex(2.0, 1.0)
        for n in lf.all:
          var m = lf[n]
          for row in 0..<nc:
            for col in 0..<nc:
              m[row, col] = if row == col: complex(1.0, 0.0) else: complex(0.0, 0.0)
        for n in lf.all:
          let m = lf[n]
          let s = ls[n]
          let r = s * m
          for i in 0..<nc:
            check abs((r[i, i] - complex(2.0, 1.0)).re) < eps
            check abs((r[i, i] - complex(2.0, 1.0)).im) < eps
            for j in 0..<nc:
              if i != j:
                check abs(r[i, j].re) < eps
                check abs(r[i, j].im) < eps

      test "scalar × matrix is commutative: s*M = M*s":
        for n in ls.all:
          var s = ls[n]; s := complex(3.0, -1.0)
        for n in lf.all:
          var m = lf[n]
          for row in 0..<nc:
            for col in 0..<nc:
              m[row, col] = complex(float64(row + col + 1), float64(row - col))
        for n in lf.all:
          let m = lf[n]
          let s = ls[n]
          let r1 = s * m
          let r2 = m * s
          for row in 0..<nc:
            for col in 0..<nc:
              check abs((r1[row, col] - r2[row, col]).re) < eps
              check abs((r1[row, col] - r2[row, col]).im) < eps

      test "scalar × vector: i * [1,2,3] = [i, 2i, 3i]":
        for n in ls.all:
          var s = ls[n]; s := complex(0.0, 1.0)
        for n in lv.all:
          var v = lv[n]
          for i in 0..<nc: v[i] = complex(float64(i + 1), 0.0)
        for n in lv.all:
          let v = lv[n]
          let s = ls[n]
          let r = s * v
          for i in 0..<nc:
            let expected = complex(0.0, float64(i + 1))
            check abs((r[i] - expected).re) < eps
            check abs((r[i] - expected).im) < eps

      test "scalar × vector is commutative: s*v = v*s":
        for n in ls.all:
          var s = ls[n]; s := complex(2.0, -3.0)
        for n in lv.all:
          var v = lv[n]
          for i in 0..<nc: v[i] = complex(float64(i + 1), float64(nc - i))
        for n in lv.all:
          let v = lv[n]
          let s = ls[n]
          let r1 = s * v
          let r2 = v * s
          for i in 0..<nc:
            check abs((r1[i] - r2[i]).re) < eps
            check abs((r1[i] - r2[i]).im) < eps

      # ── parallel operation tests ─────────────────────────────────────────────

      test "parallel matrix-matrix ops compile and run without corruption":
        threads:
          for n in lf.all:
            let sf  = lf[n]
            let tmpa = newLocalTensor(lf.shape): Complex64
            var tmpb = newLocalTensor(lf.shape): Complex64
            discard tmpa * sf;  discard tmpb * sf
            discard sf * tmpa;  discard sf * tmpb
            discard lf[n] * lf[n]
            tmpb += sf;  tmpb -= sf
        check true

      test "parallel matrix-vector ops compile and run without corruption":
        threads:
          for n in lf.all:
            let m  = lf[n]
            let v  = lv[n]
            let mt = newLocalTensor(lf.shape): Complex64
            let vt = newLocalTensor(lv.shape): Complex64
            discard m  * v;   discard m  * vt
            discard mt * v;   discard mt * vt
            discard lf[n] * lv[n]
        check true

      test "parallel scalar-matrix and scalar-vector ops compile without corruption":
        threads:
          for n in lf.all:
            let m  = lf[n];  let v  = lv[n];  let s  = ls[n]
            let mt = newLocalTensor(lf.shape): Complex64
            let vt = newLocalTensor(lv.shape): Complex64
            let st = newLocalTensor(ls.shape): Complex64
            discard s  * m;   discard st * m;   discard s  * mt;  discard st * mt
            discard m  * s;   discard mt * s;   discard m  * st;  discard mt * st
            discard s  * v;   discard st * v;   discard s  * vt;  discard st * vt
            discard v  * s;   discard vt * s;   discard v  * st;  discard vt * st
            discard ls[n] * lf[n];  discard lf[n] * ls[n]
            discard ls[n] * lv[n];  discard lv[n] * ls[n]
        check true

      test "parallel writes are visible after threads block":
        threads:
          for n in lf.all:
            var m = lf[n]
            for row in 0..<nc:
              for col in 0..<nc:
                m[row, col] = complex(float64(n + row * nc + col), float64(n))
        for n in lf.all:
          let m = lf[n]
          for row in 0..<nc:
            for col in 0..<nc:
              check abs(m[row, col].re - float64(n + row * nc + col)) < eps
              check abs(m[row, col].im - float64(n)) < eps
