#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/types/dual.nim
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

import complex
import std/math

type Dual*[T] = object
  re*: T
  du*: T

proc dual*[T](re, du: T): Dual[T] = Dual[T](re: re, du: du)

proc dual*[T](re: T): Dual[T] =
  ## Lift a scalar into a dual number with du = 1 (autodiff seed for variable re).
  when T is float32: Dual[T](re: re, du: 1.0'f32)
  elif T is float64: Dual[T](re: re, du: 1.0)
  elif isComplex32(T): Dual[T](re: re, du: complex(1.0'f32, 0.0'f32))
  elif isComplex64(T): Dual[T](re: re, du: complex(1.0, 0.0))

proc `+`*[T](x: Dual[T], y: Dual[T]): Dual[T] =
  return Dual[T](re: x.re + y.re, du: x.du + y.du)

proc `-`*[T](x: Dual[T], y: Dual[T]): Dual[T] =
  return Dual[T](re: x.re - y.re, du: x.du - y.du)

proc `*`*[T](x: Dual[T], y: Dual[T]): Dual[T] =
  return Dual[T](re: x.re * y.re, du: x.re * y.du + x.du * y.re)

proc `/`*[T](x: Dual[T], y: Dual[T]): Dual[T] =
  return Dual[T](re: x.re / y.re, du: (x.du * y.re - x.re * y.du) / (y.re * y.re))

proc conjugate*[T](x: Dual[T]): Dual[T] =
  ## Complex-conjugate both components: (w + uε)* = w̄ + ūε.
  ## For real T this is the identity.
  when isComplex(T): return Dual[T](re: conjugate(x.re), du: conjugate(x.du))
  else: return Dual[T](re: x.re, du: x.du)

#[ unary negation ]#

proc `-`*[T](x: Dual[T]): Dual[T] =
  Dual[T](re: -x.re, du: -x.du)

#[ mixed scalar / dual arithmetic ]#

proc `+`*[T](s: T, x: Dual[T]): Dual[T] = Dual[T](re: s + x.re, du: x.du)
proc `+`*[T](x: Dual[T], s: T): Dual[T] = Dual[T](re: x.re + s, du: x.du)
proc `-`*[T](s: T, x: Dual[T]): Dual[T] = Dual[T](re: s - x.re, du: -x.du)
proc `-`*[T](x: Dual[T], s: T): Dual[T] = Dual[T](re: x.re - s, du: x.du)
proc `*`*[T](s: T, x: Dual[T]): Dual[T] = Dual[T](re: s * x.re, du: s * x.du)
proc `*`*[T](x: Dual[T], s: T): Dual[T] = Dual[T](re: x.re * s, du: x.du * s)
proc `/`*[T](x: Dual[T], s: T): Dual[T] = Dual[T](re: x.re / s, du: x.du / s)

#[ compound assignment ]#

proc `+=`*[T](x: var Dual[T], y: Dual[T]) = x = x + y
proc `-=`*[T](x: var Dual[T], y: Dual[T]) = x = x - y
proc `*=`*[T](x: var Dual[T], y: Dual[T]) = x = x * y
proc `/=`*[T](x: var Dual[T], y: Dual[T]) = x = x / y

proc `+=`*[T](x: var Dual[T], s: T) = x = x + s
proc `-=`*[T](x: var Dual[T], s: T) = x = x - s
proc `*=`*[T](x: var Dual[T], s: T) = x = x * s
proc `/=`*[T](x: var Dual[T], s: T) = x = x / s

#[ equality ]#

proc `==`*[T](x, y: Dual[T]): bool = x.re == y.re and x.du == y.du

#[ dual conjugate and Hermitian adjoint ]#

proc dualConj*[T](x: Dual[T]): Dual[T] =
  ## Dual conjugate: negate the ε part — w + uε → w − uε.
  Dual[T](re: x.re, du: -x.du)

proc adj*[T](x: Dual[T]): Dual[T] =
  ## Hermitian adjoint: complex-conjugate both parts and negate the dual part.
  ## For real T this reduces to dualConj.
  when isComplex(T): Dual[T](re: conjugate(x.re), du: -conjugate(x.du))
  else: dualConj(x)

#[ squared norm  (x · conj(x)) ]#

proc norm2*[T](x: Dual[T]): Dual[T] =
  ## x · conj(x).
  ## Real T:     w² + 2·w·u·ε
  ## Complex T:  |w|² + 2·Re(w·ū)·ε  (both components are real-valued)
  x * conjugate(x)

#[ string representation ]#

proc `$`*[T](x: Dual[T]): string =
  "(" & $x.re & " + " & $x.du & "\xce\xb5)"   # ε = U+03B5

#[ type predicates ]#

template isDual*(T: typedesc): bool = T is Dual
template isDualReal32*(T: typedesc): bool = T is Dual[float32]
template isDualReal64*(T: typedesc): bool = T is Dual[float64]
template isDualComplex32*(T: typedesc): bool = T is Dual[Complex32]
template isDualComplex64*(T: typedesc): bool = T is Dual[Complex64]

#[ AD elementary functions
   All follow the chain rule:  f(w + u·ε) = f(w) + f′(w)·u·ε
   Works for both real and complex T because std/math and std/complex
   provide matching overloads for every function below.               ]#

proc sqrt*[T](x: Dual[T]): Dual[T] =
  ## √(w + uε) = √w + u/(2√w)·ε
  let s = sqrt(x.re)
  Dual[T](re: s, du: x.du / (s + s))

proc exp*[T](x: Dual[T]): Dual[T] =
  ## exp(w + uε) = eʷ·(1 + u·ε)
  let e = exp(x.re)
  Dual[T](re: e, du: e * x.du)

proc ln*[T](x: Dual[T]): Dual[T] =
  ## ln(w + uε) = ln w + (u/w)·ε
  Dual[T](re: ln(x.re), du: x.du / x.re)

proc sin*[T](x: Dual[T]): Dual[T] =
  ## sin(w + uε) = sin w + cos(w)·u·ε
  Dual[T](re: sin(x.re), du: cos(x.re) * x.du)

proc cos*[T](x: Dual[T]): Dual[T] =
  ## cos(w + uε) = cos w − sin(w)·u·ε
  Dual[T](re: cos(x.re), du: -sin(x.re) * x.du)

proc tan*[T](x: Dual[T]): Dual[T] =
  ## tan(w + uε) = tan w + sec²(w)·u·ε
  let c = cos(x.re)
  Dual[T](re: tan(x.re), du: x.du / (c * c))

proc sinh*[T](x: Dual[T]): Dual[T] =
  ## sinh(w + uε) = sinh w + cosh(w)·u·ε
  Dual[T](re: sinh(x.re), du: cosh(x.re) * x.du)

proc cosh*[T](x: Dual[T]): Dual[T] =
  ## cosh(w + uε) = cosh w + sinh(w)·u·ε
  Dual[T](re: cosh(x.re), du: sinh(x.re) * x.du)

proc tanh*[T](x: Dual[T]): Dual[T] =
  ## tanh(w + uε) = tanh w + sech²(w)·u·ε
  let c = cosh(x.re)
  Dual[T](re: tanh(x.re), du: x.du / (c * c))

proc pow*[T](x: Dual[T], p: T): Dual[T] =
  ## wᵖ + p·wᵖ⁻¹·u·ε  (written as (p/w)·wᵖ·u·ε to avoid a second pow call)
  let wp = pow(x.re, p)
  Dual[T](re: wp, du: (p / x.re) * wp * x.du)

proc abs*[T: SomeFloat](x: Dual[T]): Dual[T] =
  ## |w + uε| = |w| + sgn(w)·u·ε  (real T only; use norm2 for complex T)
  let a = abs(x.re)
  Dual[T](re: a, du: (x.re / a) * x.du)

when isMainModule:
  import std/[unittest]
  import std/[complex]

  # ── helpers ─────────────────────────────────────────────────────────────────
  # std/math provides almostEqual for SomeFloat (ULP-based).
  # We only need a thin wrapper for Complex64 since there is no stdlib overload.

  proc almostEqual(a, b: Complex64): bool =
    almostEqual(a.re, b.re) and almostEqual(a.im, b.im)

  # ── suites ──────────────────────────────────────────────────────────────────

  suite "Dual[float64] — constructors":

    test "two-argument constructor":
      let x = dual(3.0, 4.0)
      check x.re == 3.0
      check x.du == 4.0

    test "single-argument seed (du = 1)":
      let x = dual(5.0)
      check x.re == 5.0
      check x.du == 1.0

    test "equality":
      check dual(1.0, 2.0) == dual(1.0, 2.0)
      check not (dual(1.0, 2.0) == dual(1.0, 3.0))

    test "string representation":
      let s = $dual(2.0, -1.0)
      check s == "(2.0 + -1.0ε)"

  suite "Dual[float64] — arithmetic":

    test "addition":
      let r = dual(1.0, 2.0) + dual(3.0, 4.0)
      check r == dual(4.0, 6.0)

    test "subtraction":
      let r = dual(5.0, 3.0) - dual(2.0, 1.0)
      check r == dual(3.0, 2.0)

    test "multiplication  (w₁w₂) + (w₁u₂ + u₁w₂)ε":
      # (2 + 3ε)(4 + 5ε) = 8 + (10 + 12)ε = 8 + 22ε
      let r = dual(2.0, 3.0) * dual(4.0, 5.0)
      check r == dual(8.0, 22.0)

    test "division":
      # (6 + 4ε) / (2 + 1ε) = 3 + (4·2 - 6·1)/4·ε = 3 + 0.5ε
      let r = dual(6.0, 4.0) / dual(2.0, 1.0)
      check almostEqual(r.re, 3.0)
      check almostEqual(r.du, 0.5)

    test "unary negation":
      let r = -dual(1.0, 2.0)
      check r == dual(-1.0, -2.0)

    test "scalar left/right addition":
      check (2.0 + dual(1.0, 3.0)) == dual(3.0, 3.0)
      check (dual(1.0, 3.0) + 2.0) == dual(3.0, 3.0)

    test "scalar left/right subtraction":
      check (5.0 - dual(1.0, 3.0)) == dual(4.0, -3.0)
      check (dual(5.0, 3.0) - 1.0) == dual(4.0, 3.0)

    test "scalar multiplication":
      check (3.0 * dual(2.0, 4.0)) == dual(6.0, 12.0)
      check (dual(2.0, 4.0) * 3.0) == dual(6.0, 12.0)

    test "scalar division":
      let r = dual(6.0, 4.0) / 2.0
      check r == dual(3.0, 2.0)

    test "compound assignment +=/-=/*=//= (dual rhs)":
      var x = dual(2.0, 1.0)
      x += dual(1.0, 2.0); check x == dual(3.0, 3.0)
      x -= dual(1.0, 1.0); check x == dual(2.0, 2.0)
      x *= dual(2.0, 0.0); check x == dual(4.0, 4.0)
      x /= dual(2.0, 0.0); check x == dual(2.0, 2.0)

    test "compound assignment +=/-=/*=//= (scalar rhs)":
      var x = dual(4.0, 2.0)
      x += 1.0; check x == dual(5.0, 2.0)
      x -= 1.0; check x == dual(4.0, 2.0)
      x *= 2.0; check x == dual(8.0, 4.0)
      x /= 2.0; check x == dual(4.0, 2.0)

  suite "Dual[float64] — conjugations and norm":

    test "conjugate is identity for real T":
      let x = dual(3.0, 4.0)
      check conjugate(x) == x

    test "dualConj negates ε part":
      check dualConj(dual(3.0, 4.0)) == dual(3.0, -4.0)

    test "adj == dualConj for real T":
      let x = dual(3.0, 4.0)
      check adj(x) == dualConj(x)

    test "norm2 = x * conjugate(x)  →  w² + 2wuε":
      # (3 + 2ε)·(3 + 2ε) = 9 + 12ε
      let x = dual(3.0, 2.0)
      let n = norm2(x)
      check almostEqual(n.re, 9.0)
      check almostEqual(n.du, 12.0)

    test "abs (positive argument)":
      let r = abs(dual(3.0, -2.0))
      check almostEqual(r.re, 3.0)
      check almostEqual(r.du, -2.0)   # sgn(3)·(-2) = -2

    test "abs (negative argument)":
      let r = abs(dual(-3.0, 2.0))
      check almostEqual(r.re, 3.0)
      check almostEqual(r.du, -2.0)   # sgn(-3)·2 = -2

  suite "Dual[float64] — AD elementary functions":
    # Each test seeds dual(w, 1) so the dual part equals f′(w).

    test "sqrt:  d/dw √w = 1/(2√w)":
      let w = 4.0
      let r = sqrt(dual(w, 1.0))
      check almostEqual(r.re, 2.0)
      check almostEqual(r.du, 0.25)

    test "exp:  d/dw eʷ = eʷ":
      let w = 1.0
      let r = exp(dual(w, 1.0))
      check almostEqual(r.re, E)
      check almostEqual(r.du, E)

    test "ln:  d/dw ln w = 1/w":
      let w = 2.0
      let r = ln(dual(w, 1.0))
      check almostEqual(r.re, ln(2.0))
      check almostEqual(r.du, 0.5)

    test "sin:  d/dw sin w = cos w":
      let w = PI / 6.0
      let r = sin(dual(w, 1.0))
      check almostEqual(r.re, 0.5)
      check almostEqual(r.du, cos(w))

    test "cos:  d/dw cos w = -sin w":
      let w = PI / 3.0
      let r = cos(dual(w, 1.0))
      check almostEqual(r.re, 0.5)
      check almostEqual(r.du, -sin(w))

    test "tan:  d/dw tan w = sec²w":
      let w = PI / 4.0
      let r = tan(dual(w, 1.0))
      check almostEqual(r.re, 1.0)
      check almostEqual(r.du, 1.0 / (cos(w) * cos(w)))

    test "sinh:  d/dw sinh w = cosh w":
      let w = 1.0
      let r = sinh(dual(w, 1.0))
      check almostEqual(r.re, sinh(1.0))
      check almostEqual(r.du, cosh(1.0))

    test "cosh:  d/dw cosh w = sinh w":
      let w = 1.0
      let r = cosh(dual(w, 1.0))
      check almostEqual(r.re, cosh(1.0))
      check almostEqual(r.du, sinh(1.0))

    test "tanh:  d/dw tanh w = sech²w":
      let w = 1.0
      let r = tanh(dual(w, 1.0))
      check almostEqual(r.re, tanh(1.0))
      check almostEqual(r.du, 1.0 / (cosh(1.0) * cosh(1.0)))

    test "pow:  d/dw wᵖ = p·wᵖ⁻¹":
      let w = 2.0; let p = 3.0
      let r = pow(dual(w, 1.0), p)
      check almostEqual(r.re, 8.0)
      check almostEqual(r.du, 12.0)   # 3·2² = 12

    test "chain rule: d/dw exp(sin w)":
      # derivative = cos(w)·exp(sin(w))
      let w = PI / 6.0
      let r = exp(sin(dual(w, 1.0)))
      check almostEqual(r.re, exp(sin(w)))
      check almostEqual(r.du, cos(w) * exp(sin(w)))

    test "product rule: d/dw w·exp(w) = (1+w)·exp(w)":
      let w = 2.0
      let x = dual(w, 1.0)
      let r = x * exp(x)
      check almostEqual(r.re, w * exp(w))
      check almostEqual(r.du, (1.0 + w) * exp(w))

    test "quotient rule: d/dw sin(w)/w = (w·cos(w) - sin(w))/w²":
      let w = 1.0
      let x = dual(w, 1.0)
      let r = sin(x) / x
      let expected = (w * cos(w) - sin(w)) / (w * w)
      check almostEqual(r.du, expected)

  suite "Dual[Complex64] — constructors":

    test "two-argument constructor":
      let w = complex(1.0, 2.0)
      let u = complex(3.0, 4.0)
      let x = dual(w, u)
      check x.re == w
      check x.du == u

    test "single-argument seed (du = 1+0i)":
      let w = complex(2.0, -1.0)
      let x = dual(w)
      check x.re == w
      check x.du == complex(1.0, 0.0)

  suite "Dual[Complex64] — arithmetic":

    test "addition":
      let a = dual(complex(1.0, 2.0), complex(3.0, 4.0))
      let b = dual(complex(5.0, 6.0), complex(7.0, 8.0))
      let r = a + b
      check r.re == complex(6.0, 8.0)
      check r.du == complex(10.0, 12.0)

    test "subtraction":
      let a = dual(complex(5.0, 6.0), complex(7.0, 8.0))
      let b = dual(complex(1.0, 2.0), complex(3.0, 4.0))
      let r = a - b
      check r.re == complex(4.0, 4.0)
      check r.du == complex(4.0, 4.0)

    test "multiplication  Leibniz rule":
      # a = (1+i) + (2+0i)ε,  b = (0+i) + (1+i)ε
      # re: (1+i)(0+i) = -1+i
      # du: (1+i)(1+i) + (2)(0+i) = 2i + 2i = (0+4i)... recalc:
      #   w₁u₂ = (1+i)(1+i) = 1+2i-1 = 2i
      #   u₁w₂ = (2)(i)    = 2i
      #   → du = 4i
      let a = dual(complex(1.0, 1.0), complex(2.0, 0.0))
      let b = dual(complex(0.0, 1.0), complex(1.0, 1.0))
      let r = a * b
      check almostEqual(r.re, complex(-1.0, 1.0))
      check almostEqual(r.du, complex(0.0, 4.0))

    test "division":
      # (w + uε) / (w + 0ε)  →  re = 1, du = 0 - w/(w²) = -1/w... general:
      # Let a = (2+0i)+(4+0i)ε, b = (2+0i)+(0+0i)ε  → result = (1+0i)+(2+0i)ε
      let a = dual(complex(2.0, 0.0), complex(4.0, 0.0))
      let b = dual(complex(2.0, 0.0), complex(0.0, 0.0))
      let r = a / b
      check almostEqual(r.re, complex(1.0, 0.0))
      check almostEqual(r.du, complex(2.0, 0.0))

  suite "Dual[Complex64] — conjugations":

    test "conjugate conjugates both components":
      let x = dual(complex(1.0, 2.0), complex(3.0, -4.0))
      let c = conjugate(x)
      check c.re == complex(1.0, -2.0)
      check c.du == complex(3.0,  4.0)

    test "dualConj negates ε only":
      let x = dual(complex(1.0, 2.0), complex(3.0, -4.0))
      let d = dualConj(x)
      check d.re == complex(1.0,  2.0)
      check d.du == complex(-3.0, 4.0)

    test "adj = conj + dualConj combined":
      let x = dual(complex(1.0, 2.0), complex(3.0, -4.0))
      let a = adj(x)
      check a.re == complex(1.0, -2.0)   # conjugate of re
      check a.du == complex(-3.0, -4.0)  # -(conjugate of du)

    test "norm2 re-part equals |w|²":
      let w = complex(3.0, 4.0)   # |w|² = 25
      let u = complex(1.0, 0.0)
      let x = dual(w, u)
      let n = norm2(x)
      # re: w * conj(w) = |w|² = 25
      # du: w*conj(u) + u*conj(w) = (3+4i)(1) + (1)(3-4i) = (6 + 0i) → re-part of du = 6
      check almostEqual(n.re, complex(25.0, 0.0))
      check almostEqual(n.du.re, 6.0)

  suite "Dual[Complex64] — AD: exp and ln (holomorphic)":

    test "exp: d/dz e^z = e^z  (seed u=1+0i)":
      let w = complex(1.0, 1.0)
      let r = exp(dual(w, complex(1.0, 0.0)))
      let ew = exp(w)
      check almostEqual(r.re, ew)
      check almostEqual(r.du, ew)

    test "ln: d/dz ln z = 1/z  (seed u=1+0i)":
      let w = complex(1.0, 1.0)
      let r = ln(dual(w, complex(1.0, 0.0)))
      check almostEqual(r.re, ln(w))
      check almostEqual(r.du, complex(1.0, 0.0) / w)

    test "sqrt: d/dz √z = 1/(2√z)  (seed u=1+0i)":
      let w = complex(3.0, 4.0)   # √(3+4i) = 2+i
      let r = sqrt(dual(w, complex(1.0, 0.0)))
      let sw = sqrt(w)
      check almostEqual(r.re, sw)
      check almostEqual(r.du, complex(1.0, 0.0) / (sw + sw))

    test "Wirtinger: two passes recover ∂/∂z and ∂/∂z̄ of |z|²":
      # f(z,z̄) = |z|² = z·z̄  →  ∂f/∂z = z̄,  ∂f/∂z̄ = z
      let w = complex(2.0, 3.0)
      proc f(z: Dual[Complex64]): Dual[Complex64] = z * conjugate(z)
      # Pass 1: u = 1+0i
      let d1 = f(dual(w, complex(1.0, 0.0))).du
      # Pass 2: u = 0+1i
      let d2 = f(dual(w, complex(0.0, 1.0))).du
      let dz    = (d1 - complex(0.0, 1.0) * d2) / complex(2.0, 0.0)
      let dzbar = (d1 + complex(0.0, 1.0) * d2) / complex(2.0, 0.0)
      check almostEqual(dz,    conjugate(w))   # ∂|z|²/∂z  = z̄
      check almostEqual(dzbar, w)   # ∂|z|²/∂z̄ = z

  suite "Dual[float32] — constructor":

    test "single-argument seed uses float32 literal":
      let x = dual(2.0'f32)
      check x.re == 2.0'f32
      check x.du == 1.0'f32

  suite "type predicates":

    test "isDual":
      check isDual(Dual[float64])
      check isDual(Dual[Complex64])

    test "isDualReal32 / isDualReal64":
      check isDualReal32(Dual[float32])
      check isDualReal64(Dual[float64])
      check not isDualReal32(Dual[float64])

    test "isDualComplex32 / isDualComplex64":
      check isDualComplex32(Dual[Complex32])
      check isDualComplex64(Dual[Complex64])
      check not isDualComplex64(Dual[Complex32])