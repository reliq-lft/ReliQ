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

import types/[composite]
import types/[complex]

template eigenScalarHeader*: untyped =
  {.pragma: scalar, header: "eigenscalar.h".}

eigenScalarHeader()

type
  EigenScalarHandleRS* {.importcpp: "EigenScalarHandleRS", scalar.} = object
  EigenScalarHandleRD* {.importcpp: "EigenScalarHandleRD", scalar.} = object
  EigenScalarHandleCS* {.importcpp: "EigenScalarHandleCS", scalar.} = object
  EigenScalarHandleCD* {.importcpp: "EigenScalarHandleCD", scalar.} = object

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
  {.importcpp: "createEigenScalarRS(@)", scalar.}

proc createEigenScalarRD(data: ptr float64): EigenScalarHandleRD 
  {.importcpp: "createEigenScalarRD(@)", scalar.}

proc createEigenScalarCS(data: pointer): EigenScalarHandleCS 
  {.importcpp: "createEigenScalarCS(@)", scalar.}

proc createEigenScalarCD(data: pointer): EigenScalarHandleCD 
  {.importcpp: "createEigenScalarCD(@)", scalar.}

# temp constructors

proc createTempEigenScalarRS(): EigenScalarHandleRS 
  {.importcpp: "createTempEigenScalarRS(@)", scalar.}

proc createTempEigenScalarRD(): EigenScalarHandleRD 
  {.importcpp: "createTempEigenScalarRD(@)", scalar.}

proc createTempEigenScalarCS(): EigenScalarHandleCS 
  {.importcpp: "createTempEigenScalarCS(@)", scalar.}

proc createTempEigenScalarCD(): EigenScalarHandleCD 
  {.importcpp: "createTempEigenScalarCD(@)", scalar.}

# destructors

proc destroyEigenScalarRS(handle: EigenScalarHandleRS, ownsData: bool) 
  {.importcpp: "destroyEigenScalarRS(@)", scalar.}

proc destroyEigenScalarRD(handle: EigenScalarHandleRD, ownsData: bool) 
  {.importcpp: "destroyEigenScalarRD(@)", scalar.}

proc destroyEigenScalarCS(handle: EigenScalarHandleCS, ownsData: bool) 
  {.importcpp: "destroyEigenScalarCS(@)", scalar.}

proc destroyEigenScalarCD(handle: EigenScalarHandleCD, ownsData: bool) 
  {.importcpp: "destroyEigenScalarCD(@)", scalar.}

# accessors

proc eigenScalarRSGet(handle: EigenScalarHandleRS): float32 
  {.importcpp: "eigenScalarRSGet(@)", scalar.}

proc eigenScalarRDGet(handle: EigenScalarHandleRD): float64 
  {.importcpp: "eigenScalarRDGet(@)", scalar.}

proc eigenScalarCSGet(handle: EigenScalarHandleCS, outVal: pointer) 
  {.importcpp: "eigenScalarCSGet(@)", scalar.}

proc eigenScalarCDGet(handle: EigenScalarHandleCD, outVal: pointer) 
  {.importcpp: "eigenScalarCDGet(@)", scalar.}

proc eigenScalarRSSet(handle: EigenScalarHandleRS, value: float32) 
  {.importcpp: "eigenScalarRSSet(@)", scalar.}

proc eigenScalarRDSet(handle: EigenScalarHandleRD, value: float64) 
  {.importcpp: "eigenScalarRDSet(@)", scalar.}

proc eigenScalarCSSet(handle: EigenScalarHandleCS, value: pointer) 
  {.importcpp: "eigenScalarCSSet(@)", scalar.}

proc eigenScalarCDSet(handle: EigenScalarHandleCD, value: pointer) 
  {.importcpp: "eigenScalarCDSet(@)", scalar.}

# algebra

proc eigenScalarRSAdd(a, b: EigenScalarHandleRS; c: EigenScalarHandleRS) 
  {.importcpp: "eigenScalarRSAdd(@)", scalar.}

proc eigenScalarRDAdd(a, b: EigenScalarHandleRD; c: EigenScalarHandleRD) 
  {.importcpp: "eigenScalarRDAdd(@)", scalar.}

proc eigenScalarCSAdd(a, b: EigenScalarHandleCS; c: EigenScalarHandleCS) 
  {.importcpp: "eigenScalarCSAdd(@)", scalar.}

proc eigenScalarCDAdd(a, b: EigenScalarHandleCD; c: EigenScalarHandleCD) 
  {.importcpp: "eigenScalarCDAdd(@)", scalar.}

proc eigenScalarRSSub(a, b: EigenScalarHandleRS; c: EigenScalarHandleRS) 
  {.importcpp: "eigenScalarRSSub(@)", scalar.}

proc eigenScalarRDSub(a, b: EigenScalarHandleRD; c: EigenScalarHandleRD) 
  {.importcpp: "eigenScalarRDSub(@)", scalar.}

proc eigenScalarCSSub(a, b: EigenScalarHandleCS; c: EigenScalarHandleCS) 
  {.importcpp: "eigenScalarCSSub(@)", scalar.}

proc eigenScalarCDSub(a, b: EigenScalarHandleCD; c: EigenScalarHandleCD) 
  {.importcpp: "eigenScalarCDSub(@)", scalar.}

proc eigenScalarRSMul(a, b: EigenScalarHandleRS; c: EigenScalarHandleRS) 
  {.importcpp: "eigenScalarRSMul(@)", scalar.}

proc eigenScalarRDMul(a, b: EigenScalarHandleRD; c: EigenScalarHandleRD) 
  {.importcpp: "eigenScalarRDMul(@)", scalar.}

proc eigenScalarCSMul(a, b: EigenScalarHandleCS; c: EigenScalarHandleCS) 
  {.importcpp: "eigenScalarCSMul(@)", scalar.}

proc eigenScalarCDMul(a, b: EigenScalarHandleCD; c: EigenScalarHandleCD) 
  {.importcpp: "eigenScalarCDMul(@)", scalar.}

proc eigenScalarRSDiv(a, b: EigenScalarHandleRS; c: EigenScalarHandleRS) 
  {.importcpp: "eigenScalarRSDiv(@)", scalar.}

proc eigenScalarRDDiv(a, b: EigenScalarHandleRD; c: EigenScalarHandleRD) 
  {.importcpp: "eigenScalarRDDiv(@)", scalar.}

proc eigenScalarCSDiv(a, b: EigenScalarHandleCS; c: EigenScalarHandleCS) 
  {.importcpp: "eigenScalarCSDiv(@)", scalar.}

proc eigenScalarCDDiv(a, b: EigenScalarHandleCD; c: EigenScalarHandleCD) 
  {.importcpp: "eigenScalarCDDiv(@)", scalar.}

# compound assignment

proc eigenScalarRSAddAssign(a, b: EigenScalarHandleRS) 
  {.importcpp: "eigenScalarRSAddAssign(@)", scalar.}

proc eigenScalarRDAddAssign(a, b: EigenScalarHandleRD) 
  {.importcpp: "eigenScalarRDAddAssign(@)", scalar.}

proc eigenScalarCSAddAssign(a, b: EigenScalarHandleCS) 
  {.importcpp: "eigenScalarCSAddAssign(@)", scalar.}

proc eigenScalarCDAddAssign(a, b: EigenScalarHandleCD) 
  {.importcpp: "eigenScalarCDAddAssign(@)", scalar.}

proc eigenScalarRSSubAssign(a, b: EigenScalarHandleRS) 
  {.importcpp: "eigenScalarRSSubAssign(@)", scalar.}

proc eigenScalarRDSubAssign(a, b: EigenScalarHandleRD) 
  {.importcpp: "eigenScalarRDSubAssign(@)", scalar.}

proc eigenScalarCSSubAssign(a, b: EigenScalarHandleCS) 
  {.importcpp: "eigenScalarCSSubAssign(@)", scalar.}

proc eigenScalarCDSubAssign(a, b: EigenScalarHandleCD) 
  {.importcpp: "eigenScalarCDSubAssign(@)", scalar.}

proc eigenScalarRSMulAssign(a, b: EigenScalarHandleRS) 
  {.importcpp: "eigenScalarRSMulAssign(@)", scalar.}

proc eigenScalarRDMulAssign(a, b: EigenScalarHandleRD) 
  {.importcpp: "eigenScalarRDMulAssign(@)", scalar.}

proc eigenScalarCSMulAssign(a, b: EigenScalarHandleCS) 
  {.importcpp: "eigenScalarCSMulAssign(@)", scalar.}

proc eigenScalarCDMulAssign(a, b: EigenScalarHandleCD) 
  {.importcpp: "eigenScalarCDMulAssign(@)", scalar.}

proc eigenScalarRSDivAssign(a, b: EigenScalarHandleRS) 
  {.importcpp: "eigenScalarRSDivAssign(@)", scalar.}

proc eigenScalarRDDivAssign(a, b: EigenScalarHandleRD) 
  {.importcpp: "eigenScalarRDDivAssign(@)", scalar.}

proc eigenScalarCSDivAssign(a, b: EigenScalarHandleCS) 
  {.importcpp: "eigenScalarCSDivAssign(@)", scalar.}

proc eigenScalarCDDivAssign(a, b: EigenScalarHandleCD) 
  {.importcpp: "eigenScalarCDDivAssign(@)", scalar.}

# abs / conj

proc eigenScalarRSAbs(handle: EigenScalarHandleRS): float32 
  {.importcpp: "eigenScalarRSAbs(@)", scalar.}

proc eigenScalarRDAbs(handle: EigenScalarHandleRD): float64 
  {.importcpp: "eigenScalarRDAbs(@)", scalar.}

proc eigenScalarCSAbs(handle: EigenScalarHandleCS): float32 
  {.importcpp: "eigenScalarCSAbs(@)", scalar.}

proc eigenScalarCDAbs(handle: EigenScalarHandleCD): float64 
  {.importcpp: "eigenScalarCDAbs(@)", scalar.}

proc eigenScalarCSConj(handle: EigenScalarHandleCS, outVal: pointer) 
  {.importcpp: "eigenScalarCSConj(@)", scalar.}

proc eigenScalarCDConj(handle: EigenScalarHandleCD, outVal: pointer) 
  {.importcpp: "eigenScalarCDConj(@)", scalar.}

#[ EigenScalar implementation ]#

implement EigenScalar with:
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
  
  method deinit =
    when isReal32(T): destroyEigenScalarRS(this.data, this.ownsData)
    elif isReal64(T): destroyEigenScalarRD(this.data, this.ownsData)
    elif isComplex32(T): destroyEigenScalarCS(this.data, this.ownsData)
    elif isComplex64(T): destroyEigenScalarCD(this.data, this.ownsData)
  
  #[ copy/move semantics ]#

  method copy(src: EigenScalar[T]) =
    ## Shallow copy: share the raw data pointer with ownsData=false
    ## so this copy's =destroy never frees the underlying value.
    if addr(this) == unsafeAddr(src): return
    this.ownsData = false
    this.data = src.data

  method sink(src: EigenScalar[T]) =
    ## Transfer ownership from src to this without cloning.
    `=destroy`(this)
    this.ownsData = src.ownsData
    this.data = src.data

  #[ accessors ]#

  method `[]`: T {.immutable.} =
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
  
  method `[]=`(value: T) =
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

  #[ copy operation ]#

  method `:=`(val: T) {.immutable.} = 
    # shouldn't really be immutable, but this allows using `s := val` syntax to 
    # write through to the underlying value without needing a var scalar, which 
    # is more ergonomic for common use cases like `field[n] := val` and 
    # `s := complex(3.0, 4.0)`.
    ## Write a raw value through the scalar's underlying pointer.
    ## Enables `s := complex(3.0, 4.0)` and `field[n] := val`.
    ## dst need not be `var` — writes go through the C++ handle, not the struct.
    when isReal32(T): eigenScalarRSSet(this.data, val)
    elif isReal64(T): eigenScalarRDSet(this.data, val)
    elif isComplex32(T):
      var v = val
      eigenScalarCSSet(this.data, addr v)
    elif isComplex64(T):
      var v = val
      eigenScalarCDSet(this.data, addr v)
  
  #[ algebra ]#

  method `+`(other: EigenScalar[T]): EigenScalar[T] {.immutable.} =
    result = newEigenScalar[T](this[])
    when isReal32(T): eigenScalarRSAdd(this.data, other.data, result.data)
    elif isReal64(T): eigenScalarRDAdd(this.data, other.data, result.data)
    elif isComplex32(T): eigenScalarCSAdd(this.data, other.data, result.data)
    elif isComplex64(T): eigenScalarCDAdd(this.data, other.data, result.data)
  
  method `-`(other: EigenScalar[T]): EigenScalar[T] {.immutable.} =
    result = newEigenScalar[T](this[])
    when isReal32(T): eigenScalarRSSub(this.data, other.data, result.data)
    elif isReal64(T): eigenScalarRDSub(this.data, other.data, result.data)
    elif isComplex32(T): eigenScalarCSSub(this.data, other.data, result.data)
    elif isComplex64(T): eigenScalarCDSub(this.data, other.data, result.data)
  
  method `*`(other: EigenScalar[T]): EigenScalar[T] {.immutable.} =
    result = newEigenScalar[T](this[])
    when isReal32(T): eigenScalarRSMul(this.data, other.data, result.data)
    elif isReal64(T): eigenScalarRDMul(this.data, other.data, result.data)
    elif isComplex32(T): eigenScalarCSMul(this.data, other.data, result.data)
    elif isComplex64(T): eigenScalarCDMul(this.data, other.data, result.data)
  
  method `/`(other: EigenScalar[T]): EigenScalar[T] {.immutable.} =
    result = newEigenScalar[T](this[])
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

  method abs: auto {.immutable.} =
    when isReal32(T):
      return eigenScalarRSAbs(this.data)
    elif isReal64(T):
      return eigenScalarRDAbs(this.data)
    elif isComplex32(T):
      return eigenScalarCSAbs(this.data)
    elif isComplex64(T):
      return eigenScalarCDAbs(this.data)

  method conj: EigenScalar[T] {.immutable.} =
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
      result = newEigenScalar[T](this[])

  # Value-comparison operators — compare underlying data, not the struct.
  # The record macro skips auto-generating these when they're user-defined.
  method `==`(other: EigenScalar[T]): bool {.immutable.} = this[] == other[]
  method `!=`(other: EigenScalar[T]): bool {.immutable.} = this[] != other[]
  method `<`(other: EigenScalar[T]): bool {.immutable.} = this[] < other[]
  method `<=`(other: EigenScalar[T]): bool {.immutable.} = this[] <= other[]
  method `>`(other: EigenScalar[T]): bool {.immutable.} = this[] > other[]
  method `>=`(other: EigenScalar[T]): bool {.immutable.} = this[] >= other[]

converter toValue*[T](s: EigenScalar[T]): T =
  ## Implicitly read an EigenScalar as its element type T.
  ## Allows using a scalar anywhere T is expected (comparisons, arithmetic, abs, …).
  s[]

# Scalar vs raw value comparisons (and reverse) — different RHS type so
# these are always unambiguous overloads alongside the record methods above.
proc `==`*[T](s: EigenScalar[T]; val: T): bool = s[] == val
proc `==`*[T](val: T; s: EigenScalar[T]): bool = val == s[]
proc `!=`*[T](s: EigenScalar[T]; val: T): bool = s[] != val
proc `!=`*[T](val: T; s: EigenScalar[T]): bool = val != s[]
proc `<`*[T](s: EigenScalar[T]; val: T): bool = s[] < val
proc `<`*[T](val: T; s: EigenScalar[T]): bool = val < s[]
proc `<=`*[T](s: EigenScalar[T]; val: T): bool = s[] <= val
proc `<=`*[T](val: T; s: EigenScalar[T]): bool = val <= s[]
proc `>`*[T](s: EigenScalar[T]; val: T): bool = s[] > val
proc `>`*[T](val: T; s: EigenScalar[T]): bool = val > s[]
proc `>=`*[T](s: EigenScalar[T]; val: T): bool = s[] >= val
proc `>=`*[T](val: T; s: EigenScalar[T]): bool = val >= s[]

when isMainModule:
  import std/[unittest, math]

  suite "EigenScalar tests":
    test "real creation, access, and write-through":
      var val = 42.0
      var s = newEigenScalar(addr val)
      check s.ownsData == false
      check s == 42.0

      # write through scalar → underlying value
      s = 99.0
      check val == 99.0

      # write through value → visible via scalar
      val = 7.0
      check s == 7.0

    test "complex creation and access":
      var cval = complex(3.0, 4.0)
      var s = newEigenScalar(addr cval)
      check s == complex(3.0, 4.0)

      s := complex(10.0, 20.0)
      check cval.re == 10.0
      check cval.im == 20.0

    test "temporary scalar":
      var s = newEigenScalar[float64](0.0)
      check s.ownsData == true
      check s == 0.0

      s = 123.0
      check s == 123.0

    test "real arithmetic (+, -, *, /)":
      var aval = 10.0
      var bval = 3.0
      var a = newEigenScalar(addr aval)
      var b = newEigenScalar(addr bval)

      var s = a + b
      check s == 13.0

      var d = a - b
      check d == 7.0

      var p = a * b
      check p == 30.0

      var q = a / b
      check abs(q - 10.0 / 3.0) < 1e-12

    test "real compound assignment (+=, -=, *=, /=)":
      var aval = 10.0
      var bval = 3.0
      var a = newEigenScalar(addr aval)
      var b = newEigenScalar(addr bval)

      a += b
      check a == 13.0
      check aval == 13.0  # write-through

      a -= b
      check a == 10.0

      a *= b
      check a == 30.0

      a /= b
      check abs(a - 10.0) < 1e-12

    test "complex arithmetic":
      var ca = complex(1.0, 2.0)
      var cb = complex(3.0, 4.0)
      var a = newEigenScalar(addr ca)
      var b = newEigenScalar(addr cb)

      var s = a + b
      check s == complex(4.0, 6.0)

      var d = a - b
      check d == complex(-2.0, -2.0)

      # (1+2i)(3+4i) = 3+4i+6i+8i² = -5+10i
      var p = a * b
      check p == complex(-5.0, 10.0)

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
      check s.conj() == complex(3.0, -4.0)

      # conj of real is itself
      var rval = 7.0
      var rs = newEigenScalar(addr rval)
      check rs.conj() == 7.0
