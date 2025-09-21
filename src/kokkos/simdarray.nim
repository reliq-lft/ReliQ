#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/kokkos/simd.nim
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

import kokkosbase
import utils

# shorten pragmas pointing to Kokkos headers
kokkos: {.pragma: simd, header: "<Kokkos_SIMD.hpp>".}

type 
  SIMDArray*[T] {.importcpp: "Kokkos::Experimental::simd", simd.} = object
  ## ReliQ wrapper for Kokkos SIMDArray type
  ## 
  ## <in need of documentation>

# SIMDArray constructors
proc newSIMDArray[T]: SIMDArray[T] 
  {.importcpp: "Kokkos::Experimental::simd<'*0>()", simd.}
proc newSIMDArray*(T: typedesc): SIMDArray[T] = newSIMDArray[T]()
proc newSIMDArray*[T](value: T): SIMDArray[T] 
  {.importcpp: "Kokkos::Experimental::simd<'*0>(#)", simd.}
proc newSIMDArray*[T](other: SIMDArray[T]): SIMDArray[T] 
  {.importcpp: "Kokkos::Experimental::simd<'*0>(#)", simd.}
proc newSIMDArray*[T](values: seq[T]): SIMDArray[T] =
  result = newSIMDArray(T)
  for l in 0..<min(result.width, values.len): 
    let value = values[l]
    {.emit: """
    Kokkos::Experimental::simd_mask<`T`> m([&] (std::size_t i) { return i == `l`; });
    Kokkos::Experimental::simd<`T`> c(`value`);
    where(m, result) = c;
    """.}
proc newSIMDArray*[T](values: openArray[T]): SIMDArray[T] = newSIMDArray(values.toSeq())

# SIMDArray width
proc width*[T](simd: SIMDArray[T]): int {.importcpp: "#.size()", simd.}

# array access operator overloads
proc `[]`*[T](simd: SIMDArray[T], lane: int): T {.importcpp: "#.operator[](#)", simd.}
proc `[]=`*[T](simd: var SIMDArray[T], lane: int, value: T) =
  {.emit: """
  Kokkos::Experimental::simd_mask<`T`> m([&] (std::size_t i) { return i == `lane`; });
  Kokkos::Experimental::simd<`T`> c(`value`);
  where(m, `simd`) = c;
  """.}

# copy assignment hook
proc `=copy`*[T](x: var SIMDArray[T]; y: SIMDArray[T]) 
  {.importcpp: "#.operator=(#)", simd.}

# assignment to scalar
proc `:=`*[T](x: var SIMDArray[T]; value: T) =
  {.emit: """
  Kokkos::Experimental::simd<`T`> c(`value`);
  `x` = c;
  """.}

# copy from pointer
proc load*[T](x: var SIMDArray[T]; mem: ptr T) 
  {.importcpp: "#.copy_from(#, Kokkos::Experimental::simd_flag_default)", simd.}

# store to aligned memory starting at pointer
proc store*[T](x: SIMDArray[T]; mem: ptr T) 
  {.importcpp: "#.copy_to(#, Kokkos::Experimental::simd_flag_default)", simd.}

# assignment to another vector
proc `:=`*[T](x: var SIMDArray[T]; value: SIMDArray[T]) =
  {.emit: """
  Kokkos::Experimental::simd<`T`> c(`value`);
  `x` = c;
  """.}

# Kokkos SIMDArray arithematic overloads
proc `+`*[T](a, b: SIMDArray[T]): SIMDArray[T] {.importcpp: "operator+(#, #)", simd.}
proc `-`*[T](a, b: SIMDArray[T]): SIMDArray[T] {.importcpp: "operator-(#, #)", simd.}
proc `*`*[T](a, b: SIMDArray[T]): SIMDArray[T] {.importcpp: "operator*(#, #)", simd.}
proc `/`*[T](a, b: SIMDArray[T]): SIMDArray[T] {.importcpp: "operator/(#, #)", simd.}

# Kokkos SIMDArray compound arithematic assignment overloads
proc `+=`*[T](a: SIMDArray[T], b: SIMDArray[T]) {.importcpp: "operator+=(#, #)", simd.}
proc `-=`*[T](a: SIMDArray[T], b: SIMDArray[T]) {.importcpp: "operator-=(#, #)", simd.}
proc `*=`*[T](a: SIMDArray[T], b: SIMDArray[T]) {.importcpp: "operator*=(#, #)", simd.}
proc `/=`*[T](a: SIMDArray[T], b: SIMDArray[T]) {.importcpp: "operator/=(#, #)", simd.}

# Kokkos SIMDArray compound arithematic assignment overloads
proc `+=`*[T](a: SIMDArray[T], b: T) {.importcpp: "operator+=(#, #)", simd.}
proc `-=`*[T](a: SIMDArray[T], b: T) {.importcpp: "operator-=(#, #)", simd.}
proc `*=`*[T](a: SIMDArray[T], b: T) {.importcpp: "operator*=(#, #)", simd.}
proc `/=`*[T](a: SIMDArray[T], b: T) {.importcpp: "operator/=(#, #)", simd.}

# SIMDArray vector iterator
iterator values*[T](a: SIMDArray[T]): T =
  for l in 0..<a.width: yield a[l]

# conversion from sequence to SIMDArray
proc toSIMDArray*[T](values: seq[T]): SIMDArray[T] = newSIMDArray(values)
proc toSIMDArray*[T](values: openArray[T]): SIMDArray[T] = newSIMDArray(values.toSeq())

# conversion to sequence
proc toSeq*[T](a: SIMDArray[T]): seq[T] =
  result = newSeq[T](a.width)
  for l in 0..<a.width: result[l] = a[l]

# conversion to string
proc `$`*[T](a: SIMDArray[T]): string = "SIMDArray:" + $(a.toSeq())

proc numLanes*: int =
  # Fetch number of SIMD (SIMT) lanes
  # * on CPU: number of SIMD lanes
  # * on GPU: warp size
  let t = newSIMDArray(float64)
  return t.width

when isMainModule:
  import runtime
  reliq:
    let sq = @[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    var 
      l = newSIMDArray(float)
      x = newSIMDArray(1.0)
      y = newSIMDArray(2.0)
      z = newSIMDArray(l)
      t = x + y
    t = x + y
    t = x - y
    t = x * y
    t = x / y
    t += x
    t -= y
    t *= x
    t /= y
    print $t[0]
    print t.width
    print $t
    for v in t.values: print $v
    t[3] = 3.0
    print $t
    var 
      sqva = newSIMDArray(sq)
      sqvb = sq.toSIMDArray()
    print sqva
    print sqvb
    sqva = sqvb