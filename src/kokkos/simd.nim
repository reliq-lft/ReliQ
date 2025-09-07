#[ 
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: src/kokkos/views.nim
  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 Curtis Taylor Peterson
  
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

# SIMXVec type
type 
  SIMXVec*[T] {.importcpp: "Kokkos::Experimental::simd", simd.} = object
  ## ReliQ wrapper for Kokkos SIMXVec type
  ## 
  ## <in need of documentation>

# SIMXVec constructor
proc newSIMXVec[T](): SIMXVec[T] {.importcpp: "Kokkos::Experimental::simd<'*0>()", simd.}
proc newSIMXVec*(T: typedesc): SIMXVec[T] = newSIMXVec[T]()
proc newSIMXVec*[T](value: T): SIMXVec[T] 
  {.importcpp: "Kokkos::Experimental::simd<'*0>(#)", simd.}
proc newSIMXVec*[T](other: SIMXVec[T]): SIMXVec[T] 
  {.importcpp: "Kokkos::Experimental::simd<'*0>(#)", simd.}
# TODO: ... placeholder for generator constructor...

# SIMXVec width
proc width*[T](a: SIMXVec[T]): int {.importcpp: "#.size()", simd.}

# Kokkos SIMXVec accessors
proc `[]`*[T](a: SIMXVec[T], lane: int): T {.importcpp: "#.operator[](#)", simd.}
# TODO: ... placeholder for generator-based accessor; may need to do
# intermediate conversion to sequence or something...

# Kokkos SIMXVec assignment overloads
proc `=copy`*[T](a: var SIMXVec[T], b: SIMXVec[T]) {.importcpp: "operator=(#, #)", simd.}

# Kokkos SIMXVec arithematic overloads
proc `+`*[T](a, b: SIMXVec[T]): SIMXVec[T] {.importcpp: "operator+(#, #)", simd.}
proc `-`*[T](a, b: SIMXVec[T]): SIMXVec[T] {.importcpp: "operator-(#, #)", simd.}
proc `*`*[T](a, b: SIMXVec[T]): SIMXVec[T] {.importcpp: "operator*(#, #)", simd.}
proc `/`*[T](a, b: SIMXVec[T]): SIMXVec[T] {.importcpp: "operator/(#, #)", simd.}

# Kokkos SIMXVec compound arithematic assignment overloads
proc `+=`*[T](a: var SIMXVec[T], b: SIMXVec[T]) {.importcpp: "operator+=(#, #)", simd.}
proc `-=`*[T](a: var SIMXVec[T], b: SIMXVec[T]) {.importcpp: "operator-=(#, #)", simd.}
proc `*=`*[T](a: var SIMXVec[T], b: SIMXVec[T]) {.importcpp: "operator*=(#, #)", simd.}
proc `/=`*[T](a: var SIMXVec[T], b: SIMXVec[T]) {.importcpp: "operator/=(#, #)", simd.}

# SIMXVec vector iterator
iterator values*[T](a: SIMXVec[T]): T =
  for l in 0..<a.width: yield a[l]

# converion to sequence
proc toSeq*[T](a: SIMXVec[T]): seq[T] =
  result = newSeq[T](a.width)
  for l in 0..<a.width: result[l] = a[l]

# conversion to string
proc `$`*[T](a: SIMXVec[T]): string = "SIMXVec:" + $(a.toSeq())

when isMainModule:
  import runtime
  reliq:
    var 
      l = newSIMXVec(float)
      x = newSIMXVec(1.0)
      y = newSIMXVec(2.0)
      z = newSIMXVec(l)
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
    