#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/kokkos/kokkosview.nim
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

import utils
import kokkosbase
import kokkossimd

# import backend header files
kokkos: discard

type # frontend: Kokkos dynamic rank view
  StaticView*[T] {.importcpp: "Kokkos::View<'*0*>", kokkos.} = object

#[ frontend/backend: static view constructors ]#

# backend: Kokkos static view constructor
proc newStaticView[T](tag: cstring; n: csize_t): StaticView[T]
  {.importcpp: "Kokkos::View<'*0*>(#, #)", constructor, inline, kokkos.}

# backend: Kokkos static view constructor from pointer
proc newStaticView[T](localPtr: ptr T; n: csize_t): StaticView[T] 
  {.importcpp: "Kokkos::View<'*0*>(#, #)", constructor, inline, kokkos.}

# frontend: base Kokkos static view constructor
proc newStaticView*(n: SomeInteger; T: typedesc): StaticView[T] {.inline.} = 
  return newStaticView[T]("StaticView", csize_t(n))

# frontend: Kokkos static view constructor from pointer
proc newStaticView*[T](localPointer: ptr T; len: int): StaticView[T] {.inline.} = 
  return localPointer.newStaticView(csize_t(len))

#[ frontend: static view methods ]#

# accessors
proc `[]`*[T](view: StaticView[T]; n: SomeInteger): T
  {.importcpp: "#.operator()(#)", inline, kokkos.}
proc `[]`*[T](view: var StaticView[T]; n: SomeInteger): var T
  {.importcpp: "#.operator()(#)", inline, kokkos.}

# setters
proc `[]=`*[T](view: var StaticView[T]; n: SomeInteger; value: T)
  {.importcpp: "#.operator()(#) = #", inline, kokkos.}

when isMainModule:
  import runtime
  import kokkos/[kokkossimd]
  reliq:
    var v = newStaticView(100, int)
    v[0] = 42
    print v[0]
    print v[1]
    print v[99]
    
    type T = SIMDStorage[SIMDArray[float]]
    let size = 100
    var ta = cast[ptr UncheckedArray[T]](alloc(size*sizeof(T)))
    var pta = addr ta[][0]
    var tav = pta.newStaticView(size)
    #discard tav[0]
    