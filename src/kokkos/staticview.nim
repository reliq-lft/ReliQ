#[ 
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: src/kokkos/staticview.nim
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

# import backend header files
kokkos: discard

type # frontend: Kokkos dynamic rank view
  StaticView*[T] {.importcpp: "Views::StaticView", kokkos_wrapper.} = object

#[ frontend/backend: static view constructors ]#

# backend: Kokkos static view constructor
proc newStaticView[T](tag: cstring; n: csize_t): StaticView[T]
  {.importcpp: "Views::StaticView<'*0>(#, #)", constructor, inline, kokkos_wrapper.}

# backend: Kokkos static view constructor from pointer
proc newStaticView[T](localPtr: ptr T; n: csize_t): StaticView[T] 
  {.importcpp: "Views::StaticView<'*0>(#, #)", constructor, inline, kokkos_wrapper.}

# frontend: base Kokkos static view constructor
proc newStaticView*(n: SomeInteger; T: typedesc): StaticView[T] {.inline.} = 
  return newStaticView[T]("Views::StaticView", csize_t(n))

# frontend: Kokkos static view constructor from pointer
proc newStaticView*[T](
  n: SomeInteger;
  localPointer: ptr T
): StaticView[T] {.inline.} = newStaticView(localPointer, csize_t(n))
proc newStaticView*[T](
  localPointer: ptr T;
  n: SomeInteger
): StaticView[T] {.inline.} = newStaticView(localPointer, csize_t(n))

#[ frontend: static view methods ]#

# backend: static view accessor method
proc getViewElement[T](view: StaticView[T]; n: cint): T 
  {.importcpp: "#.operator()(#)", inline, kokkos_wrapper.}

# backend: static view accessor method
proc getViewElement[T](view: var StaticView[T]; n: cint): var T 
  {.importcpp: "#.operator()(#)", inline, kokkos_wrapper.}

# backend: set view element
proc setViewElement[T](view: var StaticView[T]; n: cint; value: T) 
  {.importcpp: "#.operator()(#) = #", inline, kokkos_wrapper.}

proc `[]`*[T](view: StaticView[T]; n: SomeInteger): T {.inline.} =
  return view.getViewElement(cint(n))

proc `[]`*[T](view: var StaticView[T]; n: SomeInteger): var T {.inline.} =
  return view.getViewElement(cint(n))

proc `[]=`*[T](view: var StaticView[T]; n: SomeInteger; value: T) {.inline.} =
  view.setViewElement(cint(n), value)

when isMainModule:
  import runtime
  reliq:
    var v = newStaticView(100, int)
    v[0] = 42
    print v[0]
    print v[1]
    print v[99]
    type T = float
    let size = 100
    var ta = cast[ptr UncheckedArray[T]](alloc(size*sizeof(T)))
    var tav = newStaticView(size, addr ta[][0])
    discard tav[0]
    # v[100] = 1 # out of bounds
    # echo v[100] # out of bounds