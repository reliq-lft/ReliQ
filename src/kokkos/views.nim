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

import utils
import kokkosbase

# shorten pragmas pointing to Kokkos headers and include local views wrapper
kokkos: {.pragma: views, header: "views.hpp".}

type # frontend: Kokkos dynamic rank view
  DynamicView*[T] {.importcpp: "DynamicView", views.} = object

# enumerate possible errors that a user may run into
type KokkosViewErrors = enum 
  TooManyDynamicViewDimensionsError,
  EmptyDynamicViewDimensionsError

# special error type for handling exception during lattice initialization
type KokkosViewError = object of CatchableError

# implementation of exception handling type
template newKokkosViewError(
  err: KokkosViewErrors,
  appendToMessage: untyped
): untyped =
  # constructs error to be raised according to LatticeInitializationErrors spec
  if myRank() == 0:
    var errorMessage {.inject.} = case err:
      of TooManyDynamicViewDimensionsError:
        "Kokkos dynamic rank views support only up to seven dimensions."
      of EmptyDynamicViewDimensionsError:
        "ReliQ wrapper of Kokkos dynamic rank view must have at least one dimension."
    errorMessage = printBreak & errorMessage
    appendToMessage
    print errorMessage & printBreak
  newException(KokkosViewError, "")

# backend: Kokkos dynamic rank view constructors (Kokkos supports up to 7 dimensions)
proc newDynamicView[T](tag: cstring; n1: csize_t): DynamicView[T]
  {.importcpp: "DynamicView" & "<'*0>(#, @)", constructor, inline, views.}
proc newDynamicView[T](tag: cstring; n1, n2: csize_t): DynamicView[T]
  {.importcpp: "DynamicView" & "<'*0>(#, @)", constructor, inline, views.}
proc newDynamicView[T](tag: cstring; n1, n2, n3: csize_t): DynamicView[T]
  {.importcpp: "DynamicView" & "<'*0>(#, @)", constructor, inline, views.}
proc newDynamicView[T](tag: cstring; n1, n2, n3, n4: csize_t): DynamicView[T]
  {.importcpp: "DynamicView" & "<'*0>(#, @)", constructor, inline, views.}
proc newDynamicView[T](tag: cstring; n1, n2, n3, n4, n5: csize_t): DynamicView[T]
  {.importcpp: "DynamicView" & "<'*0>(#, @)", constructor, inline, views.}
proc newDynamicView[T](tag: cstring; n1, n2, n3, n4, n5, n6: csize_t): DynamicView[T]
  {.importcpp: "DynamicView" & "<'*0>(#, @)", constructor, inline, views.}
proc newDynamicView[T](
  tag: cstring; n1, n2, n3, n4, n5, n6, n7: csize_t
): DynamicView[T] 
  {.importcpp: "DynamicView" & "<'*0>(#, @)", constructor, inline, views.}

# frontend: more flexible dynamic view constructor
proc newDynamicView*(
  dims: openArray[SomeInteger];
  T: typedesc
): DynamicView[T] {.inline.} =
  var d = dims.toSeq(csize_t)
  return case dims.len:
    of 1: newDynamicView[T]("DynamicView", d[0])
    of 2: newDynamicView[T]("DynamicView", d[0], d[1])
    of 3: newDynamicView[T]("DynamicView", d[0], d[1], d[2])
    of 4: newDynamicView[T]("DynamicView", d[0], d[1], d[2], d[3])
    of 5: newDynamicView[T]("DynamicView", d[0], d[1], d[2], d[3], d[4])
    of 6: newDynamicView[T]("DynamicView", d[0], d[1], d[2], d[3], d[4], d[5])
    of 7: newDynamicView[T]("DynamicView", d[0], d[1], d[2], d[3], d[4], d[5], d[6])
    else:
      if dims.len == 0:
        raise newKokkosViewError(EmptyDynamicViewDimensionsError): discard
      if dims.len > 7: 
        raise newKokkosViewError(TooManyDynamicViewDimensionsError):
          errorMessage &= "\ndimensions:" + $dims
      newDynamicView[T]("DynamicView", d[0])

# lessons learned:
# * problems with C++ "=" operator overload seem to stem from improper spec of 
#   importcpp pragma string; i.e., it is wrong or unsupported by compiler
# * private constructor procedure confuses Nim compiler w/ Kokkos
#   - repr: var "testa = newDynamicView[float]("DynamicView", 10, 10)" does not 
#     compile when combined with var "testb = newDynamicView[float]([10, 10])"
when isMainModule:
  import runtime
  reliq:
    let arr = [10, 10] 
    var viewa = arr.newDynamicView(float)
