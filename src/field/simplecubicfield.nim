#[ 
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: src/field/simplecubicfield.nim
  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 reliq-lft
  
  Permission is hereby granted, free of chadge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, medge, publish, distribute, sublicense, and/or sell
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
import backend
import lattice/[simplecubiclattice]

# shorten pragmas pointing to UPC++ & Kokkos headers and include field view wrapper
backend: discard

#[ frontend: simple cubic field type definition ]#

type
  SimpleCubicField*[T] = object
    ## Field on simple cubic Bravais lattice
    ##
    ## <in need of documentation>
    lattice: ptr SimpleCubicLattice
    field: GlobalPointer[SIMXVec[T]]
    #fieldDist: DistributedObject[GlobalPointer[SIMXVec[T]]]
    fieldView: StaticView[SIMXVec[T]]

#[ backend: types for exception handling ]#

type # enumerate possible errors that a user may run into
  SimpleCubicFieldErrors = enum 
    LatticeConformabilityError

# special error type for handling field exception
type SimpleCubicFieldError = object of CatchableError

#[ backend: exception handling ]#

template newSimpleCubicFieldError(
  err: SimpleCubicFieldErrors,
  appendToMessage: untyped
): untyped =
  # constructs error to be raised according to SimpleCubicFieldErrors spec
  if myRank() == 0:
    var errorMessage {.inject.} = case err:
      of LatticeConformabilityError: "Lattices must match for field operations"
    errorMessage = printBreak & errorMessage
    appendToMessage
    print errorMessage & printBreak
    raise newException(SimpleCubicFieldError, "")

#[ frontend: helpful simple cubic field methods ]#

proc conformable*[T](fx, fy: SimpleCubicField[T]) =
  ## Check if two fields are conformable (i.e., defined on the same lattice)
  ##
  ## <in need of documentation>
  if fx.lattice[] != fy.lattice[]: 
    newSimpleCubicFieldError(LatticeConformabilityError): 
      errorMessage &&= $(fx.lattice[]) && $(fy.lattice[])

#[ frontend: SimpleCubicField constructor ]#

proc newField*(lattice: SimpleCubicLattice; T: typedesc): SimpleCubicField[T] =
  ## Create new field on simple cubic Bravais lattice
  ##
  ## <in need of documentation>
  let numVecSites = csize_t(lattice.numVecSites)
  result = SimpleCubicField[T](
    lattice: addr lattice, 
    field: newGlobalPointerArray(numVecSites, SIMXVec[T])
  )
  #result.fieldDist = result.field.newDistributedObject()
  result.fieldView = result.field.newStaticView(numVecSites)

#[ frontend: implement field concept ]#

proc lattice*[T](f: SimpleCubicField[T]): SimpleCubicLattice =
  ## Return lattice associated with field
  ##
  ## <in need of documentation>
  return f.lattice[]

proc numVecSites*[T](f: SimpleCubicField[T]): int =
  ## Return number of vectorized sites in field
  ##
  ## <in need of documentation>
  return f.lattice[].numVecSites

proc `[]`*[T](f: SimpleCubicField[T]; n: SomeInteger): SIMXVec[T] {.inline.} =
  ## Access field value at given local site
  ##
  ## This is for testing; it is most certainly not optimal or efficient
  ## <in need of documentation>
  return f.fieldView[n]

proc `[]`*[T](f: var SimpleCubicField[T]; n: SomeInteger): var SIMXVec[T] {.inline.} =
  ## Access field value at given local site
  ##
  ## This is for testing; it is most certainly not optimal or efficient
  ## <in need of documentation>
  return f.fieldView[n]

proc `[]=`*[T](
  f: var SimpleCubicField[T]; 
  n: SomeInteger; 
  value: SIMXVec[T]
) {.inline.} =
  ## Set field value at given local site
  ##
  ## This is for testing; it is most certainly not optimal or efficient
  ## <in need of documentation>
  f.fieldView[n] := value

proc `:=`*[T](fx: var SimpleCubicField[T]; fy: SimpleCubicField[T]) =
  ## Set all field values to given scalar value
  ##
  ## <in need of documentation>
  conformable(fx, fy)
  var ctx = newParallelForContext(fx.numVecSites)
  ctx.pack(fx, fy)
  ctx.each(n):
    let fx = cast[ptr SimpleCubicField[T]](context.ptrs[0])
    let fy = cast[ptr SimpleCubicField[T]](context.ptrs[1])
    fx[][n] := fy[][n]

# I WONDER IF THE PROBLEM IS THAT ALL THREADS TRY TO ACCESS THE SAME
# MEMORY LOCATION AT ONCE... IF SO, I NEED TO MAKE SURE THAT THEY
# ARE ALL ACCESSING DIFFERENT LOCATIONS. MAYBE I NEED TO BREAK IT
# UP INTO CHUNKS, AND HAVE EACH THREAD DEAL WITH A CHUNK AT A TIME...
# ... OR MAYBE I NEED TO JUST HAVE EACH THREAD DEAL WITH A SINGLE
# LOCATION AT A TIME, AND LET KOKKOS HANDLE THE REST...
proc `:=`*[T](f: var SimpleCubicField[T]; value: T) =
  ## Set all field values to given scalar value
  ##
  ## <in need of documentation>
  var ctx = newParallelForContext(f.numVecSites, flts = @[float(value)])
  ctx.pack(f)
  ctx.each(n):
    let f = cast[ptr SimpleCubicField[T]](context.ptrs[0])
    f[][n] := newSIMXVec(T(context.flts[0]))

# frontend: slick Nim trick for defining binary operations
template defineBinaryOperations*(op: untyped; ope: untyped) =
  # idea for a later point: Chapel had a neat way of promiting arithematic
  # expressions, such that the whole expression was turned into a single loop;
  # I want this eventually, but I want to focus first on base-level functionality
  proc `op`*[T](fx, fy: SimpleCubicField[T]): SimpleCubicField[T] {.inline.} =
    conformable(fx, fy)
    var ctx = newParallelForContext(fx.numVecSites)
    var fr = fx.lattice[].newField(T)
    ctx.pack(fr, fx, fy)
    ctx.each(n):
      let fr = cast[ptr SimpleCubicField[T]](context.ptrs[0])
      let fx = cast[ptr SimpleCubicField[T]](context.ptrs[1])
      let fy = cast[ptr SimpleCubicField[T]](context.ptrs[2])
      fr[][n] := `op`(fx[][n], fy[][n])
    return fr
  proc `op`*[T](fx: SimpleCubicField[T]; y: T): SimpleCubicField[T] {.inline.} =
    var ctx = newParallelForContext(fx.numVecSites, flts = @[float(y)])
    var fr = fx.lattice[].newField(T)
    ctx.pack(fr, fx)
    ctx.each(n):
      let fr = cast[ptr SimpleCubicField[T]](context.ptrs[0])
      let fx = cast[ptr SimpleCubicField[T]](context.ptrs[1])
      fr[][n] := `op`(fx[][n], newSIMXVec(T(context.flts[0])))
    return fr
  proc `op`*[T](x: T; fy: SimpleCubicField[T]): SimpleCubicField[T] {.inline.} =
    var ctx = newParallelForContext(fy.numVecSites, flts = @[float(x)])
    var fr = fy.lattice[].newField(T)
    ctx.pack(fr, fy)
    ctx.each(n):
      let fr = cast[ptr SimpleCubicField[T]](context.ptrs[0])
      let fy = cast[ptr SimpleCubicField[T]](context.ptrs[1])
      fr[][n] := `op`(fy[][n], newSIMXVec(T(context.flts[0])))
    return fr
  proc `ope`*[T](fx: var SimpleCubicField[T]; fy: SimpleCubicField[T]) {.inline.} =
    conformable(fx, fy)
    var ctx = newParallelForContext(fx.numVecSites)
    ctx.pack(fx, fy)
    ctx.each(n):
      let fx = cast[ptr SimpleCubicField[T]](context.ptrs[0])
      let fy = cast[ptr SimpleCubicField[T]](context.ptrs[1])
      `ope`(fx[][n], fy[][n])
  proc `ope`*[T](fx: var SimpleCubicField[T]; y: T) {.inline.} =
    var ctx = newParallelForContext(fx.numVecSites, flts = @[float(y)])
    ctx.pack(fx)
    ctx.each(n):
      let fx = cast[ptr SimpleCubicField[T]](context.ptrs[0])
      `ope`(fx[][n], newSIMXVec(T(context.flts[0])))
defineBinaryOperations(`+`, `+=`) # addition
defineBinaryOperations(`-`, `-=`) # subtraction
defineBinaryOperations(`*`, `*=`) # multiplication
defineBinaryOperations(`/`, `/=`) # division

# ... next: proper "threads" block...

# ... !!!ALSO!!! a parallel reduce (sum):
#     I imagine that this will store the result of a 
#     Kokkos reduction in a GlobalPointer, followed by
#     a distributed reduction across all ranks...

# demonstration/testing
when isMainModule:
  import runtime
  import lattice/[latticeconcept]
  const verbosity = 1
  reliq:
    let l = newSimpleCubicLattice([8, 8, 8, 16])
    var field = l.newField(float)
    var 
      fa = l.newField(float)
      fb = l.newField(float)
      fc = l.newField(float)

    if compiles(Lattice(l, seq[int], float)): 
      print "SimpleCubicLattice conforms to Lattice concept"
    else: print "SimpleCubicLattice does not conform to Lattice concept"

    discard field.lattice()

    print field[0], "before"
    field[0] := newSIMXVec(1.0)
    print field[0], "after"

    print field[1], "before"
    field := 2.0
    print field[1], "after"

    fa := 2.0
    fb := 3.0
    #[
    fc := fa + fb
    print fc[0], "after addition"
    fc := fa - fb
    print fc[0], "after subtraction"
    fc := fa * fb
    print fc[0], "after multiplication"
    fc := fa / fb
    print fc[0], "after division"

    fc += fa
    print fc[0], "after addition assignment"
    fc -= fb
    print fc[0], "after subtraction assignment"
    fc *= fa
    print fc[0], "after multiplication assignment"
    fc /= fb
    print fc[0], "after division assignment"

    fc := fa + 1.0
    print fc[0], "after addition with scalar"
    fc := 1.0 + fb
    print fc[0], "after addition with scalar"
    fc := fa - 1.0
    print fc[0], "after subtraction with scalar"
    fc := 1.0 - fb
    print fc[0], "after subtraction with scalar"
    fc := fa * 2.0
    print fc[0], "after multiplication with scalar"
    fc := 2.0 * fb
    print fc[0], "after multiplication with scalar"
    fc := fa / 2.0
    print fc[0], "after division with scalar"
    fc := 2.0 / fb
    print fc[0], "after division with scalar"

    fc += 1.0
    print fc[0], "after addition assignment with scalar"
    fc -= 1.0
    print fc[0], "after subtraction assignment with scalar"
    fc *= 2.0
    print fc[0], "after multiplication assignment with scalar"
    fc /= 2.0
    print fc[0], "after division assignment with scalar"
    ]#