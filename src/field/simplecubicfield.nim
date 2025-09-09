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

import backend
import utils
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
  let 
    field = newGlobalPointerArray(numVecSites, SIMXVec[T])
    view = field.newStaticView(numVecSites)
  result = SimpleCubicField[T](lattice: addr lattice, field: field, fieldView: view)

#[ frontend: implement field concept ]#

proc lattice*[T](f: SimpleCubicField[T]): SimpleCubicLattice =
  ## Return lattice associated with field
  ##
  ## <in need of documentation>
  return f.lattice[]

#[ frontend: basic SimpleCubicField methods ]#

template sites*[T](f: SimpleCubicField[T]; n: untyped; work: untyped): untyped =
  ## Iterate over all local sites of field
  ##
  ## <in need of documentation>
  for n in 0..<f.lattice[].numVecSites: work

proc `[]`*[T](f: SimpleCubicField[T]; n: SomeInteger): SIMXVec[T] {.inline.} =
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
  f.fieldView[n] = value

proc `=copy`*[T](fx: var SimpleCubicField[T]; fy: SimpleCubicField[T]) {.inline.} =
  ## Copy all field values from another field
  ##
  ## <in need of documentation>
  conformable(fx, fy)
  fx = SimpleCubicField[T](
    lattice: fy.lattice, 
    field: fy.field, 
    fieldView: fy.fieldView
  )

proc `:=`*[T](f: var SimpleCubicField[T]; value: T) {.inline.} =
  ## Set all field values to given scalar value
  ##
  ## <in need of documentation>
  f.sites(n): f[n] = newSIMXVec(value)

# frontend: slick Nim trick for defining binary operations
template defineBinaryOperations*(op: untyped; ope: untyped) =
  proc `op`*[T](fx, fy: SimpleCubicField[T]): SimpleCubicField[T] {.inline.} =
    conformable(fx, fy)
    let l = fx.lattice
    result = l[].newField(T)
    result.sites(n): result[n] = op(fx[n], fy[n])
  proc ope*[T](fx: var SimpleCubicField[T]; fy: SimpleCubicField[T]) {.inline.} =
    conformable(fx, fy)
    fx.sites(n): ope(fx[n], fy[n])
  proc `op`*[T](fx: SimpleCubicField[T]; y: T): SimpleCubicField[T] {.inline.} =
    let l = fx.lattice
    result = l[].newField(T)
    result.sites(n): result[n] = op(fx[n], newSIMXVec(y))
  proc `op`*[T](x: T; fy: SimpleCubicField[T]): SimpleCubicField[T] {.inline.} =
    let l = fy.lattice
    result = l[].newField(T)
    result.sites(n): result[n] = op(newSIMXVec(x), fy[n])
  proc `ope`*[T](fx: var SimpleCubicField[T]; y: T) {.inline.} =
    fx.sites(n): ope(fx[n], newSIMXVec(y))
defineBinaryOperations(`+`, `+=`) # addition
defineBinaryOperations(`-`, `-=`) # subtraction
defineBinaryOperations(`*`, `*=`) # multiplication
defineBinaryOperations(`/`, `/=`) # division

# ... next: proper "threads" block...

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
    field[0] = newSIMXVec(1.0)
    print field[0], "after"

    print field[1], "before"
    field := 2.0
    print field[1], "after"

    fa := 2.0
    fb := 3.0
    fc = fa + fb
    print fc[0], "after addition"
    fc = fa - fb
    print fc[0], "after subtraction"
    fc = fa * fb
    print fc[0], "after multiplication"
    fc = fa / fb
    print fc[0], "after division"

    fc += fa
    print fc[0], "after addition assignment"
    fc -= fb
    print fc[0], "after subtraction assignment"
    fc *= fa
    print fc[0], "after multiplication assignment"
    fc /= fb
    print fc[0], "after division assignment"

    fc = fa + 1.0
    print fc[0], "after addition with scalar"
    fc = 1.0 + fb
    print fc[0], "after addition with scalar"
    fc = fa - 1.0
    print fc[0], "after subtraction with scalar"
    fc = 1.0 - fb
    print fc[0], "after subtraction with scalar"
    fc = fa * 2.0
    print fc[0], "after multiplication with scalar"
    fc = 2.0 * fb
    print fc[0], "after multiplication with scalar"
    fc = fa / 2.0
    print fc[0], "after division with scalar"
    fc = 2.0 / fb
    print fc[0], "after division with scalar"

    fc += 1.0
    print fc[0], "after addition assignment with scalar"
    fc -= 1.0
    print fc[0], "after subtraction assignment with scalar"
    fc *= 2.0
    print fc[0], "after multiplication assignment with scalar"
    fc /= 2.0
    print fc[0], "after division assignment with scalar"

    field.sites(n):
      field[n] := float(n)
      fa[n] := float(n)
      fb[n] := float(2*n)
    print field[0], "after sites"
    print fa[1], "after sites"
    print fb[1], "after sites"