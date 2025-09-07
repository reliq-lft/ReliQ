#[ 
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: src/field/simplecubicfield.nim
  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 Curtis Taylor Peterson
  
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
backend: views()

#[ frontend: simple cubic field type definition ]#

type
  SimpleCubicField*[T] = object
    ## Field on simple cubic Bravais lattice
    ##
    ## <in need of documentation>
    lattice: ptr SimpleCubicLattice
    field: GlobalPointer[SIMXVec[T]]
    fieldView: StaticView[SIMXVec[T]]

#[ frontend: SimpleCubicField constructor ]#

proc newField*(lattice: SimpleCubicLattice; T: typedesc): SimpleCubicField[T] =
  ## Create new field on simple cubic Bravais lattice
  ##
  ## <in need of documentation>
  let numVecSites = csize_t(lattice.numVecSites)
  let 
    field = newGlobalPointerArray(numVecSites, SIMXVec[T])
    fieldView = field.newStaticView(numVecSites)
  result = SimpleCubicField[T](lattice: addr lattice, field: field, fieldView: fieldView)

#[ frontend: implement field concept ]#

proc lattice*[T](f: SimpleCubicField[T]): SimpleCubicLattice =
  ## Return lattice associated with field
  ##
  ## <in need of documentation>
  return f.lattice[]

#[ frontend: basic SimpleCubicField methods ]#

proc `[]`*[T](f: SimpleCubicField[T]; n: int): SIMXVec[T] {.inline.} =
  ## Access field value at given local site
  ##
  ## This is for testing; it is most certainly not optimal or efficient
  ## <in need of documentation>
  return f.fieldView[n]

proc `[]`*[T](f: var SimpleCubicField[T]; n: int): SIMXVec[T] {.inline.} =
  ## Access field value at given local site and allow it to be modified
  ##
  ## This is for testing; it is most certainly not optimal or efficient
  ## <in need of documentation>
  return f.fieldView[n]

when isMainModule:
  import runtime
  import lattice/[latticeconcept]
  const verbosity = 1
  reliq:
    let l = newSimpleCubicLattice([8, 8, 8, 16])
    var field = l.newField(float)

    if compiles(Lattice(l, seq[int], float)): 
      print "SimpleCubicLattice conforms to Lattice concept"
    else: print "SimpleCubicLattice does not conform to Lattice concept"

    discard field.lattice()

    for n in l.sites(): 
      if verbosity > 1: 
        echo "[" & $myRank() & "] " & "site: ", n, " coord: ", field[n]

# ---

#var execPolicyA = newRangePolicy(0, l.numLocalSites)

#[
proc `()`*[T](
  f: SimpleCubicFieldView[T]; 
  n: int
): int {.exportcpp: "operator()".} = n
]#

#[ i want something like this ----v
kokkos:
  parallel_for(execPolicyA, proc(i: int) {.kokkos.} =
    field[i] = float(i)
  )
]#