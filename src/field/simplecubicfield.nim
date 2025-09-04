#[ 
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: src/field/simplecubicfield.nim
  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  Notes:
  * What kind of gain (if any) could be achieved by considering inter-node
  connectivity in partitioning of lattice?

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

# shorten pragmas pointing to UPC++ & Kokkos headers and include local 
# field view wrapper
backend: 
  {.pragma: simplecubicfieldview, header: "simplecubicfieldview.hpp".}

#[ frontend: simple cubic field type definition ]#

type
  SimpleCubicField*[T] = object
    ## Field on simple cubic Bravais lattice
    ##
    ## <in need of documentation>
    lattice: ref SimpleCubicLattice
    field: GlobalPointer[T]

type
  SimpleCubicFieldView*[T] {.
    importcpp: "SimpleCubicFieldView", 
    simplecubicfieldview
  .} = object

#[ frontend: SimpleCubicField constructor ]#

proc newField*(lattice: SimpleCubicLattice; T: typedesc): SimpleCubicField[T] =
  ## Create new field on simple cubic Bravais lattice
  ##
  ## <in need of documentation>
  let numLocalSites = csize_t(lattice.numLocalSites)
  result = SimpleCubicField[T](field: newGlobalPointerArray(numLocalSites, T))
  new(result.lattice)
  result.lattice[] = lattice  

#[ frontend: SimpleCubicFieldView constructor ]#

proc newSimpleCubicFieldView[T](field: GlobalPointer[T]): SimpleCubicFieldView[T] {.
  importcpp: "SimpleCubicFieldView(#)", 
  header: "simplecubicfieldview.hpp",
  constructor,
  inline
.}
proc newFieldView*[T](f: SimpleCubicField[T]): SimpleCubicFieldView[T] =
  ## Create new view of field on simple cubic Bravais lattice
  ##
  ## <in need of documentation>
  # Developer note: worth inlining, but do this after you've checked that current
  # state ensures that SimpleCubicField actually implements Field concept
  return newSimpleCubicFieldView(f.field)

#[ frontend: basic SimpleCubicField methods ]#

proc `[]`*[T](f: SimpleCubicField[T]; n: int): T =
  ## Access field value at given local site
  ##
  ## This is for testing; it is most certainly not optimal or efficient
  ## <in need of documentation>
  let lPtr = f.field.local()
  let nIdx = cint(n)
  var value: T
  {.emit: """value_1 = lPtr_1[nIdx_1];""".}
  return value

proc `[]`*[T](f: var SimpleCubicField[T]; n: int): T =
  ## Access field value at given local site and allow it to be modified
  ##
  ## This is for testing; it is most certainly not optimal or efficient
  ## <in need of documentation>
  let lPtr = f.field.local()
  let nIdx = cint(n)
  var value: ptr T
  {.emit: """value_1 = &lPtr_1[nIdx_1];""".}
  return value[]

#[ frontend: basic SimpleCubicFieldView methods ]#

# access field value at given local site
proc `[]`*[T](f: SimpleCubicFieldView[T]; n: int): T =
  ## Access field view value at given local site and allow it to be modified
  ##
  ## This is for testing; it is most certainly not optimal or efficient
  ## <in need of documentation>
  let nIdx = cint(n)
  var field = f
  var value: T
  {.emit: """value_1 = field_1[nIdx_1];""".}
  return value

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

    for n in l.sites(): 
      if verbosity > 1: 
        echo "[" & $myRank() & "] " & "site: ", n, " coord: ", field[n]
    
    var execPolicyA = newRangePolicy(0, l.numLocalSites)

    var fieldView = field.newFieldView()
    let fv1 = fieldView[0]
    var fv2 = fieldView[0]
    print fv1


# ---

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