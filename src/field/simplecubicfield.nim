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
  DistributedSimpleCubicField*[T] = object
    len: int
    lattice: ptr DistributedSimpleCubicLattice
    data: ptr UncheckedArray[SIMDArray[T]]
  SimpleCubicField*[T] = GlobalPointer[DistributedSimpleCubicField[T]]

#[ frontend: constructors ]#

proc newFieldData*[T](f: SimpleCubicField[T]): auto =
  return cast[ptr UncheckedArray[SIMDArray[T]]](alloc(f.local()[].len))

proc newField*(l: SimpleCubicLattice; T: typedesc): auto =
  ## Create new field on simple cubic Bravais lattice
  ##
  ## <in need of documentation>
  result = DistributedSimpleCubicField[T](
    lattice: l.local(),
    len: numThreads() * l.numVectorLaneSites * sizeof(SIMDArray[T])
  ).newGlobalPointer()
  result.local()[].data = result.newFieldData()

#[ frontend: methods ]#

template view*[T](symbol: untyped; f: SimpleCubicField[T]): untyped =
  ## Create Kokkos static view of field data
  ##
  ## <in need of documentation>
  var symbol = newStaticView(f.local()[].len, f.local()[].data)

when isMainModule:
  import runtime
  reliq:
    let 
      geometry = [8, 8, 8, 16]
      lattice = geometry.newSimpleCubicLattice() 
    var 
      fieldA = lattice.newField(float)
      fieldB = lattice.newField(float)
    view(fieldAView, fieldA) # like in Grid!