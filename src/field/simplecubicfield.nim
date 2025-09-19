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

proc newFieldData[T](f: SimpleCubicField[T]): auto =
  let sizeSIMD = sizeof(SIMDArray[T])
  return cast[ptr UncheckedArray[SIMDArray[T]]](alloc(f.local()[].len * sizeSIMD))

proc newField*(l: SimpleCubicLattice; T: typedesc): auto =
  ## Create new field on simple cubic Bravais lattice
  ##
  ## <in need of documentation>
  result = DistributedSimpleCubicField[T](
    lattice: l.local(),
    len: numThreads() * l.numVectorLaneSites
  ).newGlobalPointer()
  result.local()[].data = result.newFieldData()

#[ frontend: "virtual attribute" accessors ]#

template sites*[T](f: SimpleCubicField[T]): untyped =
  ## Gets number of local lattice sites
  f.local()[].lattice[].numVectorLaneSites 

#[ frontend: view converters ]#

template autoView*[T](symbol: untyped; f: SimpleCubicField[T]): untyped =
  ## Create Kokkos static view of field data
  ##
  ## <in need of documentation>
  var symbol = newStaticView(f.local()[].len, addr f.local()[].data[][0])

proc newView*[T](f: SimpleCubicField[T]): StaticView[SIMDArray[T]] =
  ## Create Kokkos static view of field data
  ##
  ## <in need of documentation>
  return newStaticView(f.local()[].len, addr f.local()[].data[][0])

#[ frontend: methods/procedures ]#

proc `:=`*[T](fx: var SimpleCubicField[T], fy: SimpleCubicField[T]) =
  ## Assign field fy to field fx
  ##
  ## <in need of documentation>
  var fxView = fx.newView()
  var fyView = fy.newView()
  threads:
    for i in 0..<10:
      discard fxView[i]

# ideas:
# - i'd like to have some notion of Chapel's promotion of scalar to array types 
# - use atomics to define elementary field operations?
# - have an internal "each" procedure that distributes
#   iterations across each shared-memory subdomain

when isMainModule:
  import runtime
  reliq:
    let 
      geometry = [8, 8, 8, 16]
      lattice = geometry.newSimpleCubicLattice() 
    var 
      fieldA = lattice.newField(float)
      fieldB = lattice.newField(float)
    autoView(fieldAView, fieldA) # like in Grid!
    var fieldBView = fieldB.newView()
    fieldA := fieldB
    print fieldAView[0]