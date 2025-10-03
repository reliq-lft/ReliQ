#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/field/simplecubicfield.nim
  Contact: reliq-lft@proton.me

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
backend: {.experimental: "callOperator".}

#[ frontend: simple cubic field type definition ]#

type
  SimpleCubicScalar*[T] = object
    lattice: SimpleCubicLattice
    data: DistributedArray[T]

type
  SimpleCubicTensor*[T] = object
    order: int
    shape, strides: seq[int]
    data: seq[SimpleCubicScalar[T]]

#[ frontend: constructors ]#

proc newField*(l: SimpleCubicLattice; T: typedesc): SimpleCubicScalar[T] =
  ## Create new field on simple cubic Bravais lattice
  ##
  ## <in need of documentation>
  let numSites = numLanes()*l.numVectorLaneSites*numThreads()
  return SimpleCubicScalar[T](lattice: l, data: newDistributedArray(numSites, T))

#[ testing ]#

when isMainModule:
  import runtime
  import utils/[reliqutils]
  reliq:
    let 
      geometry = [8, 8, 8, 16]
      lattice = geometry.newSimpleCubicLattice() 
    var 
      fieldA = lattice.newField(): float
      fieldB = lattice.newField(): float
      fieldC = lattice.newField(): float
      fieldD = lattice.newField(): float