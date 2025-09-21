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

import std/[macros]

import backend
import lattice/[simplecubiclattice]

# shorten pragmas pointing to UPC++ & Kokkos headers and include field view wrapper
backend: discard

#[ frontend: simple cubic field type definition ]#

type
  DistributedFieldData[T] = object
    vectorStrides, threadStrides, len, bytes: int # metadata
    lattice: ref DistributedSimpleCubicLattice # reference to local lattice
    data: ptr UncheckedArray[T] # local CPU storage
  SimpleCubicField*[T] = GlobalPointer[DistributedFieldData[T]]

#[ frontend: constructors ]#

proc newFieldLattice[T](f: var DistributedFieldData[T]; l: SimpleCubicLattice) =
  new(f.lattice)
  f.lattice[] = l.local()[]

proc newFieldData[T](f: var DistributedFieldData[T]) =
  f.data = cast[ptr UncheckedArray[T]](alloc(f.bytes))

proc newField*(l: SimpleCubicLattice; T: typedesc): auto =
  ## Create new field on simple cubic Bravais lattice
  ##
  ## <in need of documentation>
  let 
    vectorStrides = numLanes()
    threadStrides = vectorStrides * l.numVectorLaneSites
    len = threadStrides * numThreads()
    bytes = len * sizeof(T)
  var field = DistributedFieldData[T](
    threadStrides: threadStrides, 
    vectorStrides: vectorStrides,
    len: len, 
    bytes: bytes
  )
  field.newFieldLattice(l)
  field.newFieldData()
  return field.newGlobalPointer()

#[ backend: destructors and move semantics ]#

# ... do in parallel???

#[ frontend: virtual attributes ]#

proc threadStrides*[T](f: SimpleCubicField[T]): int =
  ## Gets number strides across threads
  return f.local()[].threadStrides

proc vectorStrides*[T](f: SimpleCubicField[T]): int =
  ## Gets number strides across SIMD lanes
  return f.local()[].vectorStrides

proc len*[T](f: SimpleCubicField[T]): int =
  ## Gets number of local lattice sites
  return f.local()[].len

proc numSites*[T](f: SimpleCubicField[T]): int =
  ## Gets number of local lattice sites
  return f.threadStrides div f.vectorStrides

proc data*[T](f: SimpleCubicField[T]): ptr UncheckedArray[T] =
  ## Downcast global pointer to local field data pointer
  return f.local()[].data

#[ frontend: downcasting ]#

proc localField*[T](f: SimpleCubicField[T]): StaticView[T] =
  ## Downcast global pointer to local field data wrapped with Kokkos view
  return (addr f.data[][0]).newStaticView(f.len)

#[ frontend: accessors ]#

proc `[]`*[T](f: SimpleCubicField[T]; n: SomeInteger): T =
  ## Access field element at local index n; cannot mutate
  return f.data[][n]

#[ frontend: Chapel-like promotion/fusion of elementary arithematic operations ]#

macro `:=`*[T](f: var SimpleCubicField[T]; expression: untyped): untyped =
  result = quote do:
    let fv = f.localField()

#[ ### vvvvvvvvvvvvvvv------ will be replaced by promotion macro
#[ frontend: setters ]#

template `<-`*[T](f: var SimpleCubicField[T]; value: T): untyped =
  ## Set all field elements to scalar value
  let 
    strides = f.vectorStrides
    arr = newSIMDArray(value)
  var fdata = f.data
  forall(0, f.numSites() div strides, n): arr.store(addr fdata[][n*strides])

#[ frontend: Chapel-like promotion/fusion of elementary arithematic operations ]#
]#

#[
macro `:=`*[T](f: var SimpleCubicField[T]; exp: untyped): untyped =
  ## Fused, elementwise assignment with promotion for SimpleCubicField
  ## - Casts all SimpleCubicField to localField() views
  ## - Uses forall for parallelism

  # Helper: recursively replace SimpleCubicField symbols with their localField() view
  proc promoteFields(e: NimNode): NimNode =
    case e.kind
    of nnkIdent, nnkSym:
      # Try to detect if it's a SimpleCubicField symbol (by name convention or type info)
      # Here, we conservatively wrap all symbols except the destination
      if $e != $f:
        result = newCall(bindSym"localField", e)
      else:
        result = e
    of nnkCall, nnkInfix, nnkPrefix:
      result = copyNimNode(e)
      for c in e:
        result.add(promoteFields(c))
    else:
      result = e

  let rhs = promoteFields(exp)
  let nSym = genSym(nskLet, "n")
  result = quote do:
    let n = `f`.len
    let destView = `f`.localField()
    forall(0, n, `nSym`):
      destView[`nSym`] = `rhs`[`nSym`]
]#

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
    var localFieldA = fieldA.localField()
    fieldB <- 3.14
    print fieldB[0]