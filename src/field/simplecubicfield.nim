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

import std/[macros, tables]

import backend
import lattice/[simplecubiclattice]
import utils/[nimutils]

# shorten pragmas pointing to UPC++ & Kokkos headers and include field view wrapper
backend: {.experimental: "callOperator".}

#[ frontend: simple cubic field type definition ]#

type
  FieldData[T] = object
    vectorStrides, threadStrides, len, bytes: int # metadata
    lattice: ref DistributedSimpleCubicLattice # reference to local lattice
    data: ptr UncheckedArray[T] # local CPU storage

type
  SimpleCubicScalar*[T] = GlobalPointer[FieldData[T]]
  SimpleCubicTensor*[T] = object
    order: int # tensor order
    shape, strides: seq[int] # tensor shape and strides
    components: seq[SimpleCubicScalar[T]] # tensor components

#[ frontend: constructors ]#

proc toStrides(shape: openArray[SomeInteger]): seq[int] =
  ## Convert tensor shape to tensor strides
  result = newSeq[int](shape.len)
  result[shape.len-1] = 1
  for i in countdown(shape.len - 2, 0):
    result[i] = result[i + 1] * shape[i + 1]

proc newFieldLattice[T](f: var FieldData[T]; l: SimpleCubicLattice) =
  new(f.lattice)
  f.lattice[] = l.local()[]

proc newFieldData[T](f: var FieldData[T]) =
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
  var field = FieldData[T](
    threadStrides: threadStrides, 
    vectorStrides: vectorStrides,
    len: len, 
    bytes: bytes
  )
  field.newFieldLattice(l)
  field.newFieldData()
  return field.newGlobalPointer()

proc newField*(
  l: SimpleCubicLattice; 
  shape: openArray[SomeInteger]; 
  T: typedesc
): auto =
  ## Create new tensor field on simple cubic Bravais lattice
  ##
  ## <in need of documentation>
  result = SimpleCubicTensor[T](
    order: shape.len,
    shape: shape.toSeq(),
    strides: shape.toStrides(),
    components: newSeq[SimpleCubicScalar[T]](shape.len)
  )
  for i in 0..<shape.len: result.components[i] = l.newField(T)

#[ frontend: virtual scalar field attributes ]#

proc threadStrides*[T](f: SimpleCubicScalar[T]): int =
  ## Gets number strides across threads
  return f.local()[].threadStrides

proc vectorStrides*[T](f: SimpleCubicScalar[T]): int =
  ## Gets number strides across SIMD lanes
  return f.local()[].vectorStrides

proc numSites*[T](f: SimpleCubicScalar[T]): int =
  ## Gets number of local lattice sites
  return f.local()[].len

proc data*[T](f: SimpleCubicScalar[T]): ptr UncheckedArray[T] =
  ## Downcast global pointer to local field data pointer
  return f.local()[].data

#[ frontend: virtual tensor field attributes ]#

proc len*[T](f: SimpleCubicTensor[T]): int =
  ## Gets number of tensor components
  return f.components.len

#[ frontend: downcasting ]#

proc localField*[T](f: SimpleCubicScalar[T]): StaticView[T] =
  ## Downcast global pointer to local field data wrapped with Kokkos view
  return (addr f.data[][0]).newStaticView(f.numSites)

#[ frontend: accessors ]#

proc toLex[T](f: SimpleCubicTensor[T]; ijk: openArray[SomeInteger]): int =
  # Compute tensor component offset from tensor index and tensor strides
  assert(ijk.len == f.strides.len, "tensor index does not match tensor order")
  result = 0
  for i in 0..<ijk.len: result += ijk[i] * f.strides[i]

proc `[]`*[T](f: SimpleCubicScalar[T]; n: SomeInteger): T =
  ## Access field element at local index n; cannot mutate
  return f.data[][n]

proc `[]`*[T](
  f: SimpleCubicTensor[T]; 
  ijk: openArray[SomeInteger]
): SimpleCubicScalar[T] =
  ## Access tensor field component at local index ijk; cannot mutate
  return f.components[f.toLex(ijk)]

proc `[]`*[T](
  f: var SimpleCubicTensor[T]; 
  ijk: openArray[SomeInteger]
): var SimpleCubicScalar[T] =
  ## Access tensor field component at local index ijk; can mutate
  return f.components[f.toLex(ijk)]

proc `[]=`*[T](
  f: var SimpleCubicTensor[T]; 
  ijk: openArray[SomeInteger]; 
  value: SimpleCubicScalar[T]
) =
  ## Assign tensor field component at local index ijk
  f.components[f.toLex(ijk)] = value

proc `()`*[T](
  f: SimpleCubicTensor[T]; 
  ijk: openArray[SomeInteger]
): StaticView[T] =
  ## Access tensor field component at local index ijk as view; cannot mutate
  return f[ijk].localField()

proc `()`*[T](
  f: var SimpleCubicTensor[T]; 
  ijk: openArray[SomeInteger]
): StaticView[T] =
  ## Access tensor field component at local index ijk as view; can mutate
  return f[ijk].localField()

#[ backend: Chapel-like promotion/fusion of elementary arithematic operations ]#

# local field for ordinary types; just the identity
proc localField(x: SomeInteger | SomeFloat): auto = x

# field accessor for StaticView types; just the identity
proc `[]`(x: SomeInteger | SomeFloat; n: SomeInteger): auto = x

# collects identifiers from syntax tree and transforms them into view declarations
proc declViews(assn: var seq[NimNode]; repls: var Table[string, string]; node: NimNode) =
  case node.kind:
    of nnkIdent: # if ident, declare view if not already done
      let (identStr, newIdentStr) = ($node, $node & "View")
      if not repls.hasKey(identStr):
        assn.add newVarStmt(
          newIdentNode(newIdentStr),
          newCall(newIdentNode("localField"), newIdentNode(identStr)) 
        ) # compiles to "var nodev = localField(node)"
        repls[identStr] = newIdentStr # store new name to make sure no repeats
    of nnkInfix: # if infix, recurse
      assn.declViews(repls, node[1]) 
      assn.declViews(repls, node[2])
    else: discard

# transform rhs AST into indexed access of views
proc promoteAST(repls: Table[string, string]; node: NimNode): NimNode =
  result = case node.kind
    of nnkIdent:
      if repls.hasKey($node): 
        newTree(nnkBracketExpr, newIdentNode(repls[$node]), newIdentNode("n"))
      else: node
    of nnkInfix:
      let (lhs, rhs) = (promoteAST(repls, node[1]), promoteAST(repls, node[2]))
      newTree(nnkInfix, node[0], lhs, rhs)
    else: node

# step 1: collect lhs/rhs identifiers and declare/create views out of them
# step 2: transform rhs AST into parallel loop over views
# TODO: handle calls (well, really calls that grab fields; e.g., from a shift buffer)
# this is absolutely the coolest macro i've ever written
macro promote(ident: untyped; lhs, rhs: untyped): untyped =
  var repls: Table[string, string] = initTable[string, string]()
  var lhsAssn, rhsAssn: seq[NimNode] = @[]
  var lhsViews, rhsViews: NimNode
  var newLHS, newRHS, newExpr: NimNode

  # step 1
  lhsAssn.declViews(repls): lhs
  rhsAssn.declViews(repls): rhs
  lhsViews = newStmtList(lhsAssn)
  rhsViews = newStmtList(rhsAssn)

  # step 2
  newLHS = promoteAST(repls): lhs
  newRHS = promoteAST(repls): rhs
  newExpr = newTree(nnkInfix, newIdentNode($ident), newLHS, newRHS)

  # combine steps 1 & 2 into concrete AST
  result = quote do:
    `lhsViews`
    `rhsViews`
    forevery(0, `lhs`.numSites, n): `newExpr`

#[ frontend: Chapel-like promotion/fusion of elementary arithematic operations ]#

template `:=`*(lhs, rhs: untyped) =
  ## Fused assignment to arbitrary arithematic operation of fields/scalars
  ##
  ## Turns expressions like 
  ## ```
  ## var num = 10.0
  ## fieldA := fieldB + 2.0*fieldB - num*fieldC
  ## ``` 
  ## into 
  ## ```
  ## var fieldAView = fieldA.localField()
  ## var fieldBView = fieldB.localField()
  ## var fieldCView = fieldC.localField()
  ## var numView = num
  ## forevery 0, fieldA.numSites(), n:
  ##   fieldAView[n] = fieldBView[n] + 2.0*fieldBView[n] - numView[n]*fieldCView[n]
  ## ```
  ## As such, this avoids:
  ## * creating temporaries to store the result of each intermediate operation 
  ## * passing through the data with a parallel loop for each intermediate operation.
  ## Note that we learned this trick from the Chapel programming language; specifically,
  ## we owe Bradford Chamberlain much gratitude for pointing this feature of Chapel out
  ## to us and hence inspiring us to implement a version of it in ReliQ. For information
  ## about Chapel's promotion/fusion, see: 
  ## * https://chapel-lang.org/docs/users-guide/datapar/promotion.html
  ## Additionally, notice that we've "numView[n]"; your eyes do not deceive you:
  ## scalars are given a dummy accessor that acts as the identity. 
  block: `=`.promote(lhs, rhs)

template `+=`*(lhs, rhs: untyped) =
  ## Element-wise promotion of `+=` operator
  ## 
  ## Turns expressions like 
  ## ```
  ## fieldA += fieldB
  ## ``` 
  ## into 
  ## ```
  ## var fieldAView = fieldA.localField()
  ## var fieldBView = fieldB.localField()
  ## forevery 0, fieldA.numSites(), n:
  ##   fieldAView[n] += fieldBView[n]
  ## ```
  block: `+=`.promote(lhs, rhs)

template `-=`*(lhs, rhs: untyped) =
  ## Element-wise promotion of `-=` operator
  ## 
  ## Turns expressions like 
  ## ```
  ## fieldA -= fieldB
  ## ``` 
  ## into 
  ## ```
  ## var fieldAView = fieldA.localField()
  ## var fieldBView = fieldB.localField()
  ## forevery 0, fieldA.numSites(), n:
  ##   fieldAView[n] -= fieldBView[n]
  ## ```
  block: `-=`.promote(lhs, rhs)

template `*=`*(lhs, rhs: untyped) =
  ## Element-wise promotion of `*=` operator
  ## 
  ## Turns expressions like 
  ## ```
  ## fieldA *= fieldB
  ## ``` 
  ## into 
  ## ```
  ## var fieldAView = fieldA.localField()
  ## var fieldBView = fieldB.localField()
  ## forevery 0, fieldA.numSites(), n:
  ##   fieldAView[n] *= fieldBView[n]
  ## ```
  block: `*=`.promote(lhs, rhs)

template `/=`*(lhs, rhs: untyped) =
  ## Element-wise promotion of `/=` operator
  ## 
  ## Turns expressions like 
  ## ```
  ## fieldA /= fieldB
  ## ``` 
  ## into 
  ## ```
  ## var fieldAView = fieldA.localField()
  ## var fieldBView = fieldB.localField()
  ## forevery 0, fieldA.numSites(), n:
  ##   fieldAView[n] /= fieldBView[n]
  ## ```
  block: `/=`.promote(lhs, rhs)

#[ frontend: tensor methods ]#

proc `⊗`*[T](tensorA, tensorB: SimpleCubicTensor[T]): SimpleCubicTensor[T] =
  ## Tensor product of two tensor fields
  ##
  ## <in need of documentation>
  #[
  assert(tensorA.order > 0 and tensorB.order > 0, "tensor order must be positive")
  result = SimpleCubicTensor[T](
    order: tensorA.order + tensorB.order,
    shape: tensorA.shape & tensorB.shape,
    strides: tensorA.strides & tensorB.strides,
    components: newSeq[SimpleCubicScalar[T]](tensorA.components.len * tensorB.components.len)
  )
  for i in 0..<tensorA.components.len:
    for j in 0..<tensorB.components.len:
      result.components[i*tensorB.components.len + j] = 
        tensorA.components[i] * tensorB.components[j]
  ]#
  return tensorA

#[ testing ]#

# ideas:
# - i'd like to have some notion of Chapel's promotion of scalar to array types 
# - use atomics to define elementary field operations?
# - have an internal "each" procedure that distributes
#   iterations across each shared-memory subdomain
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
    var localFieldA = fieldA.localField()
    let 
      num = 10.0
      fav = 5.0 + 2.0*5.0 - num*9.0 + 3.0*5.0/9.0 + num*9.0*5.0
    fieldB := 5.0
    fieldC := 9.0 
    fieldA := fieldB + 2.0*fieldB - num*fieldC + 3.0*fieldB/fieldC + num*fieldC*fieldB
    forall(0, fieldA.numSites, n):
      if fieldB[n] != 5.0: 
        print "fieldB[", n, "] = ", fieldB[n]
        assert(fieldB[n] == 5.0)
      if fieldC[n] != 9.0: 
        print "fieldC[", n, "] = ", fieldC[n]
        assert(fieldC[n] == 9.0)
      if fieldA[n] != fav:
        print "fieldA[", n, "] = ", fieldA[n]
        assert(fieldA[n] == fav)
    fieldD += fieldA
    assert(fieldD[0] == fav)
    fieldD += 2.0
    assert(fieldD[0] == fav + 2.0)
    fieldD += num
    assert(fieldD[0] == fav + 2.0 + num)
    fieldD *= fieldA
    assert(fieldD[0] == (fav + 2.0 + num)*fav)
    fieldD *= 2.0
    assert(fieldD[0] == (fav + 2.0 + num)*fav*2.0)
    fieldD *= num
    assert(fieldD[0] == (fav + 2.0 + num)*fav*2.0*num)
    fieldD -= fieldA
    assert(fieldD[0] == (fav + 2.0 + num)*fav*2.0*num - fav)
    fieldD -= 2.0
    assert(fieldD[0] == (fav + 2.0 + num)*fav*2.0*num - fav - 2.0)
    fieldD -= num
    assert(fieldD[0] == (fav + 2.0 + num)*fav*2.0*num - fav - 2.0 - num)
    fieldD /= fieldA
    assert(fieldD[0] == ((fav + 2.0 + num)*fav*2.0*num - fav - 2.0 - num)/fav)
    fieldD /= 2.0
    assert(fieldD[0] == (((fav + 2.0 + num)*fav*2.0*num - fav - 2.0 - num)/fav)/2.0)
    fieldD /= num
    assert(fieldD[0] == ((((fav + 2.0 + num)*fav*2.0*num - fav - 2.0 - num)/fav)/2.0)/num)
    fieldD = fieldA
    #`=move`(fieldC, fieldA)
    #`=copy`(fieldB, fieldC)
    #`=destroy`(fieldB)

    #var 
    #  tensorA = lattice.newField([3,3]): float
    #  tensorB = lattice.newField([3,3]): float
    #  tensorC = lattice.newField([3,3]): float
    #let 
    #  tensorAComp = tensorA[[1,2]]
    #  tensorACompView = tensorA([1,2])
    #tensorA[[1,1]] = fieldA