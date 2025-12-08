import reliq
import arrays

import std/[macros, tables]

import utils/[nimutils]
import lattice/[simplecubiclattice]
import kokkos/[kokkosdispatch]

type SimpleCubicField*[D: static[int], T] = object
  ## Simple cubic field implementation
  ##
  ## Represents a field defined on a simple cubic lattice.
  lattice*: SimpleCubicLattice[D]
  field: GlobalArray[D, T]

#[ field constructor ]#

proc newSimpleCubicField*[D: static[int]](
  lattice: SimpleCubicLattice[D],
  T: typedesc
): SimpleCubicField[D, T] =
  ## Create a new SimpleCubicField
  ##
  ## Parameters:
  ## - `lattice`: SimpleCubicLattice on which the field is defined
  ##
  ## Returns:
  ## A new SimpleCubicField instance
  ## 
  ## Example:
  ## ```nim
  ## let lattice = newSimpleCubicLattice([8, 8, 8, 16])
  ## let field = newSimpleCubicField[int](lattice)
  ## ```
  let dimensions = lattice.dimensions
  let mpiGrid = lattice.mpiGrid
  let ghostGrid = lattice.ghostGrid
  return SimpleCubicField[D, T](
    lattice: lattice, 
    field: newGlobalArray(dimensions, mpiGrid, ghostGrid, T)
  )

#[ virtual accessors ]#

proc numSites*[D: static[int], T](field: SimpleCubicField[D, T]): int =
  ## Get the number of local sites in the field
  ##
  ## Parameters:
  ## - `field`: SimpleCubicField instance
  ##
  ## Returns:
  ## The total number of local sites in the field
  return field.field.numSites()

#[ Chapel-like arithmetic promotion ]#

proc localField*[D: static[int], T](field: SimpleCubicField[D, T]): LocalView[D, T] =
  ## Get the local view of the field
  ##
  ## Parameters:
  ## - `field`: SimpleCubicField instance
  ##
  ## Returns:
  ## A LocalView representing the local portion of the field
  field.field.localView()

proc localField(x: SomeInteger | SomeFloat): auto = x

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
  
  # For assignment operators, generate proper []= call instead of infix =
  if $ident == "=":
    # newLHS is view[n], extract view and n to create `[]=`(view, n, rhs)
    let viewNode = newLHS[0]  # The view identifier
    let indexNode = newLHS[1]  # The n identifier
    newExpr = newCall(ident"[]=", viewNode, indexNode, newRHS)
  else:
    newExpr = newTree(nnkInfix, newIdentNode($ident), newLHS, newRHS)

  # combine steps 1 & 2 into concrete AST
  result = quote do:
    `lhsViews`
    `rhsViews`
    forevery(0, `lhs`.numSites, n): `newExpr`

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
  block: lhs := lhs + rhs

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
  block: lhs := lhs - rhs

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
  block: lhs := lhs*rhs

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
  block: lhs := lhs/rhs

#[ unit tests ]#

test:
  let lattice = newSimpleCubicLattice([8, 8, 8, 8*numRanks()])
  var field1 = newSimpleCubicField(lattice): float
  var field2 = newSimpleCubicField(lattice): float
  var field3 = newSimpleCubicField(lattice): float
  var field4 = newSimpleCubicField(lattice): float
  
  # Test basic field properties
  assert(field1.lattice.dimensions == lattice.dimensions, "Lattice dimensions mismatch")
  assert(field1.field.isInitialized(), "Field GlobalArray not initialized")
  assert(field1.numSites() > 0, "Field should have positive number of local sites")
  
  echo "Process ", myRank(), "/", numRanks(), ": Local sites = ", field1.numSites()
  
  # Test simple assignment
  field1 := 2.0
  field2 := 3.0
  
  # Test arithmetic promotion
  field3 := field1 + field2  # Should be 5.0
  field4 := field1 * field2  # Should be 6.0
  
  # Verify values using local views
  let view3 = field3.localField()
  let view4 = field4.localField()
  
  for i in 0..<field3.numSites():
    assert(abs(view3[i] - 5.0) < 1e-10, "field3 should be 5.0")
    assert(abs(view4[i] - 6.0) < 1e-10, "field4 should be 6.0")
  
  # Test complex expression
  field1 := field2 + 2.0*field3 - field4  # 3.0 + 10.0 - 6.0 = 7.0
  
  let view1 = field1.localField()
  for i in 0..<field1.numSites():
    assert(abs(view1[i] - 7.0) < 1e-10, "Complex expression failed")
  
  # Test compound assignment operators
  field2 := 4.0
  field2 += 2.0  # Should be 6.0
  
  let view2 = field2.localField()
  for i in 0..<field2.numSites():
    assert(abs(view2[i] - 6.0) < 1e-10, "Compound += failed")
  
  field2 *= 2.0  # Should be 12.0
  for i in 0..<field2.numSites():
    assert(abs(view2[i] - 12.0) < 1e-10, "Compound *= failed")
  
  field2 /= 3.0  # Should be 4.0
  for i in 0..<field2.numSites():
    assert(abs(view2[i] - 4.0) < 1e-10, "Compound /= failed")
  
  field2 -= 1.0  # Should be 3.0
  for i in 0..<field2.numSites():
    assert(abs(view2[i] - 3.0) < 1e-10, "Compound -= failed")
  
  echo "Process ", myRank(), "/", numRanks(), ": All SimpleCubicField tests passed!"
