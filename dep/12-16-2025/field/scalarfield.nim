#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/field/scalarfield.nim
  Contact: reliq-lft@proton.me

  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 reliq-lft
  
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
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
import std/[tables]
import std/[strutils]
import std/[math, complex]

import reliq

type Field*[D: static[int], T] = object
  lattice: ref SimpleCubicLattice[D]
  when isComplex32(T):
    fieldRe*: GlobalArray[D, float32]
    fieldIm*: GlobalArray[D, float32]
  elif isComplex64(T):
    fieldRe*: GlobalArray[D, float64]
    fieldIm*: GlobalArray[D, float64]
  else:
    field*: GlobalArray[D, T]

#[ constructor ]#

proc newField*[D: static[int]](
  lattice: SimpleCubicLattice[D],
  T: typedesc
): Field[D, T] =
  ## Create a new Field
  ##
  ## Parameters:
  ## - `lattice`: SimpleCubicLattice on which the field is defined
  ##
  ## Returns:
  ## A new Field instance
  ## 
  ## Example:
  ## ```nim
  ## let lattice = newSimpleCubicLattice([8, 8, 8, 16])
  ## let field = lattice.newField: float
  ## let cfield = lattice.newField: Complex64
  ## ```
  let dimensions = lattice.dimensions
  let mpiGrid = lattice.mpiGrid
  let ghostGrid = lattice.ghostGrid

  when isComplex32(T):
    result = Field[D, T](
      fieldRe: newGlobalArray(dimensions, mpiGrid, ghostGrid, float32),
      fieldIm: newGlobalArray(dimensions, mpiGrid, ghostGrid, float32)
    )
  elif isComplex64(T):
    result = Field[D, T](
      fieldRe: newGlobalArray(dimensions, mpiGrid, ghostGrid, float64),
      fieldIm: newGlobalArray(dimensions, mpiGrid, ghostGrid, float64)
    )
  else: result = Field[D, T](field: newGlobalArray(dimensions, mpiGrid, ghostGrid, T))

  new result.lattice
  result.lattice[] = lattice

#[ downcasting to local view ]#

template getLattice*[D: static[int], T](field: Field[D, T]): SimpleCubicLattice[D] =
  ## Get the lattice associated with the field
  ##
  ## Parameters:
  ## - `field`: Field instance
  ##
  ## Returns:
  ## The SimpleCubicLattice associated with the field
  field.lattice[]

proc numSites*[D: static[int], T](field: Field[D, T]): int =
  ## Get the number of local sites in the field
  ##
  ## Parameters:
  ## - `field`: Field instance
  ##
  ## Returns:
  ## The total number of local sites in the field
  when isComplex32(T) or isComplex64(T): 
    return field.fieldRe.numSites()
  else: 
    return field.field.numSites()

proc localField*[D: static[int], T](field: Field[D, T]): auto =
  ## Get the local view of the field
  ##
  ## Parameters:
  ## - `field`: Field instance
  ##
  ## Returns:
  ## A LocalArray representing the local portion of the field
  when isComplex32(T) or isComplex64(T):
    # For complex fields, return a complex LocalArray
    newLocalArray(field.fieldRe, field.fieldIm)
  else: 
    # For real fields, return a real LocalArray
    field.field.newLocalArray()

proc localField*(x: SomeInteger | SomeFloat): auto = x
proc localField*[T](x: Complex[T]): auto = x

proc `[]`*(x: SomeInteger | SomeFloat; n: SomeInteger): auto = x
proc `[]`*[T](x: Complex[T]; n: SomeInteger): auto = x

#[ virtual attributes ]#

#[ type conversions ]#

template toComplex*[D: static[int], T](f: Field[D, T]): Field[D, Complex[T]] =
  ## Convert a real field to a complex field (imaginary part = 0)
  ##
  ## Parameters:
  ## - `field`: Real Field instance
  ##
  ## Returns:
  ## A complex Field with real part = field, imaginary part = 0
  var result = f.getLattice().newField: Complex[T]
  let fview = f.localField()
  var rview = result.localField()
  for i in every 0..<f.numSites(): rview[i] = complex(fview[i], 0.0)
  result

template realPart*[D: static[int], T](
  f: Field[D, Complex[T]]
): Field[D, T] =
  ## Extract the real part of a complex field
  ##
  ## Parameters:
  ## - `field`: Complex Field instance
  ##
  ## Returns:
  ## A real Field containing the real parts
  var result = f.getLattice().newField: T
  let fview = f.localField()
  var rview = result.localField()
  for i in every 0..<f.numSites(): rview[i] = fview[i].re
  result

template imagPart*[D: static[int], T](
  f: Field[D, Complex[T]]
): Field[D, T] =
  ## Extract the imaginary part of a complex field
  ##
  ## Parameters:
  ## - `field`: Complex Field instance
  ##
  ## Returns:
  ## A real Field containing the imaginary parts
  var result = f.getLattice().newField: T
  let fview = f.localField()
  var rview = result.localField()
  for i in every 0..<f.numSites(): rview[i] = fview[i].im
  result

#[ promotion ]#

# Stub operators for field arithmetic - these exist so that expressions like
# "field1 * field2" can be written and passed to the promotion macro.
# They should never actually be called at runtime - the promotion macro
# intercepts them and generates fused loops instead.
template `*`*[D: static[int], T](a, b: Field[D, T]): Field[D, T] =
  {.error: "Field arithmetic operators should only be used within := promotion context".}

template `/`*[D: static[int], T](a, b: Field[D, T]): Field[D, T] =
  {.error: "Field arithmetic operators should only be used within := promotion context".}

template `+`*[D: static[int], T](a, b: Field[D, T]): Field[D, T] =
  {.error: "Field arithmetic operators should only be used within := promotion context".}

template `-`*[D: static[int], T](a, b: Field[D, T]): Field[D, T] =
  {.error: "Field arithmetic operators should only be used within := promotion context".}

# collects identifiers from syntax tree and transforms them into view declarations
proc declViews(assn: var seq[NimNode]; repls: var Table[string, string]; node: NimNode) =
  when defined(MACRO_DEBUG): echo node.repr, " -> ", node.kind
  case node.kind:
    of nnkIdent, nnkSym, nnkBracketExpr, nnkDotExpr, nnkCall: # treat whole expr as identifier
      let identStr = node.repr
      let newIdentStrA = node.repr.replace(".", "_") # dot
      let newIdentStrB = newIdentStrA.replace("[", "").replace("]", "") # bracket
      let newIdentStrC = newIdentStrB.replace(",", "_").replace(" ", "") # comma/space
      let newIdentStrD = newIdentStrC.replace("(", "_").replace(")", "") # paren
      let newIdentViewStr = newIdentStrD & "View"
      if not repls.hasKey(identStr):
        let ident = newIdentNode(newIdentViewStr)
        let local = newCall(newIdentNode("localField"), node) 
        assn.add newVarStmt(ident, local) # compiles to "var nodev = localField(node)"
        repls[identStr] = newIdentViewStr # store new name to make sure no repeats
    of nnkHiddenDeref:
      # Unwrap hidden dereference and process the inner node
      assn.declViews(repls, node[0])
    of nnkInfix: # if infix, recurse both operands
      assn.declViews(repls, node[1]) 
      assn.declViews(repls, node[2])
    of nnkPrefix: # prefix operators like -x
      assn.declViews(repls, node[1])
    of nnkPar: # parenthesized expressions
      assn.declViews(repls, node[0])
    else: 
      when defined(MACRO_DEBUG): echo "  (ignoring node kind: ", node.kind, ")"

# transform rhs AST into indexed access of views
proc promoteAST(repls: Table[string, string]; node: NimNode): NimNode =
  result = case node.kind
    of nnkIdent, nnkSym, nnkBracketExpr, nnkDotExpr, nnkCall:
      let key = node.repr  # Use same repr format as declViews
      if repls.hasKey(key): 
        newTree(nnkBracketExpr, newIdentNode(repls[key]), newIdentNode("n"))
      else: node
    of nnkHiddenDeref: # Unwrap hidden dereference and promote the inner node
      promoteAST(repls, node[0])
    of nnkInfix:
      let (lhs, rhs) = (promoteAST(repls, node[1]), promoteAST(repls, node[2]))
      newTree(nnkInfix, node[0], lhs, rhs)
    of nnkPrefix:
      let operand = promoteAST(repls, node[1])
      newTree(nnkPrefix, node[0], operand)
    of nnkPar: # parenthesized expressions
      promoteAST(repls, node[0])
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
    # newLHS should be view[n], extract view and n to create `[]=`(view, n, rhs)
    if newLHS.kind == nnkBracketExpr and newLHS.len == 2:
      let viewNode = newLHS[0]  # The view identifier
      let indexNode = newLHS[1]  # The n identifier
      newExpr = newCall(ident"[]=", viewNode, indexNode, newRHS)
    else: error("Assignment failed: LHS did not promote to view[n] pattern. Got: " & newLHS.repr, lhs)
  else: newExpr = newTree(nnkInfix, newIdentNode($ident), newLHS, newRHS)

  # combine steps 1 & 2 into concrete AST
  result = quote do:
    `lhsViews`
    `rhsViews`
    forevery(0, `lhs`.numSites(), n): `newExpr`

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

#[ mathematical operations/functions ]#

template conjugate*[D: static[int], T](f: Field[D, Complex[T]]): Field[D, Complex[T]] =
  ## Compute the complex conjugate of a field
  ##
  ## Parameters:
  ## - `field`: Complex Field instance
  ##
  ## Returns:
  ## A complex Field with conjugated values
  var result = newField(field.lattice): Complex[T]
  let fview = f.localField()
  var rview = result.localField()
  for i in every 0..<f.numSites():
    let val = fview[i]
    rview[i] = complex(val.re, -val.im)
  result

template absSquared*[D: static[int], T](f: Field[D, Complex[T]]): Field[D, T] =
  ## Compute |z|² = re² + im² for each element
  ##
  ## Parameters:
  ## - `field`: Complex Field instance
  ##
  ## Returns:
  ## A real Field containing |z|² values
  var result = newField(f.lattice): T
  let fview = f.localField()
  var rview = result.localField()
  for i in every 0..<f.numSites():
    let val = fview[i]
    rview[i] = val.re * val.re + val.im * val.im
  result

template abs*[D: static[int], T](f: Field[D, Complex[T]]): Field[D, T] =
  ## Compute |z| = sqrt(re² + im²) for each element
  ##
  ## Parameters:
  ## - `field`: Complex Field instance
  ##
  ## Returns:
  ## A real Field containing |z| values
  var result = f.absSquared()
  result = result.sqrt()
  result

#[ unit tests ]#

test:
  let lattice = newSimpleCubicLattice([8, 8, 8, 16])
  var field = lattice.newField: float
  var cfield = lattice.newField: Complex64
  var fieldView = field.localField()
  var cfieldView = cfield.localField()

  # test conversion

  var fieldRe = cfield.realPart()
  var fieldIm = cfield.imagPart()

  echo "field conversion test passed"

  # test assignment

  field := 2.0
  
  for i in 0..<field.numSites():
    assert fieldView[i] == 2.0

  var fieldA = lattice.newField: float
  var fieldB = lattice.newField: float
  var fieldC = lattice.newField: float
  var fieldD = lattice.newField: float
  
  fieldA := 2.0
  fieldB := 3.0
  fieldC := 4.0
  fieldD := fieldA + fieldB*fieldC - 5.0

  var fieldDView = fieldD.localField()
  for i in 0..<fieldD.numSites():
    assert fieldDView[i] == 9.0
  
  echo "Before +=: fieldD[0] = ", fieldDView[0], " on process ", myRank()
  
  fieldD += 1.0
  fieldDView = fieldD.localField()  # Create fresh view after operation
  
  echo "After +=: fieldD[0] = ", fieldDView[0], " on process ", myRank()
  
  for i in 0..<fieldD.numSites():
    assert fieldDView[i] == 10.0, $fieldDView[i] & " != 10.0" & " at index " & $i
  
  fieldD -= 2.0
  fieldDView = fieldD.localField()  # Create fresh view after operation
  for i in 0..<fieldD.numSites():
    assert fieldDView[i] == 8.0
  
  fieldD *= 2.0
  fieldDView = fieldD.localField()  # Create fresh view after operation
  for i in 0..<fieldD.numSites():
    assert fieldDView[i] == 16.0
  
  fieldD /= 4.0
  for i in 0..<fieldD.numSites():
    assert fieldDView[i] == 4.0
  
  fieldD *= fieldA + fieldB
  for i in 0..<fieldD.numSites():
    assert fieldDView[i] == 20.0
  
  fieldD /= fieldC + 1.0
  for i in 0..<fieldD.numSites():
    assert fieldDView[i] == 4.0

  var cfieldA = lattice.newField: Complex64
  var cfieldB = lattice.newField: Complex64
  var cfieldC = lattice.newField: Complex64
  var cfieldD = lattice.newField: Complex64

  cfieldA := complex(2.0, 1.0)
  cfieldB := complex(3.0, -1.0)
  cfieldC := complex(4.0, 0.5)
  cfieldD := cfieldA + cfieldB*cfieldC - complex(5.0, 0.0)

  var ca = complex(2.0, 1.0)
  var cb = complex(3.0, -1.0)
  var cc = complex(4.0, 0.5)
  var cd = ca + cb*cc - complex(5.0, 0.0)

  var cfieldDView = cfieldD.localField()
  for i in 0..<cfieldD.numSites():
    assert cfieldDView[i] == cd
  
  cfieldD := fieldD

  for i in 0..<cfieldD.numSites():
    assert cfieldDView[i] == complex(4.0, 0.0)
  
  echo "field assignment test passed"

  echo "scalarfield.nim tests passed"
