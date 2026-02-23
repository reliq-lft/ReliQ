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

import reliq

import std/[macros]
import std/[tables]
import std/[strutils]
import std/[math]

import types/[complex]

type SimpleCubicField*[D: static[int], T] = object
  ## Simple cubic field implementation
  ##
  ## Represents a field defined on a simple cubic lattice.
  ## For complex fields (T = Complex[F]), stores real and imaginary parts as separate GlobalArrays.
  ## For real fields (T = SomeFloat), stores directly as a single GlobalArray.
  lattice*: SimpleCubicLattice[D]
  when isComplex32(T):
    fieldRe*: GlobalArray[D, float32]
    fieldIm*: GlobalArray[D, float32]
  elif isComplex64(T):
    fieldRe*: GlobalArray[D, float64]
    fieldIm*: GlobalArray[D, float64]
  else:
    field*: GlobalArray[D, T]

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
  ## let field = lattice.newSimpleCubicField(float)
  ## let cfield = lattice.newSimpleCubicField(Complex64)
  ## ```
  let dimensions = lattice.dimensions
  let mpiGrid = lattice.mpiGrid
  let ghostGrid = lattice.ghostGrid
  when isComplex32(T):
    return SimpleCubicField[D, T](
      lattice: lattice,
      fieldRe: newGlobalArray(dimensions, mpiGrid, ghostGrid, float32),
      fieldIm: newGlobalArray(dimensions, mpiGrid, ghostGrid, float32)
    )
  elif isComplex64(T):
    return SimpleCubicField[D, T](
      lattice: lattice,
      fieldRe: newGlobalArray(dimensions, mpiGrid, ghostGrid, float64),
      fieldIm: newGlobalArray(dimensions, mpiGrid, ghostGrid, float64)
    )
  else: 
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
  when isComplex(T): return field.fieldRe.numSites()
  else: return field.field.numSites()

#[ type conversion ]#

template toComplex*[D: static[int], T](f: SimpleCubicField[D, T]): SimpleCubicField[D, Complex[T]] =
  ## Convert a real field to a complex field (imaginary part = 0)
  ##
  ## Parameters:
  ## - `field`: Real SimpleCubicField instance
  ##
  ## Returns:
  ## A complex SimpleCubicField with real part = field, imaginary part = 0
  var result = f.lattice.newSimpleCubicField(): Complex[T]
  let fview = f.localSimpleCubicField()
  var rview = result.localSimpleCubicField()
  for i in every 0..<f.numSites(): rview[i] = complex(fview[i], 0.0)
  result

template re*[D: static[int], T](f: SimpleCubicField[D, Complex[T]]): SimpleCubicField[D, T] =
  ## Extract the real part of a complex field
  ##
  ## Parameters:
  ## - `field`: Complex SimpleCubicField instance
  ##
  ## Returns:
  ## A real SimpleCubicField containing the real parts
  var result = f.lattice.newSimpleCubicField(): T
  let fview = f.localSimpleCubicField()
  var rview = result.localSimpleCubicField()
  for i in every 0..<f.numSites(): rview[i] = fview[i].re
  result

template im*[D: static[int], T](f: SimpleCubicField[D, Complex[T]]): SimpleCubicField[D, T] =
  ## Extract the imaginary part of a complex field
  ##
  ## Parameters:
  ## - `field`: Complex SimpleCubicField instance
  ##
  ## Returns:
  ## A real SimpleCubicField containing the imaginary parts
  var result = f.lattice.newSimpleCubicField(): T
  let fview = f.localSimpleCubicField()
  var rview = result.localSimpleCubicField()
  for i in every 0..<f.numSites(): rview[i] = fview[i].im
  result

#[ grid conversion ]#

template toPaddedSimpleCubicField*[D: static[int], T](
  tightSimpleCubicField: SimpleCubicField[D, T],
  ghostGrid: array[D, int]
): SimpleCubicField[D, T] =
  ## Convert a field to its padded version according to its lattice's ghost grid
  ##
  ## Parameters:
  ## - `tightSimpleCubicField`: SimpleCubicField instance
  ## - `ghostGrid`: Array specifying ghost zone sizes in each dimension
  ##
  ## Returns:
  ## A new SimpleCubicField with dimensions padded according to the lattice's ghost grid
  let paddedLattice = newSimpleCubicLattice(
    tightSimpleCubicField.lattice.dimensions,
    tightSimpleCubicField.lattice.mpiGrid,
    ghostGrid
  )
  var paddedSimpleCubicField = newSimpleCubicField(paddedLattice): T

  paddedSimpleCubicField := tightSimpleCubicField

  # halo exchange
  when isComplex(T):
    paddedSimpleCubicField.fieldRe.updateGhosts()
    paddedSimpleCubicField.fieldIm.updateGhosts()
  else: paddedSimpleCubicField.field.updateGhosts()

  paddedSimpleCubicField

template toTightSimpleCubicField*[D: static[int], T](paddedSimpleCubicField: SimpleCubicField[D, T]): SimpleCubicField[D, T] =
  ## Convert a padded field to its tight version by removing ghost zones
  ##
  ## Parameters:
  ## - `paddedSimpleCubicField`: SimpleCubicField instance
  ##
  ## Returns:
  ## A new SimpleCubicField with dimensions tightened by removing ghost zones
  let tightLattice = newSimpleCubicLattice(
    paddedSimpleCubicField.lattice.dimensions,
    paddedSimpleCubicField.lattice.mpiGrid
  )
  var tightSimpleCubicField = newSimpleCubicField(tightLattice): T

  tightSimpleCubicField := paddedSimpleCubicField

  tightSimpleCubicField

template exchange*[D: static[int], T](f: SimpleCubicField[D, T]) =
  ## Perform halo exchange on the field
  ##
  ## Parameters:
  ## - `field`: SimpleCubicField instance
  ##
  ## Returns:
  ## Nothing; performs in-place halo exchange
  when isComplex(T):
    f.fieldRe.updateGhosts()
    f.fieldIm.updateGhosts()
  else: f.field.updateGhosts()

#[ promotion ]#

proc localSimpleCubicField*[D: static[int], T](field: SimpleCubicField[D, T]): auto =
  ## Get the local view of the field
  ##
  ## Parameters:
  ## - `field`: SimpleCubicField instance
  ##
  ## Returns:
  ## A LocalView (real) or ComplexLocalView (complex) representing the local 
  ## portion of the field
  when isComplex32(T):
    ComplexLocalView[D, float32](
      re: field.fieldRe.localView(),
      im: field.fieldIm.localView()
    )
  elif isComplex64(T):
    ComplexLocalView[D, float64](
      re: field.fieldRe.localView(),
      im: field.fieldIm.localView()
    )
  else: field.field.localView()

proc localSimpleCubicField*(x: SomeInteger | SomeFloat): auto = x
proc localSimpleCubicField*[T](x: Complex[T]): auto = x

proc `[]`*(x: SomeInteger | SomeFloat; n: SomeInteger): auto = x
proc `[]`*[T](x: Complex[T]; n: SomeInteger): auto = x

# Stub operators for field arithmetic - these exist so that expressions like
# "field1 * field2" can be written and passed to the promotion macro.
# They should never actually be called at runtime - the promotion macro
# intercepts them and generates fused loops instead.
template `*`*[D: static[int], T](a, b: SimpleCubicField[D, T]): SimpleCubicField[D, T] =
  {.error: "SimpleCubicField arithmetic operators should only be used within := promotion context".}

template `/`*[D: static[int], T](a, b: SimpleCubicField[D, T]): SimpleCubicField[D, T] =
  {.error: "SimpleCubicField arithmetic operators should only be used within := promotion context".}

template `+`*[D: static[int], T](a, b: SimpleCubicField[D, T]): SimpleCubicField[D, T] =
  {.error: "SimpleCubicField arithmetic operators should only be used within := promotion context".}

template `-`*[D: static[int], T](a, b: SimpleCubicField[D, T]): SimpleCubicField[D, T] =
  {.error: "SimpleCubicField arithmetic operators should only be used within := promotion context".}

# collects identifiers from syntax tree and transforms them into view declarations
proc declViews(assn: var seq[NimNode]; repls: var Table[string, string]; node: NimNode) =
  when defined(MACRO_DEBUG): print node.repr, " -> ", node.kind
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
        let local = newCall(newIdentNode("localSimpleCubicField"), node) 
        assn.add newVarStmt(ident, local) # compiles to "var nodev = localSimpleCubicField(node)"
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
      when defined(MACRO_DEBUG): print "  (ignoring node kind: ", node.kind, ")"

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
  ## var fieldAView = fieldA.localSimpleCubicField()
  ## var fieldBView = fieldB.localSimpleCubicField()
  ## var fieldCView = fieldC.localSimpleCubicField()
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
  ## var fieldAView = fieldA.localSimpleCubicField()
  ## var fieldBView = fieldB.localSimpleCubicField()
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
  ## var fieldAView = fieldA.localSimpleCubicField()
  ## var fieldBView = fieldB.localSimpleCubicField()
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
  ## var fieldAView = fieldA.localSimpleCubicField()
  ## var fieldBView = fieldB.localSimpleCubicField()
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
  ## var fieldAView = fieldA.localSimpleCubicField()
  ## var fieldBView = fieldB.localSimpleCubicField()
  ## forevery 0, fieldA.numSites(), n:
  ##   fieldAView[n] /= fieldBView[n]
  ## ```
  block: lhs := lhs/rhs

#[ mathematical operations ]#

template adj*[D: static[int], T](f: SimpleCubicField[D, Complex[T]]): SimpleCubicField[D, Complex[T]] =
  ## Compute the complex conjugate of a field
  ##
  ## Parameters:
  ## - `f`: Complex SimpleCubicField instance
  ##
  ## Returns:
  ## A complex SimpleCubicField with conjugated values
  var result = f.lattice.newSimpleCubicField: Complex[T]
  let fview = f.localSimpleCubicField()
  var rview = result.localSimpleCubicField()
  for i in every 0..<f.numSites():
    let val = fview[i]
    rview[i] = complex(val.re, -val.im)
  result

template norm2*[D: static[int], T](f: SimpleCubicField[D, Complex[T]]): SimpleCubicField[D, T] =
  ## Compute |z|² = re² + im² for each element
  ##
  ## Parameters:
  ## - `f`: Complex SimpleCubicField instance
  ##
  ## Returns:
  ## A real SimpleCubicField containing |z|² values
  var result = f.lattice.newSimpleCubicField: T
  let fview = f.localSimpleCubicField()
  var rview = result.localSimpleCubicField()
  for i in every 0..<f.numSites():
    let val = fview[i]
    rview[i] = val.re * val.re + val.im * val.im
  result

template sqrt*[D: static[int], T](f: SimpleCubicField[D, T]): SimpleCubicField[D, T] =
  ## Element-wise square root of field
  ##
  ## Parameters:
  ## - `f`: SimpleCubicField instance
  ##
  ## Returns:
  ## A new SimpleCubicField with the square root applied element-wise
  var r = f.lattice.newSimpleCubicField: T
  var rv = r.localSimpleCubicField()
  let fv = f.localSimpleCubicField()
  for n in every 0..<f.numSites(): rv[n] = sqrt(fv[n])
  r

template norm*[D: static[int], T](f: SimpleCubicField[D, Complex[T]]): SimpleCubicField[D, T] =
  ## Compute |z| = sqrt(re² + im²) for each element
  ##
  ## Parameters:
  ## - `f`: Complex SimpleCubicField instance
  ##
  ## Returns:
  ## A real SimpleCubicField containing |z| values
  var result = f.norm2()
  result = result.sqrt()
  result

template exp*[D: static[int], T](f: SimpleCubicField[D, T]): SimpleCubicField[D, T] =
  ## Element-wise exponential of field
  ##
  ## Parameters:
  ## - `f`: SimpleCubicField instance
  ##
  ## Returns:
  ## A new SimpleCubicField with the exponential applied element-wise
  var r = f.lattice.newSimpleCubicField: T
  var rv = r.localSimpleCubicField()
  let fv = f.localSimpleCubicField()
  for n in every 0..<f.numSites(): rv[n] = exp(fv[n])
  r

template ln*[D: static[int], T](f: SimpleCubicField[D, T]): SimpleCubicField[D, T] =
  ## Element-wise natural logarithm of field
  ##
  ## Parameters:
  ## - `f`: SimpleCubicField instance
  ##
  ## Returns:
  ## A new SimpleCubicField with the natural logarithm applied element-wise
  var r = f.lattice.newSimpleCubicField: T
  var rv = r.localSimpleCubicField()
  let fv = f.localSimpleCubicField()
  for n in every 0..<f.numSites(): rv[n] = ln(fv[n])
  r

template sin*[D: static[int], T](f: SimpleCubicField[D, T]): SimpleCubicField[D, T] =
  ## Element-wise sine of field
  ##
  ## Parameters:
  ## - `f`: SimpleCubicField instance
  ##
  ## Returns:
  ## A new SimpleCubicField with the sine applied element-wise
  var r = f.lattice.newSimpleCubicField: T
  var rv = r.localSimpleCubicField()
  let fv = f.localSimpleCubicField()
  for n in every 0..<f.numSites(): rv[n] = sin(fv[n])
  r

template cos*[D: static[int], T](f: SimpleCubicField[D, T]): SimpleCubicField[D, T] =
  ## Element-wise cosine of field
  ##
  ## Parameters:
  ## - `f`: SimpleCubicField instance
  ##
  ## Returns:
  ## A new SimpleCubicField with the cosine applied element-wise
  var r = f.lattice.newSimpleCubicField: T
  var rv = r.localSimpleCubicField()
  let fv = f.localSimpleCubicField()
  for n in every 0..<f.numSites(): rv[n] = cos(fv[n])
  r

template tan*[D: static[int], T](f: SimpleCubicField[D, T]): SimpleCubicField[D, T] =
  ## Element-wise tangent of field
  ##
  ## Parameters:
  ## - `f`: SimpleCubicField instance
  ##
  ## Returns:
  ## A new SimpleCubicField with the tangent applied element-wise
  var r = f.lattice.newSimpleCubicField: T
  var rv = r.localSimpleCubicField()
  let fv = f.localSimpleCubicField()
  for n in every 0..<f.numSites(): rv[n] = tan(fv[n])
  r

template abs*[D: static[int], T](f: SimpleCubicField[D, T]): SimpleCubicField[D, T] =
  ## Element-wise absolute value of field
  ##
  ## Parameters:
  ## - `f`: SimpleCubicField instance
  ##
  ## Returns:
  ## A new SimpleCubicField with the absolute value applied element-wise
  var r = f.lattice.newSimpleCubicField: T
  var rv = r.localSimpleCubicField()
  let fv = f.localSimpleCubicField()
  for n in every 0..<f.numSites(): rv[n] = abs(fv[n])
  r

template abs*[D: static[int], T](f: SimpleCubicField[D, Complex[T]]): SimpleCubicField[D, T] =
  ## Element-wise absolute value of complex field
  ##
  ## Parameters:
  ## - `f`: Complex SimpleCubicField instance
  ##
  ## Returns:
  ## A new real SimpleCubicField with the absolute value (magnitude) applied element-wise
  var r = f.lattice.newSimpleCubicField: T
  var rv = r.localSimpleCubicField()
  let fv = f.localSimpleCubicField()
  for n in every 0..<f.numSites(): rv[n] = abs(fv[n])
  r

template `^`*[D: static[int], T](base: SimpleCubicField[D, T], exponent: T): SimpleCubicField[D, T] =
  ## Element-wise power of field
  ##
  ## Parameters:
  ## - `base`: SimpleCubicField instance (the base)
  ## - `exponent`: Exponent value
  ##
  ## Returns:
  ## A new SimpleCubicField with the power applied element-wise
  var r = base.lattice.newSimpleCubicField: T
  var rv = r.localSimpleCubicField()
  let bv = base.localSimpleCubicField()
  for n in every 0..<base.numSites(): rv[n] = pow(bv[n], exponent)
  r

template `^`*[D: static[int], T](base: T, exponent: SimpleCubicField[D, T]): SimpleCubicField[D, T] =
  ## Element-wise power of field
  ##
  ## Parameters:
  ## - `base`: Base value
  ## - `exponent`: SimpleCubicField instance (the exponent)
  ##
  ## Returns:
  ## A new SimpleCubicField with the power applied element-wise
  var r = exponent.lattice.newSimpleCubicField: T
  var rv = r.localSimpleCubicField()
  let ev = exponent.localSimpleCubicField()
  for n in every 0..<exponent.numSites(): rv[n] = pow(base, ev[n])
  r

template `^`*[D: static[int], T](base: SimpleCubicField[D, T], exponent: SimpleCubicField[D, T]): SimpleCubicField[D, T] =
  ## Element-wise power of field
  ##
  ## Parameters:
  ## - `base`: SimpleCubicField instance (the base)
  ## - `exponent`: SimpleCubicField instance (the exponent)
  ##
  ## Returns:
  ## A new SimpleCubicField with the power applied element-wise
  var r = base.lattice.newSimpleCubicField: T
  var rv = r.localSimpleCubicField()
  let bv = base.localSimpleCubicField()
  let ev = exponent.localSimpleCubicField()
  for n in every 0..<base.numSites(): rv[n] = pow(bv[n], ev[n])
  r

#[ reduction ]#

template sumSimpleCubicField*[D: static[int], T](f: SimpleCubicField[D, T]): T =
  ## Compute the sum of all elements in the field
  ##
  ## Parameters:
  ## - `f`: SimpleCubicField instance
  ##
  ## Returns:
  ## The sum of all elements in the field
  let fv = f.localSimpleCubicField()
  when isComplex(T): # Handle complex types by summing re and im parts separately
    when isComplex32(T):
      let realSumLocal = sum[float32](0, f.numSites(), n): fv[n].re
      let imagSumLocal = sum[float32](0, f.numSites(), n): fv[n].im

      let realSumGlobal = gaGlobalSumFloat32(unsafeAddr realSumLocal)
      let imagSumGlobal = gaGlobalSumFloat32(unsafeAddr imagSumLocal)

      complex(realSumGlobal, imagSumGlobal)
    elif isComplex64(T):
      let realSumLocal = sum[float64](0, f.numSites(), n): fv[n].re
      let imagSumLocal = sum[float64](0, f.numSites(), n): fv[n].im

      let realSumGlobal = gaGlobalSumFloat64(unsafeAddr realSumLocal)
      let imagSumGlobal = gaGlobalSumFloat64(unsafeAddr imagSumLocal)

      complex(realSumGlobal, imagSumGlobal)
  else: # Handle real/integer types directly
    let localSum = sum[T](0, f.numSites(), n): fv[n]

    when T is int32: gaGlobalSumInt32(unsafeAddr localSum)
    elif T is int64: gaGlobalSumInt64(unsafeAddr localSum)
    elif T is float32: gaGlobalSumFloat32(unsafeAddr localSum)
    elif T is float64: gaGlobalSumFloat64(unsafeAddr localSum)
    else: {.error: "Unsupported type for global sum reduction".}

template sum*[D: static[int], T](f: SimpleCubicField[D, T]): T =
  ## Compute the sum of all elements in the field
  ##
  ## Parameters:
  ## - `f`: SimpleCubicField instance
  ##
  ## Returns:
  ## The sum of all elements in the field
  sumSimpleCubicField(f)

#[ unit tests ]#

test:
  let lattice = newSimpleCubicLattice(
    [8, 8, 8, 8*numRanks()], 
    [1, 1, 1, numRanks()],
    [0, 0, 0, 0]
  )  # Distribute 8xn across n processes
  let field = lattice.newSimpleCubicField: float64
  let cfield = lattice.newSimpleCubicField: Complex64

  var fieldA = lattice.newSimpleCubicField: float64
  var fieldB = lattice.newSimpleCubicField: float64
  var fieldC = lattice.newSimpleCubicField: float64 
  var fieldD = lattice.newSimpleCubicField: float64

  fieldA := 2.0
  fieldB := 3.0
  fieldC := 4.0
  fieldD := fieldA + fieldB*fieldC - 5.0

  let localD = fieldD.localSimpleCubicField()
  for n in 0..<localD.numSites():
    assert localD[n] == 2.0 + 3.0*4.0 - 5.0, "got: " & $localD[n]
  
  fieldD += 1.0
  for n in 0..<localD.numSites():
    assert localD[n] == 2.0 + 3.0*4.0 - 5.0 + 1.0, "got: " & $localD[n]
  
  fieldD *= 2.0
  for n in 0..<localD.numSites():
    assert localD[n] == (2.0 + 3.0*4.0 - 5.0 + 1.0)*2.0, "got: " & $localD[n]
  
  fieldD /= 4.0
  for n in 0..<localD.numSites():
    assert localD[n] == (2.0 + 3.0*4.0 - 5.0 + 1.0)*2.0 / 4.0, "got: " & $localD[n]

  fieldD -= 0.5 
  for n in 0..<localD.numSites():
    assert localD[n] == (2.0 + 3.0*4.0 - 5.0 + 1.0)*2.0 / 4.0 - 0.5, "got: " & $localD[n]
  
  var cfieldA = lattice.newSimpleCubicField: Complex64
  var cfieldB = lattice.newSimpleCubicField: Complex64
  var cfieldC = lattice.newSimpleCubicField: Complex64
  var cfieldD = lattice.newSimpleCubicField: Complex64

  cfieldA := complex(1.0, -1.0)
  cfieldB := complex(2.0, -2.0)
  cfieldC := complex(3.0, -3.0)
  cfieldD := cfieldA + cfieldB*cfieldC - complex(10.0, -10.0)

  var ca = complex(1.0, -1.0)
  var cb = complex(2.0, -2.0)
  var cc = complex(3.0, -3.0)
  var cd = ca + cb*cc - complex(10.0, -10.0)

  let localCD = cfieldD.localSimpleCubicField()
  for n in 0..<localCD.numSites():
    assert localCD[n] == cd, "got: " & $localCD[n]
  
  cfieldD += complex(1.0, -1.0)
  for n in 0..<localCD.numSites():
    assert localCD[n] == cd + complex(1.0, -1.0), "got: " & $localCD[n]
  
  cfieldD *= complex(2.0, -2.0)
  for n in 0..<localCD.numSites():
    assert localCD[n] == (cd + complex(1.0, -1.0))*complex(2.0, -2.0), "got: " & $localCD[n]
  
  cfieldD /= complex(4.0, -4.0)
  for n in 0..<localCD.numSites():
    assert localCD[n] == (cd + complex(1.0, -1.0))*complex(2.0, -2.0) / complex(4.0, -4.0), "got: " & $localCD[n]
  
  cfieldD := 5.0
  for n in 0..<localCD.numSites():
    assert localCD[n] == complex(5.0, 0.0), "got: " & $localCD[n]
  
  print "field promotion tests passed"

  # Test sum reduction
  fieldA := 1.0
  let sumA = fieldA.sum()
  let expectedSumA = float64(fieldA.numSites()) * float64(numRanks())
  assert abs(sumA - expectedSumA) < 1e-10, "Sum test failed: got " & $sumA & ", expected " & $expectedSumA
  
  fieldB := 2.5
  let sumB = fieldB.sum()
  let expectedSumB = 2.5 * float64(fieldB.numSites()) * float64(numRanks())
  assert abs(sumB - expectedSumB) < 1e-10, "Sum test failed: got " & $sumB & ", expected " & $expectedSumB
  assert abs(sumB - expectedSumB) < 1e-10, "Sum test failed: got " & $sumB & ", expected " & $expectedSumB
  
  # Test sum with different values - use a simple pattern for the test
  fieldA := 0.0  # Reset to zero first
  var localSimpleCubicFieldA = fieldA.localSimpleCubicField()
  for n in 0..<localSimpleCubicFieldA.numSites():
    localSimpleCubicFieldA[n] = float64(n)
  
  let sumSequential = fieldA.sum()
  var expectedSequential = 0.0
  for n in 0..<fieldA.numSites():
    expectedSequential += float64(n)
  expectedSequential *= float64(numRanks())  # Scale by number of ranks
  assert abs(sumSequential - expectedSequential) < 1e-10, "Sequential sum test failed: got " & $sumSequential & ", expected " & $expectedSequential
  
  # Test complex sum
  cfieldA := complex(1.0, 2.0)
  let sumCA = cfieldA.sum()
  let expectedSumCA = complex(1.0, 2.0) * float64(cfieldA.numSites()) * float64(numRanks())
  assert abs(sumCA.re - expectedSumCA.re) < 1e-10 and abs(sumCA.im - expectedSumCA.im) < 1e-10, 
    "Complex sum test failed: got " & $sumCA & ", expected " & $expectedSumCA
  
  print "field sum tests passed"
  
  var rfieldD = cfieldD.re
  var ifieldD = cfieldD.im

  let localRD = rfieldD.localSimpleCubicField()
  let localID = ifieldD.localSimpleCubicField()
  for n in 0..<localCD.numSites():
    assert localRD[n] == 5.0
    assert localID[n] == 0.0
  
  var cfieldE = rfieldD.toComplex()
  let localE = cfieldE.localSimpleCubicField()
  for n in 0..<cfieldE.localSimpleCubicField().numSites():
    assert localE[n] == complex(5.0, 0.0)
  
  print "field conversion tests passed"

  cfieldE := cfieldD * cfieldD.adj 
  var fieldE = cfieldD.norm2
  let clocalFE = cfieldE.localSimpleCubicField()
  let localFE = fieldE.localSimpleCubicField()
  for n in 0..<clocalFE.numSites():
    assert localFE[n] == clocalFE[n].re

  # Test sqrt function
  fieldD := 4.0
  fieldE = fieldD.sqrt
  let localFsqrt = fieldE.localSimpleCubicField()
  for n in 0..<localFsqrt.numSites():
    assert abs(localFsqrt[n] - 2.0) < 1e-10

  # Test exp function  
  fieldD := 0.0
  fieldE = fieldD.exp
  let localFexp = fieldE.localSimpleCubicField()
  for n in 0..<localFexp.numSites():
    assert abs(localFexp[n] - 1.0) < 1e-10

  # Test ln function
  fieldD := 1.0
  fieldE = fieldD.ln
  let localFln = fieldE.localSimpleCubicField()
  for n in 0..<localFln.numSites():
    assert abs(localFln[n] - 0.0) < 1e-10

  # Test sin function
  fieldD := 0.0
  fieldE = fieldD.sin
  let localFsin = fieldE.localSimpleCubicField()
  for n in 0..<localFsin.numSites():
    assert abs(localFsin[n] - 0.0) < 1e-10

  # Test cos function
  fieldD := 0.0
  fieldE = fieldD.cos
  let localFcos = fieldE.localSimpleCubicField()
  for n in 0..<localFcos.numSites():
    assert abs(localFcos[n] - 1.0) < 1e-10

  # Test tan function
  fieldD := 0.0
  fieldE = fieldD.tan
  let localFtan = fieldE.localSimpleCubicField()
  for n in 0..<localFtan.numSites():
    assert abs(localFtan[n] - 0.0) < 1e-10

  # Test abs function  
  fieldD := -3.0
  fieldE = fieldD.abs
  let localFabs = fieldE.localSimpleCubicField()
  for n in 0..<localFabs.numSites():
    assert abs(localFabs[n] - 3.0) < 1e-10

  # Test norm function for complex fields
  cfieldD := complex(3.0, 4.0)
  fieldE = cfieldD.norm
  let localFnorm = fieldE.localSimpleCubicField()
  for n in 0..<localFnorm.numSites():
    assert abs(localFnorm[n] - 5.0) < 1e-10  # sqrt(3^2 + 4^2) = 5

  # Test power operations: field^scalar
  fieldD := 2.0
  fieldE = fieldD ^ 3.0
  let localFpow1 = fieldE.localSimpleCubicField()
  for n in 0..<localFpow1.numSites():
    assert abs(localFpow1[n] - 8.0) < 1e-10  # 2^3 = 8

  # Test power operations: scalar^field  
  fieldD := 3.0
  fieldE = 2.0 ^ fieldD
  let localFpow2 = fieldE.localSimpleCubicField()
  for n in 0..<localFpow2.numSites():
    assert abs(localFpow2[n] - 8.0) < 1e-10  # 2^3 = 8

  # Test power operations: field^field
  var fieldF = newSimpleCubicField(lattice, float64)
  fieldD := 2.0
  fieldF := 3.0
  fieldE = fieldD ^ fieldF
  let localFpow3 = fieldE.localSimpleCubicField()
  for n in 0..<localFpow3.numSites():
    assert abs(localFpow3[n] - 8.0) < 1e-10  # 2^3 = 8

  # Test mathematical functions with more complex values
  fieldD := PI / 2.0  # π/2
  fieldE = fieldD.sin
  let localFsinPi2 = fieldE.localSimpleCubicField()
  for n in 0..<localFsinPi2.numSites():
    assert abs(localFsinPi2[n] - 1.0) < 1e-10  # sin(π/2) = 1

  fieldD := PI / 2.0  # π/2  
  fieldE = fieldD.cos
  let localFcosPi2 = fieldE.localSimpleCubicField()
  for n in 0..<localFcosPi2.numSites():
    assert abs(localFcosPi2[n] - 0.0) < 1e-10  # cos(π/2) = 0

  fieldD := PI / 4.0  # π/4
  fieldE = fieldD.tan
  let localFtanPi4 = fieldE.localSimpleCubicField()
  for n in 0..<localFtanPi4.numSites():
    assert abs(localFtanPi4[n] - 1.0) < 1e-10  # tan(π/4) = 1

  # Test exp and ln are inverse operations
  fieldD := 2.0
  fieldE = fieldD.exp
  fieldE = fieldE.ln
  let localFexpln = fieldE.localSimpleCubicField()
  for n in 0..<localFexpln.numSites():
    assert abs(localFexpln[n] - 2.0) < 1e-10  # ln(exp(2)) = 2

  # Test sqrt and square are inverse operations
  fieldD := 9.0
  fieldE = fieldD.sqrt 
  fieldE = fieldE ^ 2.0
  let localFsqrtSq = fieldE.localSimpleCubicField() 
  for n in 0..<localFsqrtSq.numSites():
    assert abs(localFsqrtSq[n] - 9.0) < 1e-10  # (sqrt(9))^2 = 9

  print "comprehensive mathematical operation tests passed"

  let paddedSimpleCubicField = fieldD.toPaddedSimpleCubicField([1, 1, 1, 1])
  let tightSimpleCubicField = paddedSimpleCubicField.toTightSimpleCubicField()

  #tightSimpleCubicField := fieldD
  #paddedSimpleCubicField := tightSimpleCubicField

  let localPadded = paddedSimpleCubicField.localSimpleCubicField()
  let localTight = tightSimpleCubicField.localSimpleCubicField()
  let localOriginal = fieldD.localSimpleCubicField()
  
  for n in 0..<localOriginal.numSites():
    assert localPadded[n] == localOriginal[n]
    assert localTight[n] == localOriginal[n]
    assert localPadded[n] == localTight[n]

  paddedSimpleCubicField.exchange()

  print "grid conversion tests passed"

  ## --

  print "scalarfield.nim tests passed"