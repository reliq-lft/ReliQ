import reliq
import arrays

import std/[macros]
import std/[tables]
import std/[strutils]
import std/[math, complex]

import utils/[nimutils]
import lattice/[simplecubiclattice]
import kokkos/[kokkosdispatch]

template isComplexType(T: typedesc): bool =
  ## Check if a type is a complex number type
  T is Complex64 or T is Complex32

type Field*[D: static[int], T] = object
  ## Simple cubic field implementation
  ##
  ## Represents a field defined on a simple cubic lattice.
  ## For complex fields (T = Complex[F]), stores real and imaginary parts as separate GlobalArrays.
  ## For real fields (T = SomeFloat), stores directly as a single GlobalArray.
  lattice*: SimpleCubicLattice[D]
  when isComplexType(T):
    fieldRe: GlobalArray[D, float64]  # Real part (Complex64 only for now)
    fieldIm: GlobalArray[D, float64]  # Imaginary part
  else:
    field: GlobalArray[D, T]

#[ field constructor ]#

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
  ## let field = lattice.newField(float)
  ## let cfield = lattice.newField(Complex64)
  ## ```
  let dimensions = lattice.dimensions
  let mpiGrid = lattice.mpiGrid
  let ghostGrid = lattice.ghostGrid
  when isComplexType(T):
    return Field[D, T](
      lattice: lattice,
      fieldRe: newGlobalArray(dimensions, mpiGrid, ghostGrid, float64),
      fieldIm: newGlobalArray(dimensions, mpiGrid, ghostGrid, float64)
    )
  else:
    return Field[D, T](
      lattice: lattice, 
      field: newGlobalArray(dimensions, mpiGrid, ghostGrid, T)
    )

#[ virtual accessors ]#

proc numSites*[D: static[int], T](field: Field[D, T]): int =
  ## Get the number of local sites in the field
  ##
  ## Parameters:
  ## - `field`: Field instance
  ##
  ## Returns:
  ## The total number of local sites in the field
  when isComplexType(T): return field.fieldRe.numSites()
  else: return field.field.numSites()

#[ complex/real conversion ]#

template toComplex*[D: static[int]](field: Field[D, float]): Field[D, Complex64] =
  ## Convert a real field to a complex field (imaginary part = 0)
  ##
  ## Parameters:
  ## - `field`: Real Field instance
  ##
  ## Returns:
  ## A complex Field with real part = field, imaginary part = 0
  var result = newField(field.lattice): Complex64
  let fview = field.localField()
  var rview = result.localField()
  for i in every 0..<field.numSites(): rview[i] = complex(fview[i], 0.0)
  result

template realPart*[D: static[int]](field: Field[D, Complex64]): Field[D, float] =
  ## Extract the real part of a complex field
  ##
  ## Parameters:
  ## - `field`: Complex Field instance
  ##
  ## Returns:
  ## A real Field containing the real parts
  var result = newField(field.lattice): float
  let fview = field.localField()
  var rview = result.localField()
  for i in every 0..<field.numSites(): rview[i] = fview[i].re
  result

template imagPart*[D: static[int]](field: Field[D, Complex64]): Field[D, float] =
  ## Extract the imaginary part of a complex field
  ##
  ## Parameters:
  ## - `field`: Complex Field instance
  ##
  ## Returns:
  ## A real Field containing the imaginary parts
  var result = newField(field.lattice): float
  let fview = field.localField()
  var rview = result.localField()
  for i in every 0..<field.numSites(): rview[i] = fview[i].im
  result

template conjugate*[D: static[int]](field: Field[D, Complex64]): Field[D, Complex64] =
  ## Compute the complex conjugate of a field
  ##
  ## Parameters:
  ## - `field`: Complex Field instance
  ##
  ## Returns:
  ## A complex Field with conjugated values
  var result = newField(field.lattice): Complex64
  let fview = field.localField()
  var rview = result.localField()
  for i in every 0..<field.numSites():
    let val = fview[i]
    rview[i] = complex(val.re, -val.im)
  result

template absSquared*[D: static[int]](field: Field[D, Complex64]): Field[D, float] =
  ## Compute |z|² = re² + im² for each element
  ##
  ## Parameters:
  ## - `field`: Complex Field instance
  ##
  ## Returns:
  ## A real Field containing |z|² values
  var result = newField(field.lattice): float
  let fview = field.localField()
  var rview = result.localField()
  for i in every 0..<field.numSites():
    let val = fview[i]
    rview[i] = val.re * val.re + val.im * val.im
  result

template abs*[D: static[int]](field: Field[D, Complex64]): Field[D, float] =
  ## Compute |z| = sqrt(re² + im²) for each element
  ##
  ## Parameters:
  ## - `field`: Complex Field instance
  ##
  ## Returns:
  ## A real Field containing |z| values
  var result = field.absSquared()
  result = result.sqrt()
  result

#[ Chapel-like arithmetic promotion ]#

proc `[]`*[D: static[int], F](view: ComplexLocalView[D, F], idx: int): Complex[F] =
  ## Access complex value at index
  complex(view.re[idx], view.im[idx])

proc `[]=`*[D: static[int], F](view: var ComplexLocalView[D, F], idx: int, val: Complex[F]) =
  ## Set complex value at index
  view.re[idx] = val.re
  view.im[idx] = val.im

proc `[]=`*[D: static[int], F](view: var ComplexLocalView[D, F], idx: int, val: SomeNumber) =
  ## Set complex value at index from a real number (imaginary part = 0)
  view.re[idx] = F(val)
  view.im[idx] = F(0)

proc localField*[D: static[int], T](field: Field[D, T]): auto =
  ## Get the local view of the field
  ##
  ## Parameters:
  ## - `field`: Field instance
  ##
  ## Returns:
  ## A LocalView (real) or ComplexLocalView (complex) representing the local 
  ## portion of the field
  when isComplexType(T):
    ComplexLocalView[D, float64](
      re: field.fieldRe.localView(),
      im: field.fieldIm.localView()
    )
  else: field.field.localView()

proc localField*(x: SomeInteger | SomeFloat): auto = x
proc localField*[T](x: Complex[T]): auto = x

proc `[]`*(x: SomeInteger | SomeFloat; n: SomeInteger): auto = x
proc `[]`*[T](x: Complex[T]; n: SomeInteger): auto = x

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

#[ ordinary mathematical functions ]#

template sqrt*[D: static[int], T](field: Field[D, T]): Field[D, T] =
  ## Element-wise square root of field
  ##
  ## Parameters:
  ## - `field`: Field instance
  ##
  ## Returns:
  ## A new Field with the square root applied element-wise
  var r = newField(field.lattice): T
  var rv = r.localField()
  let fv = field.localField()
  for n in every 0..<field.numSites(): rv[n] = sqrt(fv[n])
  r

template exp*[D: static[int], T](field: Field[D, T]): Field[D, T] =
  ## Element-wise exponential of field
  ##
  ## Parameters:
  ## - `field`: Field instance
  ##
  ## Returns:
  ## A new Field with the exponential applied element-wise
  var r = newField(field.lattice): T
  var rv = r.localField()
  let fv = field.localField()
  for n in every 0..<field.numSites(): rv[n] = exp(fv[n])
  r

template ln*[D: static[int], T](field: Field[D, T]): Field[D, T] =
  ## Element-wise natural logarithm of field
  ##
  ## Parameters:
  ## - `field`: Field instance
  ##
  ## Returns:
  ## A new Field with the natural logarithm applied element-wise
  var r = newField(field.lattice): T
  var rv = r.localField()
  let fv = field.localField()
  for n in every 0..<field.numSites(): rv[n] = ln(fv[n])
  r

template sin*[D: static[int], T](field: Field[D, T]): Field[D, T] =
  ## Element-wise sine of field
  ##
  ## Parameters:
  ## - `field`: Field instance
  ##
  ## Returns:
  ## A new Field with the sine applied element-wise
  var r = newField(field.lattice): T
  var rv = r.localField()
  let fv = field.localField()
  for n in every 0..<field.numSites(): rv[n] = sin(fv[n])
  r

template cos*[D: static[int], T](field: Field[D, T]): Field[D, T] =
  ## Element-wise cosine of field
  ##
  ## Parameters:
  ## - `field`: Field instance
  ##
  ## Returns:
  ## A new Field with the cosine applied element-wise
  var r = newField(field.lattice): T
  var rv = r.localField()
  let fv = field.localField()
  for n in every 0..<field.numSites(): rv[n] = cos(fv[n])
  r

template tan*[D: static[int], T](field: Field[D, T]): Field[D, T] =
  ## Element-wise tangent of field
  ##
  ## Parameters:
  ## - `field`: Field instance
  ##
  ## Returns:
  ## A new Field with the tangent applied element-wise
  var r = newField(field.lattice): T
  var rv = r.localField()
  let fv = field.localField()
  for n in every 0..<field.numSites(): rv[n] = tan(fv[n])
  r

template abs*[D: static[int], T](field: Field[D, T]): Field[D, T] =
  ## Element-wise absolute value of field
  ##
  ## Parameters:
  ## - `field`: Field instance
  ##
  ## Returns:
  ## A new Field with the absolute value applied element-wise
  var r = newField(field.lattice): T
  var rv = r.localField()
  let fv = field.localField()
  for n in every 0..<field.numSites(): rv[n] = abs(fv[n])
  r

template `^`*[D: static[int], T](base: Field[D, T], exponent: T): Field[D, T] =
  ## Element-wise power of field
  ##
  ## Parameters:
  ## - `base`: Field instance (the base)
  ## - `exponent`: Exponent value
  ##
  ## Returns:
  ## A new Field with the power applied element-wise
  var r = newField(base.lattice): T
  var rv = r.localField()
  let bv = base.localField()
  for n in every 0..<base.numSites(): rv[n] = pow(bv[n], exponent)
  r

template `^`*[D: static[int], T](base: T, exponent: Field[D, T]): Field[D, T] =
  ## Element-wise power of field
  ##
  ## Parameters:
  ## - `base`: Base value
  ## - `exponent`: Field instance (the exponent)
  ##
  ## Returns:
  ## A new Field with the power applied element-wise
  var r = newField(exponent.lattice): T
  var rv = r.localField()
  let ev = exponent.localField()
  for n in every 0..<exponent.numSites(): rv[n] = pow(base, ev[n])
  r

template `^`*[D: static[int], T](base: Field[D, T], exponent: Field[D, T]): Field[D, T] =
  ## Element-wise power of field
  ##
  ## Parameters:
  ## - `base`: Field instance (the base)
  ## - `exponent`: Field instance (the exponent)
  ##
  ## Returns:
  ## A new Field with the power applied element-wise
  var r = newField(base.lattice): T
  var rv = r.localField()
  let bv = base.localField()
  let ev = exponent.localField()
  for n in every 0..<base.numSites(): rv[n] = pow(bv[n], ev[n])
  r

#[ unit tests ]#

test:
  let lattice = newSimpleCubicLattice([8, 8, 8, 8*numRanks()])
  var field1 = newField(lattice): float
  var field2 = newField(lattice): float
  var field3 = newField(lattice): float
  var field4 = newField(lattice): float
  
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
  
  echo "Process ", myRank(), "/", numRanks(), ": All Field tests passed!"

  # test promotion on arrays of fields
  var fieldArray: array[4, Field[4, float]]
  for i in 0..<4:
    fieldArray[i] = newField(lattice): float
    fieldArray[i] := float(i + 1)
  fieldArray[0] := 1.0
  
  # Test mathematical function overloads
  var mathField = newField(lattice): float
  mathField := 4.0
  
  # Test sqrt
  var sqrtField = mathField.sqrt()
  let sqrtView = sqrtField.localField()
  for i in 0..<sqrtField.numSites():
    assert(abs(sqrtView[i] - 2.0) < 1e-10, "sqrt(4.0) should be 2.0")
  
  # Test exp and ln (inverse operations)
  mathField := 2.0
  var expField = mathField.exp()
  var lnField = expField.ln()
  let lnView = lnField.localField()
  for i in 0..<lnField.numSites():
    assert(abs(lnView[i] - 2.0) < 1e-10, "ln(exp(2.0)) should be 2.0")
  
  # Test sin and cos
  mathField := 0.0
  var sinField = mathField.sin()
  var cosField = mathField.cos()
  let sinView = sinField.localField()
  let cosView = cosField.localField()
  for i in 0..<sinField.numSites():
    assert(abs(sinView[i] - 0.0) < 1e-10, "sin(0.0) should be 0.0")
    assert(abs(cosView[i] - 1.0) < 1e-10, "cos(0.0) should be 1.0")
  
  # Test tan
  mathField := PI/4.0
  var tanField = mathField.tan()
  let tanView = tanField.localField()
  for i in 0..<tanField.numSites():
    assert(abs(tanView[i] - 1.0) < 1e-10, "tan(π/4) should be 1.0")
  
  # Test abs
  mathField := -5.0
  var absField = mathField.abs()
  let absView = absField.localField()
  for i in 0..<absField.numSites():
    assert(abs(absView[i] - 5.0) < 1e-10, "abs(-5.0) should be 5.0")
  
  # Test power operator (field ^ scalar)
  mathField := 2.0
  var powField1 = mathField ^ 3.0
  let powView1 = powField1.localField()
  for i in 0..<powField1.numSites():
    assert(abs(powView1[i] - 8.0) < 1e-10, "2.0^3.0 should be 8.0")
  
  # Test power operator (scalar ^ field)
  mathField := 3.0
  var powField2 = 2.0 ^ mathField
  let powView2 = powField2.localField()
  for i in 0..<powField2.numSites():
    assert(abs(powView2[i] - 8.0) < 1e-10, "2.0^3.0 should be 8.0")
  
  # Test power operator (field ^ field)
  var baseField = newField(lattice): float
  var expField2 = newField(lattice): float
  baseField := 2.0
  expField2 := 4.0
  var powField3 = baseField ^ expField2
  let powView3 = powField3.localField()
  for i in 0..<powField3.numSites():
    assert(abs(powView3[i] - 16.0) < 1e-10, "2.0^4.0 should be 16.0")
  
  echo "Process ", myRank(), "/", numRanks(), ": All mathematical function tests passed!"
  
  # Test complex fields
  var cfield1 = newField(lattice): Complex64
  var cfield2 = newField(lattice): Complex64
  
  # Test complex assignment
  cfield1 := complex(2.0, 3.0)
  cfield2 := complex(1.0, -1.0)
  
  let cview1 = cfield1.localField()
  let cview2 = cfield2.localField()
  for i in 0..<cfield1.numSites():
    let val1 = cview1[i]
    let val2 = cview2[i]
    assert(abs(val1.re - 2.0) < 1e-10, "Complex real part should be 2.0")
    assert(abs(val1.im - 3.0) < 1e-10, "Complex imag part should be 3.0")
    assert(abs(val2.re - 1.0) < 1e-10, "Complex real part should be 1.0")
    assert(abs(val2.im + 1.0) < 1e-10, "Complex imag part should be -1.0")
  
  # Test complex arithmetic with promotion
  var cfield3 = newField(lattice): Complex64
  var cfield4 = newField(lattice): Complex64
  var cfield5 = newField(lattice): Complex64
  
  # Test addition: (2+3i) + (1-i) = (3+2i)
  cfield3 := cfield1 + cfield2
  let cview3 = cfield3.localField()
  for i in 0..<cfield3.numSites():
    let val = cview3[i]
    assert(abs(val.re - 3.0) < 1e-10, "Complex sum real part should be 3.0")
    assert(abs(val.im - 2.0) < 1e-10, "Complex sum imag part should be 2.0")
  
  # Test subtraction: (2+3i) - (1-i) = (1+4i)
  cfield3 := cfield1 - cfield2
  for i in 0..<cfield3.numSites():
    let val = cview3[i]
    assert(abs(val.re - 1.0) < 1e-10, "Complex difference real part should be 1.0")
    assert(abs(val.im - 4.0) < 1e-10, "Complex difference imag part should be 4.0")
  
  # Test multiplication: (2+3i) * (1-i) = 2-2i+3i-3i² = 2+i+3 = (5+i)
  cfield3 := cfield1 * cfield2
  for i in 0..<cfield3.numSites():
    let val = cview3[i]
    assert(abs(val.re - 5.0) < 1e-10, "Complex product real part should be 5.0")
    assert(abs(val.im - 1.0) < 1e-10, "Complex product imag part should be 1.0")
  
  # Test division: (2+3i) / (1-i) = (2+3i)(1+i) / ((1-i)(1+i)) = (2+2i+3i+3i²) / 2 = (-1+5i)/2 = (-0.5+2.5i)
  cfield3 := cfield1 / cfield2
  for i in 0..<cfield3.numSites():
    let val = cview3[i]
    assert(abs(val.re - (-0.5)) < 1e-10, "Complex quotient real part should be -0.5")
    assert(abs(val.im - 2.5) < 1e-10, "Complex quotient imag part should be 2.5")
  
  # Test complex expression with multiple operations: (2+3i) + 2*(1-i) - (2+3i)*(1-i)
  # = (2+3i) + (2-2i) - (5+i) = (4+i) - (5+i) = (-1+0i)
  cfield4 := cfield1 + 2.0*cfield2
  cfield5 := cfield1 * cfield2
  cfield3 := cfield4 - cfield5
  for i in 0..<cfield3.numSites():
    let val = cview3[i]
    assert(abs(val.re - (-1.0)) < 1e-10, "Complex expression real part should be -1.0")
    assert(abs(val.im - 0.0) < 1e-10, "Complex expression imag part should be 0.0")
  
  # Test compound assignment operators with complex fields
  cfield3 := complex(3.0, 4.0)
  cfield3 += complex(1.0, 1.0)  # (3+4i) + (1+i) = (4+5i)
  for i in 0..<cfield3.numSites():
    let val = cview3[i]
    assert(abs(val.re - 4.0) < 1e-10, "Complex += real part should be 4.0")
    assert(abs(val.im - 5.0) < 1e-10, "Complex += imag part should be 5.0")
  
  cfield3 *= complex(2.0, 0.0)  # (4+5i) * 2 = (8+10i)
  for i in 0..<cfield3.numSites():
    let val = cview3[i]
    assert(abs(val.re - 8.0) < 1e-10, "Complex *= real part should be 8.0")
    assert(abs(val.im - 10.0) < 1e-10, "Complex *= imag part should be 10.0")
  
  cfield3 /= complex(2.0, 0.0)  # (8+10i) / 2 = (4+5i)
  for i in 0..<cfield3.numSites():
    let val = cview3[i]
    assert(abs(val.re - 4.0) < 1e-10, "Complex /= real part should be 4.0")
    assert(abs(val.im - 5.0) < 1e-10, "Complex /= imag part should be 5.0")
  
  cfield3 -= complex(1.0, 2.0)  # (4+5i) - (1+2i) = (3+3i)
  for i in 0..<cfield3.numSites():
    let val = cview3[i]
    assert(abs(val.re - 3.0) < 1e-10, "Complex -= real part should be 3.0")
    assert(abs(val.im - 3.0) < 1e-10, "Complex -= imag part should be 3.0")
  
  # Test scalar multiplication with complex fields
  cfield1 := complex(1.0, 2.0)
  cfield3 := 3.0 * cfield1  # 3 * (1+2i) = (3+6i)
  for i in 0..<cfield3.numSites():
    let val = cview3[i]
    assert(abs(val.re - 3.0) < 1e-10, "Scalar * complex real part should be 3.0")
    assert(abs(val.im - 6.0) < 1e-10, "Scalar * complex imag part should be 6.0")
  
  # Test mixed real and complex field arithmetic
  var rfield1 = newField(lattice): float
  var rfield2 = newField(lattice): float
  rfield1 := 2.0
  rfield2 := 5.0
  
  # Complex field + real scalar: (3+4i) + 2 = (5+4i)
  cfield1 := complex(3.0, 4.0)
  cfield3 := cfield1 + 2.0
  for i in 0..<cfield3.numSites():
    let val = cview3[i]
    assert(abs(val.re - 5.0) < 1e-10, "Complex + scalar real part should be 5.0")
    assert(abs(val.im - 4.0) < 1e-10, "Complex + scalar imag part should be 4.0")
  
  # Real scalar + complex field: 2 + (3+4i) = (5+4i)
  cfield3 := 2.0 + cfield1
  for i in 0..<cfield3.numSites():
    let val = cview3[i]
    assert(abs(val.re - 5.0) < 1e-10, "Scalar + complex real part should be 5.0")
    assert(abs(val.im - 4.0) < 1e-10, "Scalar + complex imag part should be 4.0")
  
  # Complex field * real scalar: (3+4i) * 2 = (6+8i)
  cfield3 := cfield1 * 2.0
  for i in 0..<cfield3.numSites():
    let val = cview3[i]
    assert(abs(val.re - 6.0) < 1e-10, "Complex * scalar real part should be 6.0")
    assert(abs(val.im - 8.0) < 1e-10, "Complex * scalar imag part should be 8.0")
  
  # Real scalar * complex field: 2 * (3+4i) = (6+8i)
  cfield3 := 2.0 * cfield1
  for i in 0..<cfield3.numSites():
    let val = cview3[i]
    assert(abs(val.re - 6.0) < 1e-10, "Scalar * complex real part should be 6.0")
    assert(abs(val.im - 8.0) < 1e-10, "Scalar * complex imag part should be 8.0")
  
  # Complex field / real scalar: (6+8i) / 2 = (3+4i)
  cfield1 := complex(6.0, 8.0)
  cfield3 := cfield1 / 2.0
  for i in 0..<cfield3.numSites():
    let val = cview3[i]
    assert(abs(val.re - 3.0) < 1e-10, "Complex / scalar real part should be 3.0")
    assert(abs(val.im - 4.0) < 1e-10, "Complex / scalar imag part should be 4.0")
  
  # Complex field - real scalar: (5+4i) - 2 = (3+4i)
  cfield1 := complex(5.0, 4.0)
  cfield3 := cfield1 - 2.0
  for i in 0..<cfield3.numSites():
    let val = cview3[i]
    assert(abs(val.re - 3.0) < 1e-10, "Complex - scalar real part should be 3.0")
    assert(abs(val.im - 4.0) < 1e-10, "Complex - scalar imag part should be 4.0")
  
  # Real scalar - complex field: 10 - (3+4i) = (7-4i)
  cfield3 := 10.0 - cfield1
  for i in 0..<cfield3.numSites():
    let val = cview3[i]
    assert(abs(val.re - 5.0) < 1e-10, "Scalar - complex real part should be 5.0")
    assert(abs(val.im - (-4.0)) < 1e-10, "Scalar - complex imag part should be -4.0")
  
  # Complex expression with real and complex: 2 * (3+4i) + 5 = (6+8i) + 5 = (11+8i)
  cfield1 := complex(3.0, 4.0)
  cfield3 := 2.0 * cfield1 + 5.0
  for i in 0..<cfield3.numSites():
    let val = cview3[i]
    assert(abs(val.re - 11.0) < 1e-10, "Mixed expression real part should be 11.0")
    assert(abs(val.im - 8.0) < 1e-10, "Mixed expression imag part should be 8.0")
  
  # Test conjugate-like operations (using subtraction of imaginary part)
  cfield1 := complex(3.0, 4.0)
  cfield2 := complex(3.0, -4.0)  # Conjugate
  cfield3 := cfield1 * cfield2  # (3+4i)(3-4i) = 9 - 16i² = 9 + 16 = 25
  for i in 0..<cfield3.numSites():
    let val = cview3[i]
    assert(abs(val.re - 25.0) < 1e-10, "Complex conjugate product real part should be 25.0")
    assert(abs(val.im - 0.0) < 1e-10, "Complex conjugate product imag part should be 0.0")
  
  echo "Process ", myRank(), "/", numRanks(), ": All complex field tests passed!"
  

  ########################################################## <--- hang location

  # Test complex/real conversion functions
  
  # Test toComplex: convert real field to complex
  rfield1 := 5.0
  var cfield_from_real = rfield1.toComplex()
  let cfr_view = cfield_from_real.localField()
  for i in 0..<cfield_from_real.numSites():
    let val = cfr_view[i]
    assert(abs(val.re - 5.0) < 1e-10, "toComplex real part should be 5.0")
    assert(abs(val.im - 0.0) < 1e-10, "toComplex imag part should be 0.0")
  
  # Test realPart: extract real part from complex field
  cfield1 := complex(3.0, 4.0)
  var real_part = cfield1.realPart()
  let rp_view = real_part.localField()
  for i in 0..<real_part.numSites():
    assert(abs(rp_view[i] - 3.0) < 1e-10, "realPart should be 3.0")
  
  # Test imagPart: extract imaginary part from complex field
  var imag_part = cfield1.imagPart()
  let ip_view = imag_part.localField()
  for i in 0..<imag_part.numSites():
    assert(abs(ip_view[i] - 4.0) < 1e-10, "imagPart should be 4.0")
  
  # Test conjugate: compute complex conjugate
  var cfield_conj = cfield1.conjugate()
  let cc_view = cfield_conj.localField()
  for i in 0..<cfield_conj.numSites():
    let val = cc_view[i]
    assert(abs(val.re - 3.0) < 1e-10, "conjugate real part should be 3.0")
    assert(abs(val.im - (-4.0)) < 1e-10, "conjugate imag part should be -4.0")
  
  # Test absSquared: |z|² for z = 3+4i should be 9+16 = 25
  var abs_sq = cfield1.absSquared()
  let absq_view = abs_sq.localField()
  for i in 0..<abs_sq.numSites():
    assert(abs(absq_view[i] - 25.0) < 1e-10, "absSquared should be 25.0")
  
  # Test abs: |z| for z = 3+4i should be 5
  var cabs_field = cfield1.abs()
  let cabsf_view = cabs_field.localField()
  for i in 0..<cabs_field.numSites():
    assert(abs(cabsf_view[i] - 5.0) < 1e-10, "abs should be 5.0")
  
  # Test that conjugate * original gives real result with imaginary part = 0
  cfield1 := complex(2.0, 3.0)
  cfield2 := cfield1.conjugate()
  cfield3 := cfield1 * cfield2  # Should be |z|² = 4 + 9 = 13
  for i in 0..<cfield3.numSites():
    let val = cview3[i]
    assert(abs(val.re - 13.0) < 1e-10, "z * conj(z) real part should be 13.0")
    assert(abs(val.im - 0.0) < 1e-10, "z * conj(z) imag part should be 0.0")
  
  # Test arithmetic with converted fields
  rfield1 := 3.0
  rfield2 := 4.0
  var cfield_r1 = rfield1.toComplex()
  var cfield_r2 = rfield2.toComplex()
  cfield3 := cfield_r1 + cfield_r2  # (3+0i) + (4+0i) = (7+0i)
  for i in 0..<cfield3.numSites():
    let val = cview3[i]
    assert(abs(val.re - 7.0) < 1e-10, "Converted field sum real part should be 7.0")
    assert(abs(val.im - 0.0) < 1e-10, "Converted field sum imag part should be 0.0")
  echo "Process ", myRank(), "/", numRanks(), ": All complex/real conversion tests passed!"
