#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/opencl/clcmp.nim
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
import std/[strutils]
import std/[sequtils]
import std/[options]

import clir

## OpenCL C code generation
##
## Converts ClAst (OpenCL intermediate representation) to OpenCL C source code.

#[ ClType -> OpenCL C string ]#

proc toClC*(t: ClTypeKind): string =
  ## Converts a ClTypeKind to its OpenCL C string representation
  case t
  of cltBool: "bool"
  of cltUint8: "uchar"
  of cltUint16: "ushort"
  of cltUint32: "uint"
  of cltUint64: "ulong"
  of cltInt8: "char"
  of cltInt16: "short"
  of cltInt32: "int"
  of cltInt64: "long"
  of cltFloat32: "float"
  of cltFloat64: "double"
  of cltVoid: "void"
  of cltSize_t: "size_t"
  of cltPtr: "*"
  of cltVoidPtr: "void*"
  of cltObject: "struct"
  of cltString: "const char*"
  of cltArray: raiseAssert "Array type requires special handling"

proc toClC*(t: ClType; ident: string = ""; allowEmptyIdent = false): string
  ## Forward declaration for recursive type stringification

proc getInnerArrayType(t: ClType): string =
  ## Recursively extracts the innermost non-array type from a nested array type
  case t.kind
  of cltArray: getInnerArrayType(t.aTyp)
  else: toClC(t)

proc getInnerArrayLengths(t: ClType): string =
  ## Recursively builds array dimension string for nested arrays
  case t.kind
  of cltArray:
    let inner = getInnerArrayLengths(t.aTyp)
    result = "[" & $t.aLen & "]"
    if inner.len > 0: result.add inner
  else: result = ""

proc toClC*(t: ClType; ident: string = ""; allowEmptyIdent = false): string =
  ## Converts a ClType to its OpenCL C string representation
  ## 
  ## Parameters:
  ##   t: The type to convert
  ##   ident: Optional identifier name (required for array types)
  ##   allowEmptyIdent: If true, allows empty identifier for array types
  var skipIdent = false
  case t.kind
  of cltPtr:
    if t.to.kind == cltArray:
      let ptrStar = toClC(t.kind)
      result = toClC(t.to, "(" & ptrStar & ident & ")")
      skipIdent = true
    else:
      let typ = toClC(t.to, allowEmptyIdent = allowEmptyIdent)
      let ptrStar = toClC(t.kind)
      result = typ & ptrStar
  of cltArray:
    if ident.len == 0 and not allowEmptyIdent:
      raiseAssert "Array type requires identifier: " & $t[]
    case t.aTyp.kind
    of cltArray:
      let typ = getInnerArrayType(t)
      let lengths = getInnerArrayLengths(t)
      result = typ & " " & ident & lengths
    else:
      if t.aLen == 0:
        result = toClC(t.aTyp, allowEmptyIdent = allowEmptyIdent) & " " & ident & "[]"
      else:
        result = toClC(t.aTyp, allowEmptyIdent = allowEmptyIdent) & " " & ident & "[" & $t.aLen & "]"
    skipIdent = true
  of cltObject: result = t.name
  of cltVoidPtr: result = "void*"
  else: result = toClC(t.kind)

  if ident.len > 0 and not skipIdent:
    result.add " " & ident

#[ ClAst -> OpenCL C code generation ]#

proc toClC*(ctx: var ClContext; ast: ClAst; indent = 0): string
  ## Forward declaration for recursive code generation

template withoutSemicolon(ctx: var ClContext; body: untyped): untyped =
  ## Temporarily disables semicolon insertion for the duration of body
  ctx.skipSemicolon = true
  body
  ctx.skipSemicolon = false

proc toClC*(ctx: var ClContext; ast: ClAst; indent = 0): string =
  ## Converts a ClAst node to OpenCL C source code
  ##
  ## Parameters:
  ##   ctx: The context for code generation state
  ##   ast: The AST node to convert
  ##   indent: Current indentation level
  let indentStr = "  ".repeat(indent)

  case ast.kind
  of clnVoid: return

  of clnProc:
    var attrs: seq[string]
    for att in ast.pAttributes: attrs.add $att

    # parameters with memory qualifiers
    var params: seq[string]
    for (name, typ, pAttrs) in ast.pParams:
      var paramStr = ""
      for attr in pAttrs: paramStr.add $attr & " "
      paramStr.add toClC(typ, name)
      params.add paramStr

    let fnArgs = params.join(", ")
    let retType = toClC(ast.pRetType, allowEmptyIdent = true)

    result = indentStr & attrs.join(" ")
    if attrs.len > 0: result.add " "
    result.add retType & " " & ast.pName & "(" & fnArgs & ") {\n"

    # check if function uses implicit result variable (non-void return type)
    let hasResultVar = ast.pRetType != nil and ast.pRetType.kind != cltVoid
    if hasResultVar:
      # declare result variable at start of function
      result &= "  ".repeat(indent + 1) & retType & " result;\n"

    result &= ctx.toClC(ast.pBody, indent + 1)

    # add implicit return if function uses result
    if hasResultVar:
      result &= "\n" & "  ".repeat(indent + 1) & "return result;"

    result &= "\n" & indentStr & "}"

  of clnBlock:
    result = ""
    if ast.blockLabel.len > 0:
      result.add "\n" & indentStr & "{ // " & ast.blockLabel & "\n"
    for idx, el in ast.statements:
      result.add ctx.toClC(el, indent)
      # don't add semicolon after blocks (if/for/while/proc) or when skipSemicolon is set
      if el.kind notin {clnBlock, clnIf, clnFor, clnWhile, clnProc} and not ctx.skipSemicolon:
        result.add ";"
      if idx < ast.statements.high:
        result.add "\n"
    if ast.blockLabel.len > 0:
      result.add "\n" & indentStr & "} // " & ast.blockLabel & "\n"

  of clnVar:
    result = indentStr
    for attr in ast.vAttributes: result.add $attr & " "
    result.add toClC(ast.vType, ast.vName)
    if ast.vInit != nil and not ast.vRequiresMemcpy:
      result &= " = " & ctx.toClC(ast.vInit)

  of clnAssign:
    result = indentStr & ctx.toClC(ast.aLeft) & " = " & ctx.toClC(ast.aRight)

  of clnIf:
    ctx.withoutSemicolon:
      result = indentStr & "if (" & ctx.toClC(ast.ifCond) & ") {\n"
    result &= ctx.toClC(ast.ifThen, indent + 1) & "\n"
    result &= indentStr & "}"
    if ast.ifElse.isSome:
      result &= " else {\n"
      result &= ctx.toClC(ast.ifElse.get, indent + 1) & "\n"
      result &= indentStr & "}"

  of clnFor:
    result = indentStr & "for (int " & ast.fVar & " = " &
             ctx.toClC(ast.fStart) & "; " &
             ast.fVar & " < " & ctx.toClC(ast.fEnd) & "; " &
             ast.fVar & "++) {\n"
    result &= ctx.toClC(ast.fBody, indent + 1) & "\n"
    result &= indentStr & "}"

  of clnWhile:
    ctx.withoutSemicolon:
      result = indentStr & "while (" & ctx.toClC(ast.wCond) & ") {\n"
    result &= ctx.toClC(ast.wBody, indent + 1) & "\n"
    result &= indentStr & "}"

  of clnDot:
    result = ctx.toClC(ast.dParent) & "." & ctx.toClC(ast.dField)

  of clnIndex:
    result = ctx.toClC(ast.iArr) & "[" & ctx.toClC(ast.iIndex) & "]"

  of clnCall:
    result = indentStr & ast.cName & "(" &
             ast.cArgs.mapIt(ctx.toClC(it)).join(", ") & ")"

  of clnBinOp:
    result = "(" & ctx.toClC(ast.bLeft) & " " & ast.bOp & " " & ctx.toClC(ast.bRight) & ")"

  of clnIdent: result = ast.iName

  of clnLit:
    if ast.lType.kind == cltString:
      result = "\"" & ast.lValue & "\""
    elif ast.lValue == "DEFAULT":
      result = "{}"
    else:
      result = ast.lValue

  of clnArrayLit:
    result = "{"
    for idx, el in ast.aValues:
      result.add "(" & toClC(ast.aLitType) & ")" & el
      if idx < ast.aValues.high: result.add ", "
    result.add "}"

  of clnReturn:
    result = indentStr & "return " & ctx.toClC(ast.rValue)

  of clnPrefix:
    result = ast.pOp & ctx.toClC(ast.pVal)

  of clnTypeDef:
    result = "typedef struct {\n"
    for el in ast.tFields:
      result.add "  " & toClC(el.typ, el.name) & ";\n"
    result.add "} " & ast.tName

  of clnObjConstr:
    result = "{"
    for idx, el in ast.ocFields:
      result.add ctx.toClC(el.value)
      if idx < ast.ocFields.len - 1: result.add ", "
    result.add "}"

  of clnComment:
    result = indentStr & "/* " & ast.comment & " */"

  of clnCast:
    result = "(" & toClC(ast.cTo, allowEmptyIdent = true) & ")" & ctx.toClC(ast.cExpr)

  of clnAddr:
    result = "(&" & ctx.toClC(ast.aOf) & ")"

  of clnDeref:
    result = "(*" & ctx.toClC(ast.dOf) & ")"

  of clnConstexpr:
    if ast.cType.kind == cltArray:
      result = indentStr & "__constant " & toClC(ast.cType, ctx.toClC(ast.cIdent)) & " = " & ctx.toClC(ast.cValue)
    else:
      result = indentStr & "__constant " & toClC(ast.cType, allowEmptyIdent = true) & " " & ctx.toClC(ast.cIdent) & " = " & ctx.toClC(ast.cValue)

  of clnVecConstr:
    result = toClC(ast.vcType, allowEmptyIdent = true) & "(" &
             ast.vcArgs.mapIt(ctx.toClC(it)).join(", ") & ")"

#[ Double precision detection ]#

proc usesDoublePrecision*(ast: ClAst): bool =
  ## Recursively checks if the AST uses any double precision types
  case ast.kind
  of clnVoid: return false

  of clnProc:
    # check return type
    if ast.pRetType != nil and ast.pRetType.kind == cltFloat64:
      return true
    # check parameters
    for param in ast.pParams:
      let ptype = param[1]
      if ptype.kind == cltFloat64: return true
      if ptype.kind == cltPtr and ptype.to != nil and ptype.to.kind == cltFloat64:
        return true
    # check body
    if ast.pBody != nil: return usesDoublePrecision(ast.pBody)

  of clnBlock:
    for stmt in ast.statements:
      if usesDoublePrecision(stmt): return true

  of clnVar:
    if ast.vType.kind == cltFloat64: return true
    if ast.vType.kind == cltPtr and ast.vType.to != nil and ast.vType.to.kind == cltFloat64:
      return true
    if ast.vInit != nil: return usesDoublePrecision(ast.vInit)

  of clnAssign:
    return usesDoublePrecision(ast.aLeft) or usesDoublePrecision(ast.aRight)

  of clnIf:
    if usesDoublePrecision(ast.ifCond): return true
    if usesDoublePrecision(ast.ifThen): return true
    if ast.ifElse.isSome and usesDoublePrecision(ast.ifElse.get): return true

  of clnFor:
    return usesDoublePrecision(ast.fStart) or
           usesDoublePrecision(ast.fEnd) or
           usesDoublePrecision(ast.fBody)

  of clnWhile:
    return usesDoublePrecision(ast.wCond) or usesDoublePrecision(ast.wBody)

  of clnCall:
    for arg in ast.cArgs:
      if usesDoublePrecision(arg): return true

  of clnBinOp:
    return usesDoublePrecision(ast.bLeft) or usesDoublePrecision(ast.bRight)

  of clnPrefix:
    return usesDoublePrecision(ast.pVal)

  of clnDot:
    return usesDoublePrecision(ast.dParent)

  of clnIndex:
    return usesDoublePrecision(ast.iArr) or usesDoublePrecision(ast.iIndex)

  of clnReturn:
    if ast.rValue != nil: return usesDoublePrecision(ast.rValue)

  of clnCast:
    if ast.cTo.kind == cltFloat64: return true
    return usesDoublePrecision(ast.cExpr)

  of clnAddr:
    return usesDoublePrecision(ast.aOf)

  of clnDeref:
    return usesDoublePrecision(ast.dOf)

  of clnConstexpr:
    if ast.cType.kind == cltFloat64: return true
    return usesDoublePrecision(ast.cValue)

  of clnVecConstr:
    for arg in ast.vcArgs:
      if usesDoublePrecision(arg): return true

  of clnLit:
    if ast.lType.kind == cltFloat64: return true

  of clnIdent, clnArrayLit, clnComment, clnTypeDef, clnObjConstr:
    discard

  return false

const DoublePrecisionPragma* = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n"

#[ Convenience procedures ]#

proc genOpenCL*(ctx: var ClContext; ast: ClAst): string =
  ## Generates OpenCL C source code from a ClAst using provided context
  ## Automatically adds cl_khr_fp64 pragma if double precision types are detected
  let needsFp64 = usesDoublePrecision(ast)
  if needsFp64: result = DoublePrecisionPragma
  result &= ctx.toClC(ast)

proc genOpenCL*(ast: ClAst): string =
  ## Generates OpenCL C source code from a ClAst
  ## Automatically adds cl_khr_fp64 pragma if double precision types are detected
  var ctx = ClContext()
  result = genOpenCL(ctx, ast)

proc genOpenCLNoPragma*(ctx: var ClContext; ast: ClAst): string =
  ## Generates OpenCL C source code from a ClAst without the fp64 pragma
  result = ctx.toClC(ast)

proc genOpenCLNoPragma*(ast: ClAst): string =
  ## Generates OpenCL C source code from a ClAst without the fp64 pragma
  var ctx = ClContext()
  result = ctx.toClC(ast)

macro opencl*(body: typed): string =
  ## Converts typed Nim AST to OpenCL C source code at compile time.
  ## Automatically adds cl_khr_fp64 pragma if double precision types are detected.
  ##
  ## Supported constructs:
  ## - proc/func definitions with {.kernel.} or {.device.} pragmas
  ## - let/var declarations
  ## - if/else, for, while
  ## - basic arithmetic, comparisons, bitwise ops
  ## - array indexing, struct field access
  ## - type definitions (converted to structs)
  ##
  ## Example:
  ## ```nim
  ## let code = opencl:
  ##   proc vecAdd(a: ptr float32, b: ptr float32, c: ptr float32, n: int32) {.kernel.} =
  ##     let gid = get_global_id(0)
  ##     if gid < n:
  ##       c[gid] = a[gid] + b[gid]
  ## ```
  var ctx = ClContext()
  let gpuAst = ctx.toClAst(body)

  # Check if double precision is used
  let needsFp64 = usesDoublePrecision(gpuAst)

  var code = ""
  if needsFp64:
    code = DoublePrecisionPragma
  code &= ctx.genOpenCL(gpuAst)
  result = newLit(code)

macro openclNoPragma*(body: typed): string =
  ## Like opencl but never adds the fp64 pragma (for manual control)
  var ctx = ClContext()
  let gpuAst = ctx.toClAst(body)
  let code = ctx.genOpenCL(gpuAst)
  result = newLit(code)

#[ tests ]#

when isMainModule:
  import std/unittest
  import clbase  # For OpenCL runtime execution tests

  suite "OpenCL Code Generation":

    test "Basic vector addition kernel":
      let code = opencl:
        proc vecAdd(a: ptr UncheckedArray[float32],
                    b: ptr UncheckedArray[float32],
                    c: ptr UncheckedArray[float32],
                    n: int32) {.kernel.} =
          let gid: int32 = get_global_id(0)
          if gid < n:
            c[gid] = a[gid] + b[gid]

      check "__kernel" in code
      check "get_global_id(0)" in code
      check "__global float*" in code
      check "c[gid] = (a[gid] + b[gid])" in code
      echo "\n=== Basic Vector Addition ===\n", code
    
    test "Function with result variable":
      let code = opencl:
        proc square(x: float32): float32 =
          result = x * x

      check "float result;" in code
      check "return result;" in code
      check "result = (x * x)" in code
      echo "\n=== Function with Result ===\n", code

    test "For loop":
      let code = opencl:
        proc sumArray(arr: ptr UncheckedArray[float32], n: int32): float32 =
          var sum: float32 = 0.0'f32
          for i in 0 ..< n:
            sum = sum + arr[i]
          result = sum

      check "for (int i = 0; i < n; i++)" in code
      echo "\n=== For Loop ===\n", code

    test "While loop":
      let code = opencl:
        proc countDown(start: int32): int32 =
          var n: int32 = start
          var count: int32 = 0
          while n > 0:
            count = count + 1
            n = n - 1
          result = count

      # Note: Nim may swap comparison order (n > 0) to (0 < n)
      check "while ((" in code
      check "< n))" in code or "(n > 0))" in code
      echo "\n=== While Loop ===\n", code

    test "If-else statement":
      let code = opencl:
        proc absVal(x: float32): float32 =
          if x < 0.0'f32:
            result = 0.0'f32 - x
          else:
            result = x

      check "if ((x < 0.0f))" in code
      check "} else {" in code
      echo "\n=== If-Else ===\n", code

  suite "OpenCL Runtime Execution":
    
    test "Vector addition kernel - compile and run":
      # Generate OpenCL C code from Nim
      let kernelCode = opencl:
        proc vadd(
          a: ptr UncheckedArray[float],
          b: ptr UncheckedArray[float],
          c: ptr UncheckedArray[float]
        ) {.kernel.} =
          let gid: int32 = get_global_id(0)
          c[gid] = a[gid] + b[gid]

      echo "\n=== Generated Kernel Code ===\n", kernelCode

      # Initialize OpenCL
      initCL()
      
      check clDevices.len > 0
      echo "Using device: ", clDevices[0].name

      # Compile the generated kernel
      let program = clContext.createAndBuild(kernelCode, clDevices)
      var kernel = program.createKernel("vadd")

      # Set up test data
      const size = 1_000_000
      var
        a = newSeq[float](size)
        b = newSeq[float](size)
        c = newSeq[float](size)

      for i in 0 ..< size:
        a[i] = float(i)
        b[i] = float(2 * i)

      # Allocate GPU buffers
      var
        gpuA = clContext.bufferLike(a)
        gpuB = clContext.bufferLike(b)
        gpuC = clContext.bufferLike(c)

      # Set kernel arguments
      kernel.args(gpuA, gpuB, gpuC)

      # Execute
      clQueues[0].write(a, gpuA)
      clQueues[0].write(b, gpuB)
      clQueues[0].run(kernel, size)
      clQueues[0].read(c, gpuC)

      # Verify results
      var errors = 0
      for i in 0 ..< size:
        if c[i] != a[i] + b[i]:
          inc errors
          if errors <= 5:
            echo "Error at index ", i, ": expected ", a[i] + b[i], " got ", c[i]

      check errors == 0
      echo "Verified ", size, " elements successfully!"

      # Clean up
      release(kernel)
      release(program)
      release(gpuA)
      release(gpuB)
      release(gpuC)
      finalizeCL()

    test "Scalar multiplication kernel":
      # Generate OpenCL C code from Nim
      let kernelCode = opencl:
        proc scalarMul(
          arr: ptr UncheckedArray[float],
          scalar: float32,
          result: ptr UncheckedArray[float]
        ) {.kernel.} =
          let gid: int32 = get_global_id(0)
          result[gid] = arr[gid] * scalar

      echo "\n=== Scalar Multiplication Kernel ===\n", kernelCode

      initCL()
      let program = clContext.createAndBuild(kernelCode, clDevices)
      var kernel = program.createKernel("scalarMul")

      const size = 500_000
      var scalar: float32 = 3.5'f32
      var
        input = newSeq[float](size)
        output = newSeq[float](size)

      for i in 0 ..< size:
        input[i] = float(i)

      var
        gpuIn = clContext.bufferLike(input)
        gpuOut = clContext.bufferLike(output)

      kernel.args(gpuIn, scalar, gpuOut)

      clQueues[0].write(input, gpuIn)
      clQueues[0].run(kernel, size)
      clQueues[0].read(output, gpuOut)

      var errors = 0
      for i in 0 ..< size:
        let expected = input[i] * scalar
        # Use relative error for larger values, absolute for small
        let tol = max(1e-6, abs(expected) * 1e-6)
        if abs(output[i] - expected) > tol:
          inc errors
          if errors <= 5:
            echo "Error at ", i, ": expected ", expected, " got ", output[i]

      check errors == 0
      echo "Scalar multiplication verified for ", size, " elements!"

      release(kernel)
      release(program)
      release(gpuIn)
      release(gpuOut)
      finalizeCL()

    test "SAXPY kernel (a*x + y)":
      let kernelCode = opencl:
        proc saxpy(
          x: ptr UncheckedArray[float],
          y: ptr UncheckedArray[float],
          a: float32,
          result: ptr UncheckedArray[float]
        ) {.kernel.} =
          let gid: int32 = get_global_id(0)
          result[gid] = a * x[gid] + y[gid]

      echo "\n=== SAXPY Kernel ===\n", kernelCode

      initCL()
      let program = clContext.createAndBuild(kernelCode, clDevices)
      var kernel = program.createKernel("saxpy")

      const size = 1_000_000
      var a: float32 = 2.5'f32
      var
        x = newSeq[float](size)
        y = newSeq[float](size)
        res = newSeq[float](size)

      for i in 0 ..< size:
        x[i] = float(i)
        y[i] = float(size - i)

      var
        gpuX = clContext.bufferLike(x)
        gpuY = clContext.bufferLike(y)
        gpuRes = clContext.bufferLike(res)

      kernel.args(gpuX, gpuY, a, gpuRes)

      clQueues[0].write(x, gpuX)
      clQueues[0].write(y, gpuY)
      clQueues[0].run(kernel, size)
      clQueues[0].read(res, gpuRes)

      var errors = 0
      for i in 0 ..< size:
        let expected = a * x[i] + y[i]
        # Use relative error for larger values, absolute for small
        let tol = max(1e-6, abs(expected) * 1e-6)
        if abs(res[i] - expected) > tol:
          inc errors
          if errors <= 5:
            echo "Error at ", i, ": expected ", expected, " got ", res[i]

      check errors == 0
      echo "SAXPY verified for ", size, " elements!"

      release(kernel)
      release(program)
      release(gpuX)
      release(gpuY)
      release(gpuRes)
      finalizeCL()

    test "Element-wise square kernel":
      let kernelCode = opencl:
        proc square(
          input: ptr UncheckedArray[float32],
          output: ptr UncheckedArray[float32]
        ) {.kernel.} =
          let gid: int32 = get_global_id(0)
          let val: float32 = input[gid]
          output[gid] = val * val

      echo "\n=== Square Kernel ===\n", kernelCode

      initCL()
      let program = clContext.createAndBuild(kernelCode, clDevices)
      var kernel = program.createKernel("square")

      const size = 500_000
      var
        input = newSeq[float32](size)
        output = newSeq[float32](size)

      for i in 0 ..< size:
        input[i] = float(i) * 0.001

      var
        gpuIn = clContext.bufferLike(input)
        gpuOut = clContext.bufferLike(output)

      kernel.args(gpuIn, gpuOut)

      clQueues[0].write(input, gpuIn)
      clQueues[0].run(kernel, size)
      clQueues[0].read(output, gpuOut)

      var errors = 0
      for i in 0 ..< size:
        let expected = input[i] * input[i]
        # float32 has ~7 digits of precision
        let tol = max(1e-7'f32, abs(expected) * 1e-6'f32)
        if abs(output[i] - expected) > tol:
          inc errors

      check errors == 0
      echo "Square kernel verified for ", size, " elements!"

      release(kernel)
      release(program)
      release(gpuIn)
      release(gpuOut)
      finalizeCL()

    test "Conditional kernel (clamp values)":
      let kernelCode = opencl:
        proc clampValues(
          input: ptr UncheckedArray[float],
          output: ptr UncheckedArray[float],
          minVal: float32,
          maxVal: float32
        ) {.kernel.} =
          let gid: int32 = get_global_id(0)
          var val: float32 = input[gid]
          if val < minVal:
            val = minVal
          if val > maxVal:
            val = maxVal
          output[gid] = val

      echo "\n=== Clamp Kernel ===\n", kernelCode

      initCL()
      let program = clContext.createAndBuild(kernelCode, clDevices)
      var kernel = program.createKernel("clampValues")

      const size = 100_000
      var minVal: float32 = 0.25'f32
      var maxVal: float32 = 0.75'f32
      var
        input = newSeq[float](size)
        output = newSeq[float](size)

      for i in 0 ..< size:
        input[i] = float(i) / float(size)

      var
        gpuIn = clContext.bufferLike(input)
        gpuOut = clContext.bufferLike(output)

      kernel.args(gpuIn, gpuOut, minVal, maxVal)

      clQueues[0].write(input, gpuIn)
      clQueues[0].run(kernel, size)
      clQueues[0].read(output, gpuOut)

      var errors = 0
      for i in 0..<size:
        var expected = input[i]
        if expected < minVal: expected = minVal
        if expected > maxVal: expected = maxVal
        # Clamp should be exact within float precision
        if abs(output[i] - expected) > 1e-7:
          inc errors

      check errors == 0
      echo "Clamp kernel verified for ", size, " elements!"

      release(kernel)
      release(program)
      release(gpuIn)
      release(gpuOut)
      finalizeCL()

    test "Integer arithmetic kernel":
      let kernelCode = opencl:
        proc intOps(
          a: ptr UncheckedArray[int32],
          b: ptr UncheckedArray[int32],
          sum: ptr UncheckedArray[int32],
          diff: ptr UncheckedArray[int32],
          prod: ptr UncheckedArray[int32]
        ) {.kernel.} =
          let gid: int32 = get_global_id(0)
          sum[gid] = a[gid] + b[gid]
          diff[gid] = a[gid] - b[gid]
          prod[gid] = a[gid] * b[gid]

      echo "\n=== Integer Ops Kernel ===\n", kernelCode

      initCL()
      let program = clContext.createAndBuild(kernelCode, clDevices)
      var kernel = program.createKernel("intOps")

      const size = 1000
      var
        a = newSeq[int32](size)
        b = newSeq[int32](size)
        sum = newSeq[int32](size)
        diff = newSeq[int32](size)
        prod = newSeq[int32](size)

      for i in 0 ..< size:
        a[i] = int32(i)
        b[i] = int32(size - i)

      var
        gpuA = clContext.bufferLike(a)
        gpuB = clContext.bufferLike(b)
        gpuSum = clContext.bufferLike(sum)
        gpuDiff = clContext.bufferLike(diff)
        gpuProd = clContext.bufferLike(prod)

      kernel.args(gpuA, gpuB, gpuSum, gpuDiff, gpuProd)

      clQueues[0].write(a, gpuA)
      clQueues[0].write(b, gpuB)
      clQueues[0].run(kernel, size)
      clQueues[0].read(sum, gpuSum)
      clQueues[0].read(diff, gpuDiff)
      clQueues[0].read(prod, gpuProd)

      var errors = 0
      for i in 0 ..< size:
        if sum[i] != a[i] + b[i]: inc errors
        if diff[i] != a[i] - b[i]: inc errors
        if prod[i] != a[i] * b[i]: inc errors

      check errors == 0
      echo "Integer ops verified for ", size, " elements!"

      release(kernel)
      release(program)
      release(gpuA)
      release(gpuB)
      release(gpuSum)
      release(gpuDiff)
      release(gpuProd)
      finalizeCL()

    test "Dot product partial sums":
      let kernelCode = opencl:
        proc dotPartial(
          a: ptr UncheckedArray[float],
          b: ptr UncheckedArray[float],
          partial: ptr UncheckedArray[float]
        ) {.kernel.} =
          let gid: int32 = get_global_id(0)
          partial[gid] = a[gid] * b[gid]

      echo "\n=== Dot Product Partial Sums Kernel ===\n", kernelCode

      initCL()
      let program = clContext.createAndBuild(kernelCode, clDevices)
      var kernel = program.createKernel("dotPartial")

      const size = 100_000
      var
        a = newSeq[float](size)
        b = newSeq[float](size)
        partial = newSeq[float](size)

      for i in 0 ..< size:
        a[i] = 1.0
        b[i] = float(i)

      var
        gpuA = clContext.bufferLike(a)
        gpuB = clContext.bufferLike(b)
        gpuPartial = clContext.bufferLike(partial)

      kernel.args(gpuA, gpuB, gpuPartial)

      clQueues[0].write(a, gpuA)
      clQueues[0].write(b, gpuB)
      clQueues[0].run(kernel, size)
      clQueues[0].read(partial, gpuPartial)

      # Sum partial results on CPU and verify
      var gpuSum = 0.0'f64
      var expectedSum = 0.0'f64
      for i in 0 ..< size:
        gpuSum += partial[i]
        expectedSum += a[i] * b[i]

      # Relative error for large sums
      let relError = abs(gpuSum - expectedSum) / max(1.0, abs(expectedSum))
      check relError < 1e-10
      echo "Dot product partial sums verified! Sum = ", gpuSum, " (rel error: ", relError, ")"

      release(kernel)
      release(program)
      release(gpuA)
      release(gpuB)
      release(gpuPartial)
      finalizeCL()
