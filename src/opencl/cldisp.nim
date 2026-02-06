#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/opencl/cldisp.nim
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

## OpenCL Dispatch Module for TensorFieldView
## 
## The `each` macro transforms for loops over TensorFieldView into OpenCL kernels.
## It handles:
## - AoSoA memory layout for coalesced GPU access
## - Matrix/vector operations (matmul, matadd, matvec, scalar ops)
## - Expression trees with operation type inference
## - Multi-device dispatch
## - Automatic type detection and OpenCL code generation

import std/[macros, tables, strutils]

import clbase
export clbase  # Export for setArg and other OpenCL functions

const VectorWidth* {.intdefine.} = 8
const DebugKernels* {.booldefine.} = false
const UseWorkGroups* {.booldefine.} = false

#[ ============================================================================
   View Information and Expression Types
   ============================================================================ ]#

type
  ElementType* = enum
    ## Element type for kernel code generation
    etFloat32   # float
    etFloat64   # double (default)
    etInt32     # int
    etInt64     # long

  ViewInfo = object
    name: NimNode      # The actual symbol
    nameStr: string    # String for kernel code gen
    isRead: bool
    isWrite: bool
    rank: int          # Tensor rank (1 for vector, 2 for matrix)
    shape: seq[int]    # Tensor shape
    elemType: ElementType  # Element type
  
  # Expression types - mirror the proxy types from sitetensor.nim
  ExprKind = enum
    ekSiteProxy,       # view[n] - direct site access
    ekMatMul,          # A * B - matrix multiplication (both rank 2)
    ekMatVec,          # A * v - matrix-vector multiplication (rank 2 * rank 1)
    ekMatAdd,          # A + B or A - B - element-wise add/sub
    ekScalarMul,       # scalar * A or A * scalar
    ekScalarAdd,       # scalar + A or A + scalar
    ekLiteral,         # numeric literal
    ekUnknown
  
  ExprInfo = object
    kind: ExprKind
    viewName: string   # For ekSiteProxy
    viewRank: int      # Tensor rank (1=vector, 2=matrix)
    isSubtract: bool   # For ekMatAdd: true if subtraction
    scalar: string     # For scalar ops: the scalar value as string
    left, right: ref ExprInfo  # For binary ops
  
  KernelInfo = object
    views: seq[ViewInfo]
    viewRanks: Table[string, int]  # Map view name to rank
    loopVar: NimNode
    loopVarStr: string
    elemType: ElementType  # Element type for OpenCL code gen
    outputRank: int    # Rank of output tensor (1=vector, 2=matrix)
    outputRows: int    # Number of rows in output
    outputCols: int    # Number of cols in output (1 for vectors)

proc getElementTypeFromNode(typeNode: NimNode): ElementType =
  ## Extract element type from a type node (compile-time)
  ## Handles TensorFieldView[L, T] where T is float32/float64/int32/int64
  
  # Walk through type node to find element type
  proc findElemType(n: NimNode): string =
    case n.kind
    of nnkSym:
      let s = n.strVal
      if s in ["float32", "float64", "int32", "int64", "cfloat", "cdouble", "cint", "clonglong"]:
        return s
    of nnkBracketExpr:
      # Generic type like TensorFieldView[L, T] - T is the last param
      if n.len >= 2:
        return findElemType(n[^1])  # Get last type param
    else:
      for child in n:
        let r = findElemType(child)
        if r != "":
          return r
    return ""
  
  let elemTypeStr = findElemType(typeNode)
  
  if elemTypeStr in ["float32", "cfloat"]:
    return etFloat32
  elif elemTypeStr in ["int32", "cint"]:
    return etInt32
  elif elemTypeStr in ["int64", "clonglong"]:
    return etInt64
  else:
    # Default to float64 for backward compatibility
    return etFloat64

proc elementTypeToOpenCL(et: ElementType): string =
  ## Get OpenCL C type name for element type
  case et
  of etFloat32: "float"
  of etFloat64: "double"
  of etInt32: "int"
  of etInt64: "long"

proc extractViewSym(n: NimNode): NimNode =
  ## Extract view symbol from various AST forms
  case n.kind
  of nnkSym: return n
  of nnkHiddenDeref: return extractViewSym(n[0])
  of nnkCall:
    if n.len >= 2:
      return extractViewSym(n[1])
  else: discard
  return nil

proc analyzeExpr(n: NimNode, viewNames: seq[string]): ExprInfo =
  ## Analyze an expression to determine its type and structure
  ## Works directly with AST structure, not type names
  
  case n.kind
  of nnkCall:
    if n.len >= 2 and n[0].kind == nnkSym:
      let opName = n[0].strVal
      
      # view[site] access: Call(Sym "[]", Sym "view", Sym "site")
      if opName == "[]":
        if n[1].kind == nnkSym:
          return ExprInfo(kind: ekSiteProxy, viewName: n[1].strVal)
        let viewSym = extractViewSym(n[1])
        if viewSym != nil:
          return ExprInfo(kind: ekSiteProxy, viewName: viewSym.strVal)
        return ExprInfo(kind: ekSiteProxy, viewName: "unknown")
      
      # Binary operators as calls
      if opName in ["+", "-"] and n.len >= 3:
        var leftInfo = new ExprInfo
        var rightInfo = new ExprInfo
        leftInfo[] = analyzeExpr(n[1], viewNames)
        rightInfo[] = analyzeExpr(n[2], viewNames)
        return ExprInfo(kind: ekMatAdd, isSubtract: opName == "-", left: leftInfo, right: rightInfo)
      
      if opName == "*" and n.len >= 3:
        var leftInfo = new ExprInfo
        var rightInfo = new ExprInfo
        leftInfo[] = analyzeExpr(n[1], viewNames)
        rightInfo[] = analyzeExpr(n[2], viewNames)
        
        # Determine if scalar or matrix multiply based on children
        let leftIsScalar = leftInfo.kind == ekLiteral
        let rightIsScalar = rightInfo.kind == ekLiteral
        
        if leftIsScalar:
          return ExprInfo(kind: ekScalarMul, scalar: leftInfo.scalar, right: rightInfo)
        elif rightIsScalar:
          return ExprInfo(kind: ekScalarMul, scalar: rightInfo.scalar, left: leftInfo)
        else:
          # Both are tensors -> matrix multiply
          return ExprInfo(kind: ekMatMul, left: leftInfo, right: rightInfo)
    
    # Unknown call - return unknown
    return ExprInfo(kind: ekUnknown)
  
  of nnkInfix:
    # Infix: op, left, right
    if n.len >= 3:
      let opName = if n[0].kind == nnkSym: n[0].strVal else: ""
      
      var leftInfo = new ExprInfo
      var rightInfo = new ExprInfo
      leftInfo[] = analyzeExpr(n[1], viewNames)
      rightInfo[] = analyzeExpr(n[2], viewNames)
      
      if opName in ["+", "-"]:
        return ExprInfo(kind: ekMatAdd, isSubtract: opName == "-", left: leftInfo, right: rightInfo)
      
      if opName == "*":
        let leftIsScalar = leftInfo.kind == ekLiteral
        let rightIsScalar = rightInfo.kind == ekLiteral
        
        if leftIsScalar:
          return ExprInfo(kind: ekScalarMul, scalar: leftInfo.scalar, right: rightInfo)
        elif rightIsScalar:
          return ExprInfo(kind: ekScalarMul, scalar: rightInfo.scalar, left: leftInfo)
        else:
          return ExprInfo(kind: ekMatMul, left: leftInfo, right: rightInfo)
    
    return ExprInfo(kind: ekUnknown)
  
  of nnkFloatLit..nnkFloat64Lit:
    return ExprInfo(kind: ekLiteral, scalar: $n.floatVal)
  
  of nnkIntLit..nnkInt64Lit:
    return ExprInfo(kind: ekLiteral, scalar: $n.intVal & ".0")
  
  of nnkSym:
    # Could be a view name or variable
    let name = n.strVal
    if name in viewNames:
      return ExprInfo(kind: ekSiteProxy, viewName: name)
    # Might be a float variable - treat as literal reference
    return ExprInfo(kind: ekLiteral, scalar: name)
  
  of nnkHiddenStdConv, nnkHiddenDeref, nnkConv:
    if n.len > 0:
      return analyzeExpr(n[^1], viewNames)
  
  else:
    discard
  
  return ExprInfo(kind: ekUnknown)

proc gatherViewInfo(body: NimNode, loopVar: NimNode): KernelInfo =
  ## Analyze body to find all TensorFieldView accesses
  result = KernelInfo(
    loopVar: loopVar,
    loopVarStr: loopVar.strVal,
    elemType: etFloat64,  # Default to double
    outputRank: 0,
    outputRows: 0,
    outputCols: 0
  )
  result.viewRanks = initTable[string, int]()
  var viewTable = initTable[string, ViewInfo]()
  
  proc analyzeNode(n: NimNode, isLHS: bool = false) =
    case n.kind
    of nnkCall:
      if n.len >= 2 and n[0].kind == nnkSym:
        let opName = n[0].strVal
        # view[site] = value write
        if opName == "[]=":
          let viewSym = extractViewSym(n[1])
          if viewSym != nil:
            let name = viewSym.strVal
            if name notin viewTable:
              let et = getElementTypeFromNode(viewSym.getTypeInst())
              viewTable[name] = ViewInfo(name: viewSym, nameStr: name, elemType: et)
            viewTable[name].isWrite = true
          # Recurse into RHS
          for i in 2..<n.len:
            analyzeNode(n[i])
        # view[site] read
        elif opName == "[]":
          let viewSym = extractViewSym(n[1])
          if viewSym != nil:
            let name = viewSym.strVal
            if name notin viewTable:
              let et = getElementTypeFromNode(viewSym.getTypeInst())
              viewTable[name] = ViewInfo(name: viewSym, nameStr: name, elemType: et)
            viewTable[name].isRead = true
          # Recurse
          for i in 2..<n.len:
            analyzeNode(n[i])
        else:
          # Other calls - recurse into all arguments
          for i in 1..<n.len:
            analyzeNode(n[i])
    of nnkAsgn:
      analyzeNode(n[0], isLHS = true)
      analyzeNode(n[1])
    of nnkHiddenDeref, nnkDerefExpr, nnkHiddenAddr:
      if n.len > 0:
        analyzeNode(n[0], isLHS)
    else:
      for child in n:
        analyzeNode(child, isLHS)
  
  analyzeNode(body)
  
  for name, info in viewTable:
    result.views.add info
    # Set element type from first write view (output)
    if info.isWrite:
      result.elemType = info.elemType

#[ ============================================================================
   OpenCL Kernel Code Generation
   ============================================================================ ]#

proc generateExprCode(expr: ExprInfo, rowVar, colVar: string): string =
  ## Generate OpenCL code for an expression that computes a single element [row, col]
  ## of the result matrix/tensor.
  
  case expr.kind
  of ekSiteProxy:
    # Direct element access - use view-specific elems
    let viewElems = expr.viewName & "_elems"
    result = expr.viewName & "_data[group * (VW * " & viewElems & ") + (" & rowVar & " * outCols + " & colVar & ") * VW + lane]"
  
  of ekMatMul, ekMatVec:
    # Matrix multiplication: sum_k(A[row,k] * B[k,col]) or matvec: sum_k(A[row,k] * v[k])
    let leftView = if expr.left != nil: expr.left.viewName else: "A"
    let rightView = if expr.right != nil: expr.right.viewName else: "B"
    result = "matmul_element(" & leftView & "_data, " & rightView & "_data, " & rowVar & ", " & colVar & ", outRows, " & leftView & "_elems, " & rightView & "_elems, group, lane, VW)"
  
  of ekMatAdd:
    let leftCode = if expr.left != nil: generateExprCode(expr.left[], rowVar, colVar) else: "0.0"
    let rightCode = if expr.right != nil: generateExprCode(expr.right[], rowVar, colVar) else: "0.0"
    let op = if expr.isSubtract: " - " else: " + "
    result = "(" & leftCode & op & rightCode & ")"
  
  of ekScalarMul:
    let tensorCode = if expr.left != nil: generateExprCode(expr.left[], rowVar, colVar)
                     elif expr.right != nil: generateExprCode(expr.right[], rowVar, colVar)
                     else: "0.0"
    result = "(" & expr.scalar & " * " & tensorCode & ")"
  
  of ekScalarAdd:
    let tensorCode = if expr.left != nil: generateExprCode(expr.left[], rowVar, colVar)
                     elif expr.right != nil: generateExprCode(expr.right[], rowVar, colVar)
                     else: "0.0"
    result = "(" & tensorCode & " + " & expr.scalar & ")"
  
  of ekLiteral:
    result = expr.scalar
  
  of ekUnknown:
    result = "0.0 /* unknown expr */"

proc hasMatMul(expr: ExprInfo): bool =
  ## Check if expression contains any matrix multiplication or matrix-vector
  case expr.kind
  of ekMatMul, ekMatVec: return true
  of ekMatAdd, ekScalarMul, ekScalarAdd:
    if expr.left != nil and hasMatMul(expr.left[]): return true
    if expr.right != nil and hasMatMul(expr.right[]): return true
    return false
  else:
    return false

#[ ============================================================================
   Print Statement Detection for CPU Fallback
   ============================================================================ ]#

proc hasEchoStatement(n: NimNode): bool =
  ## Check if the AST contains echo/debugEcho statements
  ## If so, we need to use CPU fallback instead of GPU kernel
  case n.kind
  of nnkCall:
    if n[0].kind == nnkSym:
      let name = n[0].strVal
      if name in ["echo", "debugEcho"]:
        return true
    # Check children
    for child in n:
      if hasEchoStatement(child):
        return true
  of nnkCommand:
    # echo "text" is parsed as nnkCommand
    if n[0].kind == nnkIdent and n[0].strVal == "echo":
      return true
    if n[0].kind == nnkSym and n[0].strVal in ["echo", "debugEcho"]:
      return true
    for child in n:
      if hasEchoStatement(child):
        return true
  of nnkStmtList:
    for child in n:
      if hasEchoStatement(child):
        return true
  else:
    for child in n:
      if hasEchoStatement(child):
        return true
  return false

proc hasDollarViewAccess(n: NimNode): bool =
  ## Check if the AST contains $view[n] patterns for formatting
  case n.kind
  of nnkPrefix:
    if n[0].kind == nnkIdent and n[0].strVal == "$":
      if n.len >= 2 and n[1].kind == nnkCall:
        # Could be $view[n]
        return true
  else:
    for child in n:
      if hasDollarViewAccess(child):
        return true
  return false

type
  ElementWrite = object
    viewName: string
    indices: seq[string]  # e.g., ["0", "1"] for [0,1] or ["0"] for [0]
    value: string         # The RHS value as a string

proc isElementLevelWrite(stmt: NimNode): bool =
  ## Check if a statement is an element-level write like view[n][i] = val or view[n][i,j] = val
  ## Pattern: Call(Sym "[]=", Call(Sym "[]", view, site), index, value)
  if stmt.kind != nnkCall: return false
  if stmt.len < 4: return false
  if stmt[0].kind != nnkSym or stmt[0].strVal != "[]=": return false
  
  # Check if the first arg is itself an index operation (double-indexing)
  let firstArg = stmt[1]
  if firstArg.kind == nnkCall and firstArg.len >= 2:
    if firstArg[0].kind == nnkSym and firstArg[0].strVal == "[]":
      return true
  return false

proc extractElementWrite(stmt: NimNode): ElementWrite =
  ## Extract info from an element-level write statement
  ## Pattern: Call(Sym "[]=", Call(Sym "[]", view, site), index1, [index2,] value)
  ## For 1D: stmt has 4 elements: "[]=", view[n], index, value
  ## For 2D: stmt has 5 elements: "[]=", view[n], row, col, value
  result = ElementWrite()
  
  if stmt.kind != nnkCall or stmt.len < 4: return
  
  let innerCall = stmt[1]  # view[site]
  if innerCall.kind == nnkCall and innerCall.len >= 2:
    let viewSym = extractViewSym(innerCall[1])
    if viewSym != nil:
      result.viewName = viewSym.strVal
  
  # Determine if 1D (4 args) or 2D (5 args)
  if stmt.len == 4:
    # 1D: "[]=", view[n], index, value
    let indexArg = stmt[2]
    case indexArg.kind
    of nnkIntLit..nnkInt64Lit:
      result.indices.add $indexArg.intVal
    else:
      result.indices.add "0"
    
    # Value is stmt[3]
    let valueArg = stmt[3]
    case valueArg.kind
    of nnkFloatLit..nnkFloat64Lit:
      result.value = $valueArg.floatVal
    of nnkIntLit..nnkInt64Lit:
      result.value = $valueArg.intVal & ".0"
    else:
      result.value = "0.0"
      
  elif stmt.len >= 5:
    # 2D: "[]=", view[n], row, col, value
    let rowArg = stmt[2]
    case rowArg.kind
    of nnkIntLit..nnkInt64Lit:
      result.indices.add $rowArg.intVal
    else:
      result.indices.add "0"
    
    let colArg = stmt[3]
    case colArg.kind
    of nnkIntLit..nnkInt64Lit:
      result.indices.add $colArg.intVal
    else:
      result.indices.add "0"
    
    # Value is stmt[4]
    let valueArg = stmt[4]
    case valueArg.kind
    of nnkFloatLit..nnkFloat64Lit:
      result.value = $valueArg.floatVal
    of nnkIntLit..nnkInt64Lit:
      result.value = $valueArg.intVal & ".0"
    else:
      result.value = "0.0"

proc generateKernelSource(
  kernelName: string,
  loopVarStr: string,
  body: NimNode,
  info: KernelInfo
): string =
  ## Generate complete OpenCL kernel source for TensorFieldView operations
  
  var viewNames: seq[string]
  for v in info.views:
    viewNames.add v.nameStr
  
  # Collect all statements from the body
  var statements: seq[NimNode]
  if body.kind == nnkStmtList:
    for child in body:
      statements.add child
  else:
    statements.add body
  
  # Check if we have element-level writes
  var elementWrites: seq[ElementWrite]
  var hasElementWrites = false
  for stmt in statements:
    if isElementLevelWrite(stmt):
      hasElementWrites = true
      elementWrites.add extractElementWrite(stmt)
  
  # For element-level writes, get the view name from the first write
  var lhsViewName: string = ""
  if hasElementWrites and elementWrites.len > 0:
    lhsViewName = elementWrites[0].viewName
  
  # Analyze the RHS expression for tensor-level operations
  var rhsExpr: ExprInfo
  var usesMatMul = false
  
  if not hasElementWrites:
    # The body can be either a StmtList containing the call, or the call directly
    var stmt: NimNode
    if body.kind == nnkStmtList and body.len > 0:
      stmt = body[0]
    else:
      stmt = body
    
    # Now check if this is an assignment call: []=
    if stmt.kind == nnkCall and stmt.len >= 4 and stmt[0].kind == nnkSym and stmt[0].strVal == "[]=":
      # stmt[1] is the view symbol directly (not a call)
      if stmt[1].kind == nnkSym:
        lhsViewName = stmt[1].strVal
      else:
        let viewSym = extractViewSym(stmt[1])
        if viewSym != nil:
          lhsViewName = viewSym.strVal
      
      # stmt[3] is the RHS expression
      rhsExpr = analyzeExpr(stmt[3], viewNames)
    
    usesMatMul = hasMatMul(rhsExpr)
  
  # If we didn't find the expected structure, check for other patterns
  if lhsViewName == "":
    for v in info.views:
      if v.isWrite:
        lhsViewName = v.nameStr
        break
  
  if lhsViewName == "":
    lhsViewName = "output"
  
  var src = ""
  
  # FP64 extension (also for int64)
  if info.elemType in {etFloat64, etInt64}:
    src &= "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n"
  
  let elemType = elementTypeToOpenCL(info.elemType)
  
  # Helper function for matrix multiplication element (matDim x matDim matrices)
  if usesMatMul:
    src &= "// Matrix-matrix multiplication: C[row,col] = sum_k(A[row,k] * B[k,col])\n"
    src &= "inline " & elemType & " matmul_element(\n"
    src &= "    __global const " & elemType & "* A,\n"
    src &= "    __global const " & elemType & "* B,\n"
    src &= "    int row, int col, int N, int Aelems, int Belems,\n"
    src &= "    int group, int lane, int VW\n"
    src &= ") {\n"
    src &= "  " & elemType & " sum = 0.0;\n"
    src &= "  for (int k = 0; k < N; k++) {\n"
    src &= "    // A is NxN matrix, B is either NxN matrix or N vector\n"
    src &= "    int aIdx = group * (VW * Aelems) + (row * N + k) * VW + lane;\n"
    src &= "    int bIdx;\n"
    src &= "    if (Belems == N * N) {\n"
    src &= "      // B is a matrix: index as [k, col]\n"
    src &= "      bIdx = group * (VW * Belems) + (k * N + col) * VW + lane;\n"
    src &= "    } else {\n"
    src &= "      // B is a vector: index as [k]\n"
    src &= "      bIdx = group * (VW * Belems) + k * VW + lane;\n"
    src &= "    }\n"
    src &= "    sum += A[aIdx] * B[bIdx];\n"
    src &= "  }\n"
    src &= "  return sum;\n"
    src &= "}\n\n"
  
  # Kernel parameters - now include elemsPerSite for each view
  var params: seq[string]
  for v in info.views:
    params.add "__global " & elemType & "* " & v.nameStr & "_data"
  for v in info.views:
    params.add "const int " & v.nameStr & "_elems"
  params.add "const int numSites"
  params.add "const int outRows"
  params.add "const int outCols"
  
  src &= "__kernel void " & kernelName & "(\n"
  src &= "    " & params.join(",\n    ")
  src &= "\n) {\n"
  
  # Work-item setup
  src &= "  const int " & loopVarStr & " = get_global_id(0);\n"
  src &= "  if (" & loopVarStr & " >= numSites) return;\n\n"
  
  # AoSoA layout constants
  let vw = $VectorWidth
  src &= "  const int VW = " & vw & ";\n"
  src &= "  const int group = " & loopVarStr & " / VW;\n"
  src &= "  const int lane = " & loopVarStr & " % VW;\n\n"
  
  src &= "  int outElems = " & lhsViewName & "_elems;\n\n"
  
  if hasElementWrites:
    # Element-level writes: generate individual assignments
    src &= "  // Element-level writes\n"
    for ew in elementWrites:
      # Calculate flat index from indices
      var flatIdx: string
      if ew.indices.len == 1:
        # Vector: flat index is just the index
        flatIdx = ew.indices[0]
      else:
        # Matrix: flat index is row * outCols + col
        flatIdx = ew.indices[0] & " * outCols + " & ew.indices[1]
      src &= "  {\n"
      src &= "    int idx = group * (VW * outElems) + (" & flatIdx & ") * VW + lane;\n"
      src &= "    " & ew.viewName & "_data[idx] = " & ew.value & ";\n"
      src &= "  }\n"
  else:
    # Tensor-level operation: Generate the computation loop
    src &= "  // Compute each element of the output tensor\n"
    src &= "  for (int _row = 0; _row < outRows; _row++) {\n"
    src &= "    for (int _col = 0; _col < outCols; _col++) {\n"
    src &= "      int outIdx = group * (VW * outElems) + (_row * outCols + _col) * VW + lane;\n"
    
    # Generate the RHS computation - pass info for view-specific indexing
    let rhsCode = generateExprCode(rhsExpr, "_row", "_col")
    src &= "      " & lhsViewName & "_data[outIdx] = " & rhsCode & ";\n"
    
    src &= "    }\n"
    src &= "  }\n"
  
  src &= "}\n"
  
  return src

#[ ============================================================================
   The `each` Macro - Main Entry Point
   ============================================================================ ]#

macro eachImpl*(loopVar: untyped, lo: typed, hi: typed, body: typed): untyped =
  ## Internal typed macro - receives body with full type information
  
  let loopVarSym = loopVar
  let info = gatherViewInfo(body, loopVarSym)
  
  if info.views.len == 0:
    error("No TensorFieldView found in loop body. The each macro requires at least one view access.")
  
  # Check for print statements - use CPU fallback if found
  if hasEchoStatement(body):
    # CPU fallback loop - must modify the original loopVar, not create new one
    # because body was already compiled with references to loopVar
    result = quote do:
      block:
        for cpuIdx in `lo`..<`hi`:
          `loopVarSym` = cpuIdx
          `body`
    return result
  
  if info.views.len == 0:
    error("No TensorFieldView found in loop body. The each macro requires at least one view access.")
  
  # Generate kernel
  let kernelName = "tfv_kernel_" & $body.lineInfoObj.line
  let kernelSource = generateKernelSource(kernelName, info.loopVarStr, body, info)
  
  let kernelNameLit = newLit(kernelName)
  let kernelSourceLit = newLit(kernelSource)
  
  # Get output view for shape info (first write view)
  var outViewSym: NimNode = nil
  for v in info.views:
    if v.isWrite:
      outViewSym = v.name
      break
  if outViewSym == nil:
    outViewSym = info.views[0].name
  
  # Use genSym to create unique identifiers
  let kernelSym = genSym(nskLet, "kernel")
  let programSym = genSym(nskLet, "program")
  let devIdxSym = genSym(nskForVar, "devIdx")
  
  # Build the setArg statements for buffers
  var setArgsStmts = newStmtList()
  for i, v in info.views:
    let viewSym = v.name
    let idxLit = newLit(i)
    let bufferAccess = nnkBracketExpr.newTree(
      nnkDotExpr.newTree(
        nnkDotExpr.newTree(viewSym, ident"data"),
        ident"buffers"
      ),
      devIdxSym
    )
    let setArgCall = nnkCall.newTree(
      nnkDotExpr.newTree(kernelSym, ident"setArg"),
      bufferAccess,
      idxLit
    )
    setArgsStmts.add setArgCall
  
  # Build the setArg statements for per-view elems
  var elemsArgsStmts = newStmtList()
  for i, v in info.views:
    let viewSym = v.name
    let elemsIdx = newLit(info.views.len + i)
    let elemsSym = genSym(nskVar, "elems_" & v.nameStr)
    # Get elementsPerSite from the view
    let getElems = quote do:
      var `elemsSym` = `viewSym`.data.elementsPerSite.int32
      `kernelSym`.setArg(`elemsSym`, `elemsIdx`)
    elemsArgsStmts.add getElems
  
  let numSitesIdx = newLit(info.views.len * 2)
  let outRowsIdx = newLit(info.views.len * 2 + 1)
  let outColsIdx = newLit(info.views.len * 2 + 2)
  
  result = quote do:
    block:
      when DebugKernels:
        echo "=== Generated OpenCL Kernel ==="
        echo `kernelSourceLit`
        echo "================================"
      
      let `programSym` = createAndBuild(clContext, `kernelSourceLit`, clDevices)
      let `kernelSym` = `programSym`.createKernel(`kernelNameLit`)
      
      let numDevices = clQueues.len
      let sitesPerDev = `outViewSym`.data.sitesPerDevice
      
      # Compute output dimensions from shape
      let outShape = `outViewSym`.shape
      let outRank = outShape.len
      let outRows = if outRank >= 1: outShape[0] else: 1
      let outCols = if outRank >= 2: outShape[1] else: 1
      
      for `devIdxSym` in 0..<numDevices:
        let devSites = sitesPerDev[`devIdxSym`]
        if devSites > 0:
          # Set buffer args
          `setArgsStmts`
          
          # Set per-view elems args
          `elemsArgsStmts`
          
          # Set numSites, outRows, outCols
          var nsArg = devSites.int32
          var orArg = outRows.int32
          var ocArg = outCols.int32
          `kernelSym`.setArg(nsArg, `numSitesIdx`)
          `kernelSym`.setArg(orArg, `outRowsIdx`)
          `kernelSym`.setArg(ocArg, `outColsIdx`)
          
          when UseWorkGroups:
            # Use explicit work-group size for debugging
            clQueues[`devIdxSym`].run(`kernelSym`, devSites, VectorWidth)
          else:
            clQueues[`devIdxSym`].run(`kernelSym`, devSites)
      
      for devIdx in 0..<numDevices:
        check clwrap.finish(clQueues[devIdx])
      
      release(`kernelSym`)
      release(`programSym`)

macro each*(x: ForLoopStmt): untyped =
  ## Transform a for loop over TensorFieldView into an OpenCL kernel dispatch.
  
  expectLen(x, 3)
  
  let loopVarNode = x[0]
  let callNode = x[1]
  let bodyNode = x[2]
  
  let iterNode = callNode[1]
  
  var lo, hi: NimNode
  if iterNode.kind == nnkInfix:
    lo = iterNode[1]
    hi = iterNode[2]
  else:
    error("each requires a range expression like 0..<N")
  
  result = quote do:
    block:
      var `loopVarNode` {.inject.}: int = 0
      eachImpl(`loopVarNode`, `lo`, `hi`, `bodyNode`)
