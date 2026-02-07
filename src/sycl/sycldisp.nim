#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/sycl/sycldisp.nim
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

## SYCL Dispatch Module for TensorFieldView - Native Kernel Edition
##
## This module provides the `each` macro for SYCL backend, using native
## pre-compiled C++ SYCL kernels instead of JIT-compiled OpenCL C.
##
## Architecture:
## - Macro analyzes the loop body at compile time to detect operation type
## - At runtime, dispatches to the appropriate pre-compiled native kernel
## - This approach works on all SYCL devices (CPU, GPU, accelerators)
##
## Supported Operations:
## - Copy: C[n] = A[n]
## - Add/Sub: C[n] = A[n] +/- B[n]
## - Scalar multiply: C[n] = scalar * A[n]
## - Matrix multiply: C[n] = A[n] * B[n]
## - Matrix-vector multiply: y[n] = M[n] * x[n]

import std/[macros, tables, strutils]

import syclbase
export syclbase

const VectorWidth* {.intdefine.} = 8
const DebugKernels* {.booldefine.} = false

#[ ============================================================================
   View Information and Expression Types
   ============================================================================ ]#

type
  ElementType* = enum
    ## Element type for kernel dispatch
    etFloat32   # float
    etFloat64   # double (default)
    etInt32     # int32
    etInt64     # int64

  ViewInfo = object
    name: NimNode      # The actual symbol
    nameStr: string    # String for debugging
    isRead: bool
    isWrite: bool
    rank: int          # Tensor rank (1 for vector, 2 for matrix)
    shape: seq[int]    # Tensor shape
    elemType: ElementType  # Element type for kernel dispatch
  
  # Stencil neighbor variable binding:
  # Maps variable name (e.g. "nbrIdx") to its stencil point index
  StencilBinding = object
    varName: string     # The let-bound variable name
    stencilSym: NimNode # The stencil symbol from the AST
    pointIdx: int       # The stencil point index (e.g. 0 for fwd-x)
  
  ExprKind* = enum
    ekSiteProxy,       # view[n] - direct site access (copy)
    ekMatMul,          # A * B - matrix multiplication (both rank 2)
    ekMatVec,          # A * v - matrix-vector multiplication (rank 2 * rank 1)
    ekMatAdd,          # A + B - element-wise add
    ekMatSub,          # A - B - element-wise subtract
    ekScalarMul,       # scalar * A or A * scalar
    ekScalarAdd,       # scalar + A or A + scalar
    ekLiteral,         # numeric literal
    ekUnknown
  
  ExprInfo* = object
    kind*: ExprKind
    viewName*: string  # For ekSiteProxy
    viewRank*: int     # Tensor rank (1=vector, 2=matrix)
    isComplex*: bool   # Whether this involves complex numbers
    isNeighborAccess*: bool  # True if this accesses a neighbor via stencil
    scalar*: float64   # For scalar ops
    scalarIm*: float64 # For complex scalar ops (imaginary part)
    left*, right*: ref ExprInfo  # For binary ops
  
  KernelInfo* = object
    views*: seq[ViewInfo]
    viewRanks*: Table[string, int]
    loopVar*: NimNode
    loopVarStr*: string
    outputRank*: int
    outputRows*: int
    outputCols*: int
    isComplex*: bool
    elemType*: ElementType  # Element type for kernel dispatch
    stencilBindings*: seq[StencilBinding]  # Stencil neighbor variables
    hasStencil*: bool   # True if stencil neighbor access detected

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

proc elementTypeSize(et: ElementType): int =
  ## Get size in bytes for an element type
  case et
  of etFloat32, etInt32: 4
  of etFloat64, etInt64: 8

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

proc getViewFromExpr(expr: ExprInfo): string =
  ## Get the view name from an expression, following through nested ops
  case expr.kind
  of ekSiteProxy:
    return expr.viewName
  of ekMatMul, ekMatVec, ekMatAdd, ekMatSub:
    if expr.left != nil:
      return getViewFromExpr(expr.left[])
  of ekScalarMul, ekScalarAdd:
    if expr.left != nil:
      return getViewFromExpr(expr.left[])
    if expr.right != nil:
      return getViewFromExpr(expr.right[])
  else:
    discard
  return ""

proc getExprDepth(expr: ExprInfo): int =
  ## Get the nesting depth of an expression
  case expr.kind
  of ekSiteProxy, ekLiteral:
    return 1
  of ekMatMul, ekMatVec, ekMatAdd, ekMatSub:
    var leftDepth = 0
    var rightDepth = 0
    if expr.left != nil:
      leftDepth = getExprDepth(expr.left[])
    if expr.right != nil:
      rightDepth = getExprDepth(expr.right[])
    return 1 + max(leftDepth, rightDepth)
  of ekScalarMul, ekScalarAdd:
    var depth = 0
    if expr.left != nil:
      depth = max(depth, getExprDepth(expr.left[]))
    if expr.right != nil:
      depth = max(depth, getExprDepth(expr.right[]))
    return 1 + depth
  else:
    return 1

proc getAllViewsFromExpr(expr: ExprInfo): seq[string] =
  ## Get all view names from an expression (for complex chains)
  case expr.kind
  of ekSiteProxy:
    return @[expr.viewName]
  of ekMatMul, ekMatVec, ekMatAdd, ekMatSub:
    var views: seq[string]
    if expr.left != nil:
      views.add getAllViewsFromExpr(expr.left[])
    if expr.right != nil:
      views.add getAllViewsFromExpr(expr.right[])
    return views
  of ekScalarMul, ekScalarAdd:
    if expr.left != nil:
      return getAllViewsFromExpr(expr.left[])
    if expr.right != nil:
      return getAllViewsFromExpr(expr.right[])
  else:
    discard
  return @[]

proc isSimpleBinaryOp(expr: ExprInfo): bool =
  ## Check if an expression is a simple binary operation on two views
  ## (not a nested chain like A*B + C*D)
  case expr.kind
  of ekSiteProxy:
    return true
  of ekMatMul, ekMatVec, ekMatAdd, ekMatSub:
    # Check if both sides are simple site proxies
    var leftSimple = false
    var rightSimple = false
    if expr.left != nil:
      leftSimple = expr.left.kind == ekSiteProxy
    if expr.right != nil:
      rightSimple = expr.right.kind == ekSiteProxy
    return leftSimple and rightSimple
  of ekScalarMul, ekScalarAdd:
    # Scalar ops with a simple view are simple
    if expr.left != nil and expr.left.kind == ekSiteProxy:
      return true
    if expr.right != nil and expr.right.kind == ekSiteProxy:
      return true
    return false
  else:
    return false

proc analyzeExpr*(n: NimNode, viewNames: seq[string], nbrVarNames: seq[string] = @[]): ExprInfo =
  ## Analyze an expression to determine its type and structure
  ## nbrVarNames: variable names that are stencil neighbor indices
  
  case n.kind
  of nnkCall:
    if n.len >= 2 and n[0].kind == nnkSym:
      let opName = n[0].strVal
      
      # view[site] access
      if opName == "[]":
        var viewName = ""
        if n[1].kind == nnkSym:
          viewName = n[1].strVal
        else:
          let viewSym = extractViewSym(n[1])
          if viewSym != nil:
            viewName = viewSym.strVal
          else:
            viewName = "unknown"
        
        # Check if the index argument is a stencil neighbor variable
        var isNbr = false
        if n.len >= 3 and n[2].kind == nnkSym:
          let idxName = n[2].strVal
          if idxName in nbrVarNames:
            isNbr = true
        return ExprInfo(kind: ekSiteProxy, viewName: viewName, isNeighborAccess: isNbr)
      
      # Binary operators
      if opName == "+" and n.len >= 3:
        var leftInfo = new ExprInfo
        var rightInfo = new ExprInfo
        leftInfo[] = analyzeExpr(n[1], viewNames, nbrVarNames)
        rightInfo[] = analyzeExpr(n[2], viewNames, nbrVarNames)
        
        # Check if left is scalar
        if leftInfo.kind == ekLiteral:
          return ExprInfo(kind: ekScalarAdd, scalar: leftInfo.scalar, right: rightInfo)
        elif rightInfo.kind == ekLiteral:
          return ExprInfo(kind: ekScalarAdd, scalar: rightInfo.scalar, left: leftInfo)
        else:
          return ExprInfo(kind: ekMatAdd, left: leftInfo, right: rightInfo)
      
      if opName == "-" and n.len >= 3:
        var leftInfo = new ExprInfo
        var rightInfo = new ExprInfo
        leftInfo[] = analyzeExpr(n[1], viewNames, nbrVarNames)
        rightInfo[] = analyzeExpr(n[2], viewNames, nbrVarNames)
        return ExprInfo(kind: ekMatSub, left: leftInfo, right: rightInfo)
      
      if opName == "*" and n.len >= 3:
        var leftInfo = new ExprInfo
        var rightInfo = new ExprInfo
        leftInfo[] = analyzeExpr(n[1], viewNames, nbrVarNames)
        rightInfo[] = analyzeExpr(n[2], viewNames, nbrVarNames)
        
        let leftIsScalar = leftInfo.kind == ekLiteral
        let rightIsScalar = rightInfo.kind == ekLiteral
        
        if leftIsScalar:
          return ExprInfo(kind: ekScalarMul, scalar: leftInfo.scalar, right: rightInfo)
        elif rightIsScalar:
          return ExprInfo(kind: ekScalarMul, scalar: rightInfo.scalar, left: leftInfo)
        else:
          # Could be matmul or matvec - we'll determine later from ranks
          return ExprInfo(kind: ekMatMul, left: leftInfo, right: rightInfo)
    
    return ExprInfo(kind: ekUnknown)
  
  of nnkInfix:
    if n.len >= 3:
      let opName = if n[0].kind == nnkSym: n[0].strVal 
                   elif n[0].kind == nnkIdent: n[0].strVal
                   else: ""
      
      var leftInfo = new ExprInfo
      var rightInfo = new ExprInfo
      leftInfo[] = analyzeExpr(n[1], viewNames, nbrVarNames)
      rightInfo[] = analyzeExpr(n[2], viewNames, nbrVarNames)
      
      if opName == "+":
        if leftInfo.kind == ekLiteral:
          return ExprInfo(kind: ekScalarAdd, scalar: leftInfo.scalar, right: rightInfo)
        elif rightInfo.kind == ekLiteral:
          return ExprInfo(kind: ekScalarAdd, scalar: rightInfo.scalar, left: leftInfo)
        else:
          return ExprInfo(kind: ekMatAdd, left: leftInfo, right: rightInfo)
      
      if opName == "-":
        return ExprInfo(kind: ekMatSub, left: leftInfo, right: rightInfo)
      
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
    return ExprInfo(kind: ekLiteral, scalar: n.floatVal)
  
  of nnkIntLit..nnkInt64Lit:
    return ExprInfo(kind: ekLiteral, scalar: n.intVal.float64)
  
  of nnkSym:
    let name = n.strVal
    if name in viewNames:
      return ExprInfo(kind: ekSiteProxy, viewName: name)
    # Assume it's a scalar variable - can't resolve value at compile time
    return ExprInfo(kind: ekLiteral, scalar: 0.0)  # Will be passed at runtime
  
  of nnkHiddenStdConv, nnkHiddenDeref, nnkConv:
    if n.len > 0:
      return analyzeExpr(n[^1], viewNames, nbrVarNames)
  
  else:
    discard
  
  return ExprInfo(kind: ekUnknown)

proc gatherViewInfo(body: NimNode, loopVar: NimNode): KernelInfo =
  ## Analyze body to find all TensorFieldView accesses and stencil bindings
  result = KernelInfo(
    loopVar: loopVar,
    loopVarStr: loopVar.strVal,
    outputRank: 0,
    outputRows: 0,
    outputCols: 0,
    isComplex: false,
    elemType: etFloat64,  # Default
    hasStencil: false
  )
  result.viewRanks = initTable[string, int]()
  var viewTable = initTable[string, ViewInfo]()
  
  proc analyzeNode(n: NimNode, isLHS: bool = false) =
    case n.kind
    of nnkCall:
      if n.len >= 2 and n[0].kind == nnkSym:
        let opName = n[0].strVal
        if opName == "[]=":
          let viewSym = extractViewSym(n[1])
          if viewSym != nil:
            let name = viewSym.strVal
            if name notin viewTable:
              let elemType = getElementTypeFromNode(viewSym.getTypeInst())
              viewTable[name] = ViewInfo(name: viewSym, nameStr: name, elemType: elemType)
            viewTable[name].isWrite = true
          for i in 2..<n.len:
            analyzeNode(n[i])
        elif opName == "[]":
          let viewSym = extractViewSym(n[1])
          if viewSym != nil:
            let name = viewSym.strVal
            if name notin viewTable:
              let elemType = getElementTypeFromNode(viewSym.getTypeInst())
              viewTable[name] = ViewInfo(name: viewSym, nameStr: name, elemType: elemType)
            viewTable[name].isRead = true
          for i in 2..<n.len:
            analyzeNode(n[i])
        else:
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
  
  # Detect stencil let-bindings
  # Pattern: LetSection(IdentDefs(Sym "nbrIdx", Empty, Call(Sym "neighbor", stencil, n, IntLit N)))
  if body.kind == nnkStmtList:
    for child in body:
      if child.kind == nnkLetSection:
        for identDef in child:
          if identDef.kind == nnkIdentDefs and identDef.len >= 3:
            let varName = if identDef[0].kind == nnkSym: identDef[0].strVal else: ""
            let valueExpr = identDef[2]
            # Check for Call(neighbor, stencil, loopVar, pointIdx)
            if valueExpr.kind == nnkCall and valueExpr.len >= 4 and
               valueExpr[0].kind == nnkSym and valueExpr[0].strVal == "neighbor":
              let stencilSym = valueExpr[1]
              let pointIdxNode = valueExpr[3]
              if pointIdxNode.kind in {nnkIntLit..nnkInt64Lit}:
                let pointIdx = pointIdxNode.intVal.int
                result.stencilBindings.add StencilBinding(
                  varName: varName,
                  stencilSym: stencilSym,
                  pointIdx: pointIdx
                )
                result.hasStencil = true
  
  for name, info in viewTable:
    result.views.add info
    # Set the kernel element type from the first write view (output)
    if info.isWrite:
      result.elemType = info.elemType

#[ ============================================================================
   Print Statement Detection for CPU Fallback
   ============================================================================ ]#

proc hasEchoStatement(n: NimNode): bool =
  case n.kind
  of nnkCall:
    if n[0].kind == nnkSym:
      let name = n[0].strVal
      if name in ["echo", "debugEcho"]:
        return true
    for child in n:
      if hasEchoStatement(child):
        return true
  of nnkCommand:
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

#[ ============================================================================
   Element-Level Write Detection and Extraction
   ============================================================================ ]#

type
  ElementWrite = object
    viewName: string
    viewSym: NimNode
    indices: seq[int]    # Element indices (e.g., [0, 1] for [0,1])
    value: float64       # The value to write

proc isElementLevelWrite(stmt: NimNode): bool =
  if stmt.kind != nnkCall: return false
  if stmt.len < 4: return false
  if stmt[0].kind != nnkSym or stmt[0].strVal != "[]=": return false
  
  let firstArg = stmt[1]
  if firstArg.kind == nnkCall and firstArg.len >= 2:
    if firstArg[0].kind == nnkSym and firstArg[0].strVal == "[]":
      return true
  return false

proc extractElementWrite(stmt: NimNode): ElementWrite =
  ## Extract info from an element-level write statement
  ## Pattern: Call(Sym "[]=", Call(Sym "[]", view, site), index1, [index2,] value)
  result = ElementWrite()
  
  if stmt.kind != nnkCall or stmt.len < 4: return
  
  let innerCall = stmt[1]  # view[site]
  if innerCall.kind == nnkCall and innerCall.len >= 2:
    let viewSym = extractViewSym(innerCall[1])
    if viewSym != nil:
      result.viewName = viewSym.strVal
      result.viewSym = viewSym
  
  # Determine if 1D (4 args) or 2D (5 args)
  if stmt.len == 4:
    # 1D: "[]=", view[n], index, value
    let indexArg = stmt[2]
    case indexArg.kind
    of nnkIntLit..nnkInt64Lit:
      result.indices.add indexArg.intVal.int
    else:
      result.indices.add 0
    
    let valueArg = stmt[3]
    case valueArg.kind
    of nnkFloatLit..nnkFloat64Lit:
      result.value = valueArg.floatVal
    of nnkIntLit..nnkInt64Lit:
      result.value = valueArg.intVal.float64
    else:
      result.value = 0.0
      
  elif stmt.len >= 5:
    # 2D: "[]=", view[n], row, col, value
    let rowArg = stmt[2]
    case rowArg.kind
    of nnkIntLit..nnkInt64Lit:
      result.indices.add rowArg.intVal.int
    else:
      result.indices.add 0
    
    let colArg = stmt[3]
    case colArg.kind
    of nnkIntLit..nnkInt64Lit:
      result.indices.add colArg.intVal.int
    else:
      result.indices.add 0
    
    let valueArg = stmt[4]
    case valueArg.kind
    of nnkFloatLit..nnkFloat64Lit:
      result.value = valueArg.floatVal
    of nnkIntLit..nnkInt64Lit:
      result.value = valueArg.intVal.float64
    else:
      result.value = 0.0

proc hasElementLevelWrites(body: NimNode): bool =
  ## Check if body contains any element-level writes
  var statements: seq[NimNode]
  if body.kind == nnkStmtList:
    for child in body:
      statements.add child
  else:
    statements.add body
  
  for stmt in statements:
    if isElementLevelWrite(stmt):
      return true
  return false

proc extractAllElementWrites(body: NimNode): seq[ElementWrite] =
  ## Extract all element-level writes from the body
  var statements: seq[NimNode]
  if body.kind == nnkStmtList:
    for child in body:
      statements.add child
  else:
    statements.add body
  
  for stmt in statements:
    if isElementLevelWrite(stmt):
      result.add extractElementWrite(stmt)

#[ ============================================================================
   Operation Dispatch - Maps AST analysis to native kernel calls
   ============================================================================ ]#

type
  DispatchKind* = enum
    dkCopy,         # C = A
    dkAdd,          # C = A + B
    dkSub,          # C = A - B  
    dkMul,          # C = A * B (element-wise)
    dkScalarMul,    # C = s * A
    dkScalarAdd,    # C = s + A
    dkMatMul,       # C = A * B (matrix multiply)
    dkMatVec,       # y = M * x (matrix-vector)
    dkMatAdd,       # C = A + B (for matrices)
    dkVecAdd,       # C = A + B (for vectors)
    dkStencilCopy,  # C[n] = A[neighbor(n)]  (gather)
    dkStencilScalarMul, # C[n] = s * A[neighbor(n)]  (gather + scalar)
    dkStencilAdd,   # C[n] = A[n] + B[neighbor(n)]  (add with gather)
    dkUnknown

#[ ============================================================================
   Execution Plan - Decompose complex expressions into kernel steps
   ============================================================================ ]#

type
  ExecStepKind* = enum
    eskCopy,        # dest = src
    eskAdd,         # dest = a + b
    eskSub,         # dest = a - b
    eskMatMul,      # dest = a * b (matrix multiply)
    eskMatVec,      # dest = a * b (matrix-vector)
    eskScalarMul,   # dest = s * a
    eskScalarAdd    # dest = s + a
  
  ExecOperand* = object
    ## Represents an operand: either a view name or a temp buffer ID
    isTemp*: bool
    viewName*: string   # For actual views
    tempId*: int        # For temp buffers (0, 1, 2, ...)
  
  ExecStep* = object
    kind*: ExecStepKind
    dest*: ExecOperand
    srcA*: ExecOperand
    srcB*: ExecOperand  # For binary ops
    scalar*: float64    # For scalar ops
  
  ExecPlan* = object
    steps*: seq[ExecStep]
    numTemps*: int      # Number of temporary buffers needed
    finalDest*: ExecOperand

proc newViewOperand(name: string): ExecOperand =
  ExecOperand(isTemp: false, viewName: name)

proc newTempOperand(id: int): ExecOperand =
  ExecOperand(isTemp: true, tempId: id)

proc decomposeExpr(expr: ExprInfo, destOp: ExecOperand, plan: var ExecPlan): ExecOperand =
  ## Recursively decompose an expression into execution steps.
  ## Returns the operand that holds the result of this sub-expression.
  
  case expr.kind
  of ekSiteProxy:
    # Leaf node - just return the view as operand
    return newViewOperand(expr.viewName)
  
  of ekMatMul, ekMatVec:
    # Matrix multiply or matvec
    var srcA, srcB: ExecOperand
    if expr.left != nil:
      srcA = decomposeExpr(expr.left[], newTempOperand(-1), plan)
    if expr.right != nil:
      srcB = decomposeExpr(expr.right[], newTempOperand(-1), plan)
    
    # If dest is a temp with id -1, allocate a new temp
    var actualDest = destOp
    if destOp.isTemp and destOp.tempId == -1:
      actualDest = newTempOperand(plan.numTemps)
      plan.numTemps += 1
    
    let stepKind = if expr.kind == ekMatVec: eskMatVec else: eskMatMul
    plan.steps.add ExecStep(kind: stepKind, dest: actualDest, srcA: srcA, srcB: srcB)
    return actualDest
  
  of ekMatAdd:
    var srcA, srcB: ExecOperand
    if expr.left != nil:
      srcA = decomposeExpr(expr.left[], newTempOperand(-1), plan)
    if expr.right != nil:
      srcB = decomposeExpr(expr.right[], newTempOperand(-1), plan)
    
    var actualDest = destOp
    if destOp.isTemp and destOp.tempId == -1:
      actualDest = newTempOperand(plan.numTemps)
      plan.numTemps += 1
    
    plan.steps.add ExecStep(kind: eskAdd, dest: actualDest, srcA: srcA, srcB: srcB)
    return actualDest
  
  of ekMatSub:
    var srcA, srcB: ExecOperand
    if expr.left != nil:
      srcA = decomposeExpr(expr.left[], newTempOperand(-1), plan)
    if expr.right != nil:
      srcB = decomposeExpr(expr.right[], newTempOperand(-1), plan)
    
    var actualDest = destOp
    if destOp.isTemp and destOp.tempId == -1:
      actualDest = newTempOperand(plan.numTemps)
      plan.numTemps += 1
    
    plan.steps.add ExecStep(kind: eskSub, dest: actualDest, srcA: srcA, srcB: srcB)
    return actualDest
  
  of ekScalarMul:
    var srcA: ExecOperand
    if expr.left != nil:
      srcA = decomposeExpr(expr.left[], newTempOperand(-1), plan)
    elif expr.right != nil:
      srcA = decomposeExpr(expr.right[], newTempOperand(-1), plan)
    
    var actualDest = destOp
    if destOp.isTemp and destOp.tempId == -1:
      actualDest = newTempOperand(plan.numTemps)
      plan.numTemps += 1
    
    plan.steps.add ExecStep(kind: eskScalarMul, dest: actualDest, srcA: srcA, scalar: expr.scalar)
    return actualDest
  
  of ekScalarAdd:
    var srcA: ExecOperand
    if expr.left != nil:
      srcA = decomposeExpr(expr.left[], newTempOperand(-1), plan)
    elif expr.right != nil:
      srcA = decomposeExpr(expr.right[], newTempOperand(-1), plan)
    
    var actualDest = destOp
    if destOp.isTemp and destOp.tempId == -1:
      actualDest = newTempOperand(plan.numTemps)
      plan.numTemps += 1
    
    plan.steps.add ExecStep(kind: eskScalarAdd, dest: actualDest, srcA: srcA, scalar: expr.scalar)
    return actualDest
  
  of ekLiteral, ekUnknown:
    # Shouldn't happen in normal usage
    return newViewOperand("")

proc buildExecPlan(expr: ExprInfo, lhsView: string): ExecPlan =
  ## Build an execution plan for a complete expression.
  ## The final result goes into lhsView.
  result = ExecPlan(numTemps: 0)
  result.finalDest = newViewOperand(lhsView)
  
  # Special case: simple copy (just view access)
  if expr.kind == ekSiteProxy:
    result.steps.add ExecStep(
      kind: eskCopy,
      dest: result.finalDest,
      srcA: newViewOperand(expr.viewName)
    )
    return
  
  # Decompose the expression, writing result to the LHS view
  discard decomposeExpr(expr, result.finalDest, result)
  
  # Optimize: if the last step writes to a temp and we need it in lhsView,
  # change the last step to write directly to lhsView
  if result.steps.len > 0:
    var lastStep = result.steps[^1]
    if lastStep.dest.isTemp:
      # Add a copy from temp to final dest
      result.steps.add ExecStep(
        kind: eskCopy,
        dest: result.finalDest,
        srcA: lastStep.dest
      )

type
  DispatchInfo* = object
    kind*: DispatchKind
    lhsView*: string          # Output view name
    rhsViews*: seq[string]    # Input view names in order
    scalar*: float64          # For scalar operations
    isComplex*: bool

proc determineDispatch(expr: ExprInfo, lhsView: string): DispatchInfo =
  ## Determine which native kernel to dispatch based on expression analysis
  result = DispatchInfo(kind: dkUnknown, lhsView: lhsView)
  
  case expr.kind
  of ekSiteProxy:
    # Simple copy or stencil gather copy
    if expr.isNeighborAccess:
      result.kind = dkStencilCopy
    else:
      result.kind = dkCopy
    result.rhsViews = @[expr.viewName]
  
  of ekMatAdd:
    # Addition: C[n] = A[n] + B[n] or stencil add C[n] = A[n] + B[neighbor(n)]
    let leftIsSimple = expr.left == nil or expr.left[].kind == ekSiteProxy
    let rightIsSimple = expr.right == nil or expr.right[].kind == ekSiteProxy
    if leftIsSimple and rightIsSimple:
      # Check if either side has a neighbor access
      let leftIsNbr = expr.left != nil and expr.left[].isNeighborAccess
      let rightIsNbr = expr.right != nil and expr.right[].isNeighborAccess
      if leftIsNbr or rightIsNbr:
        result.kind = dkStencilAdd
        # Put non-neighbor view first (srcA = direct, srcB = neighbor)
        if rightIsNbr:
          if expr.left != nil:
            let lv = getViewFromExpr(expr.left[])
            if lv != "": result.rhsViews.add lv
          if expr.right != nil:
            let rv = getViewFromExpr(expr.right[])
            if rv != "": result.rhsViews.add rv
        else:
          # Left is neighbor, right is direct - swap order
          if expr.right != nil:
            let rv = getViewFromExpr(expr.right[])
            if rv != "": result.rhsViews.add rv
          if expr.left != nil:
            let lv = getViewFromExpr(expr.left[])
            if lv != "": result.rhsViews.add lv
      else:
        result.kind = dkAdd
        if expr.left != nil:
          let lv = getViewFromExpr(expr.left[])
          if lv != "": result.rhsViews.add lv
        if expr.right != nil:
          let rv = getViewFromExpr(expr.right[])
          if rv != "": result.rhsViews.add rv
    else:
      result.kind = dkUnknown
  
  of ekMatSub:
    # Subtraction: C[n] = A[n] - B[n] - only if both operands are simple views
    let leftIsSimple = expr.left == nil or expr.left[].kind == ekSiteProxy
    let rightIsSimple = expr.right == nil or expr.right[].kind == ekSiteProxy
    if leftIsSimple and rightIsSimple:
      result.kind = dkSub
      if expr.left != nil:
        let lv = getViewFromExpr(expr.left[])
        if lv != "": result.rhsViews.add lv
      if expr.right != nil:
        let rv = getViewFromExpr(expr.right[])
        if rv != "": result.rhsViews.add rv
    else:
      result.kind = dkUnknown
  
  of ekScalarMul:
    # Scalar multiply: C[n] = s * A[n] or C[n] = s * A[neighbor(n)]
    result.scalar = expr.scalar
    # Check if the view operand is a neighbor access
    var viewExpr: ref ExprInfo = nil
    if expr.left != nil and expr.left[].kind == ekSiteProxy:
      viewExpr = expr.left
    elif expr.right != nil and expr.right[].kind == ekSiteProxy:
      viewExpr = expr.right
    
    if viewExpr != nil and viewExpr[].isNeighborAccess:
      result.kind = dkStencilScalarMul
      result.rhsViews.add viewExpr[].viewName
    else:
      result.kind = dkScalarMul
      if expr.left != nil:
        let lv = getViewFromExpr(expr.left[])
        if lv != "": result.rhsViews.add lv
      if expr.right != nil:
        let rv = getViewFromExpr(expr.right[])
        if rv != "": result.rhsViews.add rv
  
  of ekScalarAdd:
    # Scalar add: C[n] = s + A[n]
    result.kind = dkScalarAdd
    result.scalar = expr.scalar
    if expr.left != nil:
      let lv = getViewFromExpr(expr.left[])
      if lv != "": result.rhsViews.add lv
    if expr.right != nil:
      let rv = getViewFromExpr(expr.right[])
      if rv != "": result.rhsViews.add rv
  
  of ekMatMul, ekMatVec:
    # Matrix multiply or matvec - only if both operands are simple views
    let leftIsSimple = expr.left == nil or expr.left[].kind == ekSiteProxy
    let rightIsSimple = expr.right == nil or expr.right[].kind == ekSiteProxy
    if leftIsSimple and rightIsSimple:
      result.kind = dkMatMul
      if expr.left != nil:
        let lv = getViewFromExpr(expr.left[])
        if lv != "": result.rhsViews.add lv
      if expr.right != nil:
        let rv = getViewFromExpr(expr.right[])
        if rv != "": result.rhsViews.add rv
    else:
      result.kind = dkUnknown
  
  else:
    result.kind = dkUnknown

#[ ============================================================================
   The `each` Macro - Main Entry Point
   ============================================================================ ]#

macro eachImpl*(loopVar: untyped, lo: typed, hi: typed, body: typed): untyped =
  ## Internal typed macro - receives body with full type information.
  ## Generates runtime dispatch to native SYCL kernels.
  
  let loopVarSym = loopVar
  let info = gatherViewInfo(body, loopVarSym)
  
  if info.views.len == 0:
    error("No TensorFieldView found in loop body. The each macro requires at least one view access.")
  
  # Check for print statements - use CPU fallback if found
  if hasEchoStatement(body):
    result = quote do:
      block:
        for cpuIdx in `lo`..<`hi`:
          `loopVarSym` = cpuIdx
          `body`
    return result
  
  # Check for element-level writes - handle with native kernel
  if hasElementLevelWrites(body):
    let elementWrites = extractAllElementWrites(body)
    
    if elementWrites.len > 0:
      # Get the view symbol from the first write
      let viewSym = elementWrites[0].viewSym
      let vwLit = newLit(VectorWidth)
      
      # Build arrays of indices and values at compile time
      var indicesSeq = newSeq[int32](elementWrites.len)
      var valuesSeq = newSeq[float64](elementWrites.len)
      
      for i, ew in elementWrites:
        # Calculate flat index from row, col
        if ew.indices.len == 1:
          indicesSeq[i] = ew.indices[0].int32
        else:
          # For 2D: need to know cols at runtime, so we'll calculate flat index differently
          # For now, assume square matrices with shape from view
          indicesSeq[i] = ew.indices[0].int32  # Will be multiplied by cols at runtime
          valuesSeq[i] = ew.value
      
      let indicesLit = newLit(indicesSeq)
      let valuesLit = newLit(valuesSeq)
      let numWritesLit = newLit(elementWrites.len)
      
      # For 2D matrices, we need to compute flat indices at runtime
      # Check if any writes have 2 indices
      let is2D = elementWrites[0].indices.len >= 2
      
      if is2D:
        # For 2D matrices, we need to compute flat indices at runtime
        # Build the kernel calls in a way that works with the loop variable
        
        # Collect all write parameters
        var rowLits = newSeq[NimNode]()
        var colLits = newSeq[NimNode]()
        var valueLits = newSeq[NimNode]()
        
        for ew in elementWrites:
          rowLits.add newLit(ew.indices[0])
          colLits.add newLit(ew.indices[1])
          valueLits.add newLit(ew.value)
        
        # Use shared symbols that will be resolved consistently
        let devIdxSym = genSym(nskForVar, "devIdx")
        let devSitesSym = genSym(nskLet, "devSites")
        let numVGSym = genSym(nskLet, "numVectorGroups")
        
        # Build a match statement that generates all the calls
        var kernelCalls = newStmtList()
        for i in 0..<elementWrites.len:
          let rowLit = rowLits[i]
          let colLit = colLits[i]
          let valueLit = valueLits[i]
          
          let callCode = quote do:
            block:
              let outCols = if `viewSym`.shape.len >= 2: `viewSym`.shape[1] else: 1
              let flatIdx = `rowLit` * outCols + `colLit`
              kernelSetElement(syclQueues[`devIdxSym`], `viewSym`.data.buffers[`devIdxSym`],
                              flatIdx, `valueLit`,
                              `devSitesSym`, `viewSym`.data.elementsPerSite,
                              `vwLit`, `numVGSym`)
          kernelCalls.add callCode
        
        result = quote do:
          block:
            let numDevices = syclQueues.len
            let sitesPerDev = `viewSym`.data.sitesPerDevice
            
            for `devIdxSym` in 0..<numDevices:
              let `devSitesSym` = sitesPerDev[`devIdxSym`]
              if `devSitesSym` > 0:
                let `numVGSym` = (`devSitesSym` + `vwLit` - 1) div `vwLit`
                `kernelCalls`
            
            for `devIdxSym` in 0..<numDevices:
              discard finish(syclQueues[`devIdxSym`])
      else:
        # 1D case - simpler
        let devIdxSym = genSym(nskForVar, "devIdx")
        let devSitesSym = genSym(nskLet, "devSites")
        let numVGSym = genSym(nskLet, "numVectorGroups")
        
        var kernelCalls = newStmtList()
        for ew in elementWrites:
          let idxLit = newLit(ew.indices[0])
          let valueLit = newLit(ew.value)
          
          let callCode = quote do:
            kernelSetElement(syclQueues[`devIdxSym`], `viewSym`.data.buffers[`devIdxSym`],
                            `idxLit`, `valueLit`,
                            `devSitesSym`, `viewSym`.data.elementsPerSite,
                            `vwLit`, `numVGSym`)
          kernelCalls.add callCode
        
        result = quote do:
          block:
            let numDevices = syclQueues.len
            let sitesPerDev = `viewSym`.data.sitesPerDevice
            
            for `devIdxSym` in 0..<numDevices:
              let `devSitesSym` = sitesPerDev[`devIdxSym`]
              if `devSitesSym` > 0:
                let `numVGSym` = (`devSitesSym` + `vwLit` - 1) div `vwLit`
                `kernelCalls`
            
            for `devIdxSym` in 0..<numDevices:
              discard finish(syclQueues[`devIdxSym`])
      
      return result
  
  # Find the assignment statement (skip LetSection and other non-assignment stmts)
  var stmt: NimNode = nil
  if body.kind == nnkStmtList:
    for child in body:
      if child.kind == nnkCall and child.len >= 4 and child[0].kind == nnkSym and child[0].strVal == "[]=":
        stmt = child
        break
    if stmt == nil and body.len > 0:
      stmt = body[0]
  else:
    stmt = body
  
  # Gather view names for analysis
  var viewNames: seq[string]
  for v in info.views:
    viewNames.add v.nameStr
  
  # Collect stencil neighbor variable names for expression analysis
  var nbrVarNames: seq[string]
  for sb in info.stencilBindings:
    nbrVarNames.add sb.varName
  
  # Find output (LHS) view
  var lhsViewSym: NimNode = nil
  var lhsViewName = ""
  for v in info.views:
    if v.isWrite:
      lhsViewSym = v.name
      lhsViewName = v.nameStr
      break
  
  if lhsViewSym == nil:
    lhsViewSym = info.views[0].name
    lhsViewName = info.views[0].nameStr
  
  # Analyze RHS expression
  var rhsExpr: ExprInfo
  if stmt != nil and stmt.kind == nnkCall and stmt.len >= 4 and stmt[0].kind == nnkSym and stmt[0].strVal == "[]=":
    rhsExpr = analyzeExpr(stmt[3], viewNames, nbrVarNames)
  else:
    rhsExpr = ExprInfo(kind: ekUnknown)
  
  # Determine dispatch type
  let dispatch = determineDispatch(rhsExpr, lhsViewName)
  
  when DebugKernels:
    echo "SYCL dispatch: ", dispatch.kind, " lhs=", dispatch.lhsView, " rhs=", dispatch.rhsViews
  
  # Get view symbols for code generation
  var rhsView1Sym, rhsView2Sym: NimNode = nil
  for v in info.views:
    if dispatch.rhsViews.len >= 1 and v.nameStr == dispatch.rhsViews[0]:
      rhsView1Sym = v.name
    if dispatch.rhsViews.len >= 2 and v.nameStr == dispatch.rhsViews[1]:
      rhsView2Sym = v.name
  
  let vwLit = newLit(VectorWidth)
  
  # Generate dispatch code
  # Get element type identifier for typed kernel calls
  let elemTypeIdent = case info.elemType
    of etFloat32: ident("float32")
    of etFloat64: ident("float64")
    of etInt32: ident("int32")
    of etInt64: ident("int64")
  
  # Generate typed scalar literal
  let scalarLit = case info.elemType
    of etFloat32: newLit(dispatch.scalar.float32)
    of etFloat64: newLit(dispatch.scalar)
    of etInt32: newLit(dispatch.scalar.int32)
    of etInt64: newLit(dispatch.scalar.int64)
  
  case dispatch.kind
  of dkCopy:
    if rhsView1Sym == nil:
      error("Could not find source view for copy operation")
    
    result = quote do:
      block:
        let numDevices = syclQueues.len
        let sitesPerDev = `lhsViewSym`.data.sitesPerDevice
        let elemsPerSite = `lhsViewSym`.data.elementsPerSite
        
        for devIdx in 0..<numDevices:
          let devSites = sitesPerDev[devIdx]
          if devSites > 0:
            let numVectorGroups = (devSites + `vwLit` - 1) div `vwLit`
            let numElements = numVectorGroups * elemsPerSite * `vwLit`
            
            let srcBuf = `rhsView1Sym`.data.buffers[devIdx]
            let dstBuf = `lhsViewSym`.data.buffers[devIdx]
            
            kernelCopy(syclQueues[devIdx], srcBuf, dstBuf, numElements, `elemTypeIdent`)
        
        for devIdx in 0..<numDevices:
          discard finish(syclQueues[devIdx])
  
  of dkAdd:
    if rhsView1Sym == nil or rhsView2Sym == nil:
      error("Could not find views for add operation")
    
    result = quote do:
      block:
        let numDevices = syclQueues.len
        let sitesPerDev = `lhsViewSym`.data.sitesPerDevice
        let elemsPerSite = `lhsViewSym`.data.elementsPerSite
        
        for devIdx in 0..<numDevices:
          let devSites = sitesPerDev[devIdx]
          if devSites > 0:
            let numVectorGroups = (devSites + `vwLit` - 1) div `vwLit`
            let numElements = numVectorGroups * elemsPerSite * `vwLit`
            
            let bufA = `rhsView1Sym`.data.buffers[devIdx]
            let bufB = `rhsView2Sym`.data.buffers[devIdx]
            let bufC = `lhsViewSym`.data.buffers[devIdx]
            
            kernelAdd(syclQueues[devIdx], bufA, bufB, bufC, numElements, `elemTypeIdent`)
        
        for devIdx in 0..<numDevices:
          discard finish(syclQueues[devIdx])
  
  of dkSub:
    if rhsView1Sym == nil or rhsView2Sym == nil:
      error("Could not find views for subtract operation")
    
    result = quote do:
      block:
        let numDevices = syclQueues.len
        let sitesPerDev = `lhsViewSym`.data.sitesPerDevice
        let elemsPerSite = `lhsViewSym`.data.elementsPerSite
        
        for devIdx in 0..<numDevices:
          let devSites = sitesPerDev[devIdx]
          if devSites > 0:
            let numVectorGroups = (devSites + `vwLit` - 1) div `vwLit`
            let numElements = numVectorGroups * elemsPerSite * `vwLit`
            
            let bufA = `rhsView1Sym`.data.buffers[devIdx]
            let bufB = `rhsView2Sym`.data.buffers[devIdx]
            let bufC = `lhsViewSym`.data.buffers[devIdx]
            
            kernelSub(syclQueues[devIdx], bufA, bufB, bufC, numElements, `elemTypeIdent`)
        
        for devIdx in 0..<numDevices:
          discard finish(syclQueues[devIdx])
  
  of dkScalarMul:
    if rhsView1Sym == nil:
      error("Could not find view for scalar multiply operation")
    
    result = quote do:
      block:
        let numDevices = syclQueues.len
        let sitesPerDev = `lhsViewSym`.data.sitesPerDevice
        let elemsPerSite = `lhsViewSym`.data.elementsPerSite
        
        for devIdx in 0..<numDevices:
          let devSites = sitesPerDev[devIdx]
          if devSites > 0:
            let numVectorGroups = (devSites + `vwLit` - 1) div `vwLit`
            let numElements = numVectorGroups * elemsPerSite * `vwLit`
            
            let bufA = `rhsView1Sym`.data.buffers[devIdx]
            let bufC = `lhsViewSym`.data.buffers[devIdx]
            
            kernelScalarMul(syclQueues[devIdx], bufA, `scalarLit`, bufC, numElements, `elemTypeIdent`)
        
        for devIdx in 0..<numDevices:
          discard finish(syclQueues[devIdx])
  
  of dkScalarAdd:
    if rhsView1Sym == nil:
      error("Could not find view for scalar add operation")
    
    result = quote do:
      block:
        let numDevices = syclQueues.len
        let sitesPerDev = `lhsViewSym`.data.sitesPerDevice
        let elemsPerSite = `lhsViewSym`.data.elementsPerSite
        
        for devIdx in 0..<numDevices:
          let devSites = sitesPerDev[devIdx]
          if devSites > 0:
            let numVectorGroups = (devSites + `vwLit` - 1) div `vwLit`
            let numElements = numVectorGroups * elemsPerSite * `vwLit`
            
            let bufA = `rhsView1Sym`.data.buffers[devIdx]
            let bufC = `lhsViewSym`.data.buffers[devIdx]
            
            kernelScalarAdd(syclQueues[devIdx], bufA, `scalarLit`, bufC, numElements, `elemTypeIdent`)
        
        for devIdx in 0..<numDevices:
          discard finish(syclQueues[devIdx])
  
  of dkMatMul, dkMatAdd, dkVecAdd, dkMul:
    if rhsView1Sym == nil or rhsView2Sym == nil:
      error("Could not find views for matrix operation")
    
    result = quote do:
      block:
        let numDevices = syclQueues.len
        let sitesPerDev = `lhsViewSym`.data.sitesPerDevice
        
        # Get shapes to determine operation type
        let outShape = `lhsViewSym`.shape
        let outRank = outShape.len
        let outRows = if outRank >= 1: outShape[0] else: 1
        let outCols = if outRank >= 2: outShape[1] else: 1
        
        let rhsShape1 = `rhsView1Sym`.shape
        let rhsShape2 = `rhsView2Sym`.shape
        let rhs1Rank = rhsShape1.len
        let rhs2Rank = rhsShape2.len
        
        for devIdx in 0..<numDevices:
          let devSites = sitesPerDev[devIdx]
          if devSites > 0:
            let numVectorGroups = (devSites + `vwLit` - 1) div `vwLit`
            
            let bufA = `rhsView1Sym`.data.buffers[devIdx]
            let bufB = `rhsView2Sym`.data.buffers[devIdx]
            let bufC = `lhsViewSym`.data.buffers[devIdx]
            
            # Determine if this is matmul, matvec, or element-wise
            if rhs1Rank == 2 and rhs2Rank == 2:
              # Matrix multiply: C = A * B
              let rows = rhsShape1[0]
              let inner = rhsShape1[1]
              let cols = rhsShape2[1]
              kernelMatMul(syclQueues[devIdx], bufA, bufB, bufC,
                          devSites, rows, cols, inner, `vwLit`, numVectorGroups, `elemTypeIdent`)
            elif rhs1Rank == 2 and rhs2Rank == 1:
              # Matrix-vector: y = M * x
              let rows = rhsShape1[0]
              let cols = rhsShape1[1]
              kernelMatVec(syclQueues[devIdx], bufA, bufB, bufC,
                          devSites, rows, cols, `vwLit`, numVectorGroups, `elemTypeIdent`)
            else:
              # Element-wise add (fallback)
              let elemsPerSite = `lhsViewSym`.data.elementsPerSite
              let numElements = numVectorGroups * elemsPerSite * `vwLit`
              kernelAdd(syclQueues[devIdx], bufA, bufB, bufC, numElements, `elemTypeIdent`)
        
        for devIdx in 0..<numDevices:
          discard finish(syclQueues[devIdx])
  
  of dkMatVec:
    # Handled above in dkMatMul case
    result = quote do:
      discard
  
  of dkStencilCopy:
    # C[n] = A[neighbor(n, pointIdx)] - gather copy
    if rhsView1Sym == nil:
      error("Could not find source view for stencil copy operation")
    if info.stencilBindings.len == 0:
      error("Stencil copy detected but no stencil binding found")
    
    let stencilSym = info.stencilBindings[0].stencilSym
    let pointIdxLit = newLit(info.stencilBindings[0].pointIdx)
    
    result = quote do:
      block:
        let numDevices = syclQueues.len
        let sitesPerDev = `lhsViewSym`.data.sitesPerDevice
        let elemsPerSite = `lhsViewSym`.data.elementsPerSite
        let nPoints = `stencilSym`.nPoints
        let offsetBufSize = `stencilSym`.nLocalSites * nPoints * sizeof(int32)
        
        for devIdx in 0..<numDevices:
          let devSites = sitesPerDev[devIdx]
          if devSites > 0:
            # Upload stencil offset table to device
            let offsetBuf = allocate(syclQueues[devIdx], offsetBufSize)
            write(syclQueues[devIdx], cast[pointer](`stencilSym`.getOffsetBuffer()), offsetBuf, offsetBufSize)
            
            kernelStencilCopy(syclQueues[devIdx],
                              `rhsView1Sym`.data.buffers[devIdx],
                              `lhsViewSym`.data.buffers[devIdx],
                              offsetBuf,
                              `pointIdxLit`, nPoints,
                              devSites, elemsPerSite, `vwLit`,
                              `elemTypeIdent`)
            
            discard finish(syclQueues[devIdx])
            deallocate(syclQueues[devIdx], offsetBuf)
  
  of dkStencilScalarMul:
    # C[n] = scalar * A[neighbor(n, pointIdx)]
    if rhsView1Sym == nil:
      error("Could not find source view for stencil scalar mul operation")
    if info.stencilBindings.len == 0:
      error("Stencil scalar mul detected but no stencil binding found")
    
    let stencilSym = info.stencilBindings[0].stencilSym
    let pointIdxLit = newLit(info.stencilBindings[0].pointIdx)
    
    result = quote do:
      block:
        let numDevices = syclQueues.len
        let sitesPerDev = `lhsViewSym`.data.sitesPerDevice
        let elemsPerSite = `lhsViewSym`.data.elementsPerSite
        let nPoints = `stencilSym`.nPoints
        let offsetBufSize = `stencilSym`.nLocalSites * nPoints * sizeof(int32)
        
        for devIdx in 0..<numDevices:
          let devSites = sitesPerDev[devIdx]
          if devSites > 0:
            let offsetBuf = allocate(syclQueues[devIdx], offsetBufSize)
            write(syclQueues[devIdx], cast[pointer](`stencilSym`.getOffsetBuffer()), offsetBuf, offsetBufSize)
            
            kernelStencilScalarMul(syclQueues[devIdx],
                                   `rhsView1Sym`.data.buffers[devIdx],
                                   `scalarLit`,
                                   `lhsViewSym`.data.buffers[devIdx],
                                   offsetBuf,
                                   `pointIdxLit`, nPoints,
                                   devSites, elemsPerSite, `vwLit`,
                                   `elemTypeIdent`)
            
            discard finish(syclQueues[devIdx])
            deallocate(syclQueues[devIdx], offsetBuf)
  
  of dkStencilAdd:
    # C[n] = A[n] + B[neighbor(n, pointIdx)]
    # rhsView1 = A (direct access), rhsView2 = B (neighbor access)
    if rhsView1Sym == nil or rhsView2Sym == nil:
      error("Could not find views for stencil add operation")
    if info.stencilBindings.len == 0:
      error("Stencil add detected but no stencil binding found")
    
    let stencilSym = info.stencilBindings[0].stencilSym
    let pointIdxLit = newLit(info.stencilBindings[0].pointIdx)
    
    result = quote do:
      block:
        let numDevices = syclQueues.len
        let sitesPerDev = `lhsViewSym`.data.sitesPerDevice
        let elemsPerSite = `lhsViewSym`.data.elementsPerSite
        let nPoints = `stencilSym`.nPoints
        let offsetBufSize = `stencilSym`.nLocalSites * nPoints * sizeof(int32)
        
        for devIdx in 0..<numDevices:
          let devSites = sitesPerDev[devIdx]
          if devSites > 0:
            let offsetBuf = allocate(syclQueues[devIdx], offsetBufSize)
            write(syclQueues[devIdx], cast[pointer](`stencilSym`.getOffsetBuffer()), offsetBuf, offsetBufSize)
            
            kernelStencilAdd(syclQueues[devIdx],
                             `rhsView1Sym`.data.buffers[devIdx],
                             `rhsView2Sym`.data.buffers[devIdx],
                             `lhsViewSym`.data.buffers[devIdx],
                             offsetBuf,
                             `pointIdxLit`, nPoints,
                             devSites, elemsPerSite, `vwLit`,
                             `elemTypeIdent`)
            
            discard finish(syclQueues[devIdx])
            deallocate(syclQueues[devIdx], offsetBuf)
  
  of dkUnknown:
    # Complex expression - use execution plan with temporary buffers
    let plan = buildExecPlan(rhsExpr, lhsViewName)
    
    when DebugKernels:
      echo "SYCL: Building execution plan with ", plan.steps.len, " steps and ", plan.numTemps, " temps"
    
    # Build a table from view names to their symbols
    var viewSymTable = initTable[string, NimNode]()
    for v in info.views:
      viewSymTable[v.nameStr] = v.name
    
    let numTempsLit = newLit(plan.numTemps)
    
    # Create shared symbols that will be used consistently in all quotes
    let devIdxSym = genSym(nskForVar, "devIdx")
    let devSitesSym = genSym(nskLet, "devSites")
    let numVGSym = genSym(nskLet, "numVectorGroups")
    let numElemsSym = genSym(nskLet, "numElements")
    let tempBuffersSym = genSym(nskVar, "tempBuffers")
    
    # Generate code for each execution step using quote blocks
    # All steps will be spliced into the device loop where devIdx is defined
    var execCode = newStmtList()
    
    for stepIdx, step in plan.steps:
      # Get view symbols for this step
      var destSym, srcASym, srcBSym: NimNode
      
      if not step.dest.isTemp:
        destSym = viewSymTable.getOrDefault(step.dest.viewName, nil)
      if not step.srcA.isTemp:
        srcASym = viewSymTable.getOrDefault(step.srcA.viewName, nil)
      if step.kind in {eskAdd, eskSub, eskMatMul, eskMatVec} and not step.srcB.isTemp:
        srcBSym = viewSymTable.getOrDefault(step.srcB.viewName, nil)
      
      let destTempId = newLit(step.dest.tempId)
      let srcATempId = newLit(step.srcA.tempId)
      let srcBTempId = newLit(step.srcB.tempId)
      let scalarStepLit = newLit(step.scalar)
      let destIsTemp = step.dest.isTemp
      let srcAIsTemp = step.srcA.isTemp
      let srcBIsTemp = step.srcB.isTemp
      
      case step.kind
      of eskCopy:
        if srcAIsTemp:
          if destIsTemp:
            let stepCode = quote do:
              kernelCopy(syclQueues[`devIdxSym`], `tempBuffersSym`[`devIdxSym`][`srcATempId`], 
                        `tempBuffersSym`[`devIdxSym`][`destTempId`], `numElemsSym`)
            execCode.add stepCode
          else:
            let stepCode = quote do:
              kernelCopy(syclQueues[`devIdxSym`], `tempBuffersSym`[`devIdxSym`][`srcATempId`], 
                        `destSym`.data.buffers[`devIdxSym`], `numElemsSym`)
            execCode.add stepCode
        else:
          if destIsTemp:
            let stepCode = quote do:
              kernelCopy(syclQueues[`devIdxSym`], `srcASym`.data.buffers[`devIdxSym`], 
                        `tempBuffersSym`[`devIdxSym`][`destTempId`], `numElemsSym`)
            execCode.add stepCode
          else:
            let stepCode = quote do:
              kernelCopy(syclQueues[`devIdxSym`], `srcASym`.data.buffers[`devIdxSym`], 
                        `destSym`.data.buffers[`devIdxSym`], `numElemsSym`)
            execCode.add stepCode
      
      of eskAdd:
        # Generate all 8 combinations of src/dest being temp or view
        if srcAIsTemp and srcBIsTemp and destIsTemp:
          let stepCode = quote do:
            kernelAdd(syclQueues[`devIdxSym`], `tempBuffersSym`[`devIdxSym`][`srcATempId`],
                     `tempBuffersSym`[`devIdxSym`][`srcBTempId`], `tempBuffersSym`[`devIdxSym`][`destTempId`], `numElemsSym`)
          execCode.add stepCode
        elif srcAIsTemp and srcBIsTemp and not destIsTemp:
          let stepCode = quote do:
            kernelAdd(syclQueues[`devIdxSym`], `tempBuffersSym`[`devIdxSym`][`srcATempId`],
                     `tempBuffersSym`[`devIdxSym`][`srcBTempId`], `destSym`.data.buffers[`devIdxSym`], `numElemsSym`)
          execCode.add stepCode
        elif srcAIsTemp and not srcBIsTemp and destIsTemp:
          let stepCode = quote do:
            kernelAdd(syclQueues[`devIdxSym`], `tempBuffersSym`[`devIdxSym`][`srcATempId`],
                     `srcBSym`.data.buffers[`devIdxSym`], `tempBuffersSym`[`devIdxSym`][`destTempId`], `numElemsSym`)
          execCode.add stepCode
        elif srcAIsTemp and not srcBIsTemp and not destIsTemp:
          let stepCode = quote do:
            kernelAdd(syclQueues[`devIdxSym`], `tempBuffersSym`[`devIdxSym`][`srcATempId`],
                     `srcBSym`.data.buffers[`devIdxSym`], `destSym`.data.buffers[`devIdxSym`], `numElemsSym`)
          execCode.add stepCode
        elif not srcAIsTemp and srcBIsTemp and destIsTemp:
          let stepCode = quote do:
            kernelAdd(syclQueues[`devIdxSym`], `srcASym`.data.buffers[`devIdxSym`],
                     `tempBuffersSym`[`devIdxSym`][`srcBTempId`], `tempBuffersSym`[`devIdxSym`][`destTempId`], `numElemsSym`)
          execCode.add stepCode
        elif not srcAIsTemp and srcBIsTemp and not destIsTemp:
          let stepCode = quote do:
            kernelAdd(syclQueues[`devIdxSym`], `srcASym`.data.buffers[`devIdxSym`],
                     `tempBuffersSym`[`devIdxSym`][`srcBTempId`], `destSym`.data.buffers[`devIdxSym`], `numElemsSym`)
          execCode.add stepCode
        elif not srcAIsTemp and not srcBIsTemp and destIsTemp:
          let stepCode = quote do:
            kernelAdd(syclQueues[`devIdxSym`], `srcASym`.data.buffers[`devIdxSym`],
                     `srcBSym`.data.buffers[`devIdxSym`], `tempBuffersSym`[`devIdxSym`][`destTempId`], `numElemsSym`)
          execCode.add stepCode
        else: # not srcAIsTemp and not srcBIsTemp and not destIsTemp
          let stepCode = quote do:
            kernelAdd(syclQueues[`devIdxSym`], `srcASym`.data.buffers[`devIdxSym`],
                     `srcBSym`.data.buffers[`devIdxSym`], `destSym`.data.buffers[`devIdxSym`], `numElemsSym`)
          execCode.add stepCode
      
      of eskSub:
        # Generate all 8 combinations
        if srcAIsTemp and srcBIsTemp and destIsTemp:
          let stepCode = quote do:
            kernelSub(syclQueues[`devIdxSym`], `tempBuffersSym`[`devIdxSym`][`srcATempId`],
                     `tempBuffersSym`[`devIdxSym`][`srcBTempId`], `tempBuffersSym`[`devIdxSym`][`destTempId`], `numElemsSym`)
          execCode.add stepCode
        elif srcAIsTemp and srcBIsTemp and not destIsTemp:
          let stepCode = quote do:
            kernelSub(syclQueues[`devIdxSym`], `tempBuffersSym`[`devIdxSym`][`srcATempId`],
                     `tempBuffersSym`[`devIdxSym`][`srcBTempId`], `destSym`.data.buffers[`devIdxSym`], `numElemsSym`)
          execCode.add stepCode
        elif srcAIsTemp and not srcBIsTemp and destIsTemp:
          let stepCode = quote do:
            kernelSub(syclQueues[`devIdxSym`], `tempBuffersSym`[`devIdxSym`][`srcATempId`],
                     `srcBSym`.data.buffers[`devIdxSym`], `tempBuffersSym`[`devIdxSym`][`destTempId`], `numElemsSym`)
          execCode.add stepCode
        elif srcAIsTemp and not srcBIsTemp and not destIsTemp:
          let stepCode = quote do:
            kernelSub(syclQueues[`devIdxSym`], `tempBuffersSym`[`devIdxSym`][`srcATempId`],
                     `srcBSym`.data.buffers[`devIdxSym`], `destSym`.data.buffers[`devIdxSym`], `numElemsSym`)
          execCode.add stepCode
        elif not srcAIsTemp and srcBIsTemp and destIsTemp:
          let stepCode = quote do:
            kernelSub(syclQueues[`devIdxSym`], `srcASym`.data.buffers[`devIdxSym`],
                     `tempBuffersSym`[`devIdxSym`][`srcBTempId`], `tempBuffersSym`[`devIdxSym`][`destTempId`], `numElemsSym`)
          execCode.add stepCode
        elif not srcAIsTemp and srcBIsTemp and not destIsTemp:
          let stepCode = quote do:
            kernelSub(syclQueues[`devIdxSym`], `srcASym`.data.buffers[`devIdxSym`],
                     `tempBuffersSym`[`devIdxSym`][`srcBTempId`], `destSym`.data.buffers[`devIdxSym`], `numElemsSym`)
          execCode.add stepCode
        elif not srcAIsTemp and not srcBIsTemp and destIsTemp:
          let stepCode = quote do:
            kernelSub(syclQueues[`devIdxSym`], `srcASym`.data.buffers[`devIdxSym`],
                     `srcBSym`.data.buffers[`devIdxSym`], `tempBuffersSym`[`devIdxSym`][`destTempId`], `numElemsSym`)
          execCode.add stepCode
        else:
          let stepCode = quote do:
            kernelSub(syclQueues[`devIdxSym`], `srcASym`.data.buffers[`devIdxSym`],
                     `srcBSym`.data.buffers[`devIdxSym`], `destSym`.data.buffers[`devIdxSym`], `numElemsSym`)
          execCode.add stepCode
      
      of eskMatMul:
        # For matmul, we need shapes - use the LHS view
        # Common case: views on both sides, result to view
        if not srcAIsTemp and not srcBIsTemp and not destIsTemp:
          let stepCode = quote do:
            block:
              let outShape = `lhsViewSym`.shape
              let rows = if outShape.len >= 1: outShape[0] else: 1
              let cols = if outShape.len >= 2: outShape[1] else: 1
              let inner = cols
              kernelMatMul(syclQueues[`devIdxSym`], `srcASym`.data.buffers[`devIdxSym`],
                          `srcBSym`.data.buffers[`devIdxSym`], `destSym`.data.buffers[`devIdxSym`],
                          `devSitesSym`, rows, cols, inner, `vwLit`, `numVGSym`)
          execCode.add stepCode
        elif not srcAIsTemp and not srcBIsTemp and destIsTemp:
          let stepCode = quote do:
            block:
              let outShape = `lhsViewSym`.shape
              let rows = if outShape.len >= 1: outShape[0] else: 1
              let cols = if outShape.len >= 2: outShape[1] else: 1
              let inner = cols
              kernelMatMul(syclQueues[`devIdxSym`], `srcASym`.data.buffers[`devIdxSym`],
                          `srcBSym`.data.buffers[`devIdxSym`], `tempBuffersSym`[`devIdxSym`][`destTempId`],
                          `devSitesSym`, rows, cols, inner, `vwLit`, `numVGSym`)
          execCode.add stepCode
        else:
          # For other temp combinations, use view shapes as fallback
          let stepCode = quote do:
            block:
              let outShape = `lhsViewSym`.shape
              let rows = if outShape.len >= 1: outShape[0] else: 1
              let cols = if outShape.len >= 2: outShape[1] else: 1
              let inner = cols
              # Note: This is a fallback - may need temp buffer handling
              kernelMatMul(syclQueues[`devIdxSym`], `srcASym`.data.buffers[`devIdxSym`],
                          `srcBSym`.data.buffers[`devIdxSym`], `destSym`.data.buffers[`devIdxSym`],
                          `devSitesSym`, rows, cols, inner, `vwLit`, `numVGSym`)
          execCode.add stepCode
      
      of eskMatVec:
        if not srcAIsTemp and not srcBIsTemp and not destIsTemp:
          let stepCode = quote do:
            block:
              let outShape = `lhsViewSym`.shape
              let rows = if outShape.len >= 1: outShape[0] else: 1
              let cols = rows
              kernelMatVec(syclQueues[`devIdxSym`], `srcASym`.data.buffers[`devIdxSym`],
                          `srcBSym`.data.buffers[`devIdxSym`], `destSym`.data.buffers[`devIdxSym`],
                          `devSitesSym`, rows, cols, `vwLit`, `numVGSym`)
          execCode.add stepCode
        else:
          let stepCode = quote do:
            block:
              let outShape = `lhsViewSym`.shape
              let rows = if outShape.len >= 1: outShape[0] else: 1
              let cols = rows
              kernelMatVec(syclQueues[`devIdxSym`], `srcASym`.data.buffers[`devIdxSym`],
                          `srcBSym`.data.buffers[`devIdxSym`], `destSym`.data.buffers[`devIdxSym`],
                          `devSitesSym`, rows, cols, `vwLit`, `numVGSym`)
          execCode.add stepCode
      
      of eskScalarMul:
        if srcAIsTemp and destIsTemp:
          let stepCode = quote do:
            kernelScalarMul(syclQueues[`devIdxSym`], `tempBuffersSym`[`devIdxSym`][`srcATempId`],
                           `scalarStepLit`, `tempBuffersSym`[`devIdxSym`][`destTempId`], `numElemsSym`)
          execCode.add stepCode
        elif srcAIsTemp and not destIsTemp:
          let stepCode = quote do:
            kernelScalarMul(syclQueues[`devIdxSym`], `tempBuffersSym`[`devIdxSym`][`srcATempId`],
                           `scalarStepLit`, `destSym`.data.buffers[`devIdxSym`], `numElemsSym`)
          execCode.add stepCode
        elif not srcAIsTemp and destIsTemp:
          let stepCode = quote do:
            kernelScalarMul(syclQueues[`devIdxSym`], `srcASym`.data.buffers[`devIdxSym`],
                           `scalarStepLit`, `tempBuffersSym`[`devIdxSym`][`destTempId`], `numElemsSym`)
          execCode.add stepCode
        else:
          let stepCode = quote do:
            kernelScalarMul(syclQueues[`devIdxSym`], `srcASym`.data.buffers[`devIdxSym`],
                           `scalarStepLit`, `destSym`.data.buffers[`devIdxSym`], `numElemsSym`)
          execCode.add stepCode
      
      of eskScalarAdd:
        if srcAIsTemp and destIsTemp:
          let stepCode = quote do:
            kernelScalarAdd(syclQueues[`devIdxSym`], `tempBuffersSym`[`devIdxSym`][`srcATempId`],
                           `scalarStepLit`, `tempBuffersSym`[`devIdxSym`][`destTempId`], `numElemsSym`)
          execCode.add stepCode
        elif srcAIsTemp and not destIsTemp:
          let stepCode = quote do:
            kernelScalarAdd(syclQueues[`devIdxSym`], `tempBuffersSym`[`devIdxSym`][`srcATempId`],
                           `scalarStepLit`, `destSym`.data.buffers[`devIdxSym`], `numElemsSym`)
          execCode.add stepCode
        elif not srcAIsTemp and destIsTemp:
          let stepCode = quote do:
            kernelScalarAdd(syclQueues[`devIdxSym`], `srcASym`.data.buffers[`devIdxSym`],
                           `scalarStepLit`, `tempBuffersSym`[`devIdxSym`][`destTempId`], `numElemsSym`)
          execCode.add stepCode
        else:
          let stepCode = quote do:
            kernelScalarAdd(syclQueues[`devIdxSym`], `srcASym`.data.buffers[`devIdxSym`],
                           `scalarStepLit`, `destSym`.data.buffers[`devIdxSym`], `numElemsSym`)
          execCode.add stepCode
    
    result = quote do:
      block:
        let numDevices = syclQueues.len
        let sitesPerDev = `lhsViewSym`.data.sitesPerDevice
        let elemsPerSite = `lhsViewSym`.data.elementsPerSite
        
        # Allocate temp buffers per device
        var `tempBuffersSym`: seq[seq[SyclBuffer]]
        `tempBuffersSym`.setLen(numDevices)
        
        for `devIdxSym` in 0..<numDevices:
          let `devSitesSym` = sitesPerDev[`devIdxSym`]
          if `devSitesSym` > 0:
            let `numVGSym` = (`devSitesSym` + `vwLit` - 1) div `vwLit`
            let `numElemsSym` = `numVGSym` * elemsPerSite * `vwLit`
            let bufSize = `numElemsSym` * `lhsViewSym`.data.elementSize
            
            `tempBuffersSym`[`devIdxSym`].setLen(`numTempsLit`)
            for t in 0..<`numTempsLit`:
              `tempBuffersSym`[`devIdxSym`][t] = allocate(syclQueues[`devIdxSym`], bufSize)
        
        # Execute kernel steps
        for `devIdxSym` in 0..<numDevices:
          let `devSitesSym` = sitesPerDev[`devIdxSym`]
          if `devSitesSym` > 0:
            let `numVGSym` = (`devSitesSym` + `vwLit` - 1) div `vwLit`
            let `numElemsSym` = `numVGSym` * elemsPerSite * `vwLit`
            `execCode`
        
        # Synchronize
        for `devIdxSym` in 0..<numDevices:
          discard finish(syclQueues[`devIdxSym`])
        
        # Free temp buffers
        for `devIdxSym` in 0..<numDevices:
          for t in 0..<`numTempsLit`:
            if `tempBuffersSym`[`devIdxSym`].len > t:
              deallocate(syclQueues[`devIdxSym`], `tempBuffersSym`[`devIdxSym`][t])

macro each*(x: ForLoopStmt): untyped =
  ## Transform a for loop over TensorFieldView into a SYCL kernel dispatch.
  
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

# Export types to match OpenCL interface
export SyclQueue, SyclBuffer
