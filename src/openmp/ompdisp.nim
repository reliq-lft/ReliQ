#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/openmp/ompdisp.nim
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

## OpenMP Dispatch Macro for TensorFieldView
##
## This module provides the `each` macro for OpenMP-parallelized operations on
## TensorFieldView objects. The macro transforms loop bodies from the symbolic
## view[n] = viewA[n] + viewB[n] syntax into direct memory access code that
## iterates over all elements at each site.
##
## Unlike SYCL/OpenCL backends that generate kernel code strings, this generates
## Nim code that directly accesses host memory with OpenMP parallelization.
##
## Supports: float32, float64, int32, int64
## Operations: vector/matrix addition, subtraction, scalar multiply/add, matrix multiply

import std/macros
import std/tables
import std/strutils

import ompbase

export ompbase

# Compile-time configuration
const VectorWidth* {.intdefine.} = 8

#[ ============================================================================
   Expression Tree Types
   ============================================================================ ]#

type
  ExprKind* = enum
    ekSiteProxy     ## View access: view[n]
    ekMatMul        ## Matrix multiplication: A * B
    ekMatVec        ## Matrix-vector multiplication: M * v
    ekMatAdd        ## Matrix/vector addition: A + B or A - B
    ekScalarMul     ## Scalar multiply: s * A
    ekScalarAdd     ## Scalar add: A + s
    ekLiteral       ## Literal value
    ekUnknown       ## Unknown expression

  ExprInfo* = object
    kind*: ExprKind
    viewName*: string        ## For ekSiteProxy
    viewNode*: NimNode       ## Reference to the view symbol
    left*, right*: ref ExprInfo  ## For binary ops
    scalar*: NimNode         ## For scalar operations
    isSubtract*: bool        ## For ekMatAdd: true if subtraction

#[ ============================================================================
   View Info Extraction
   ============================================================================ ]#

type
  ViewInfo* = object
    name*: NimNode
    nameStr*: string
    isRead*: bool
    isWrite*: bool

proc extractViewSym(n: NimNode): NimNode =
  ## Extract the view symbol from a potentially wrapped node
  case n.kind
  of nnkSym: return n
  of nnkHiddenDeref: return extractViewSym(n[0])
  of nnkCall:
    if n.len >= 2:
      return extractViewSym(n[1])
  else: discard
  return nil

proc gatherViewInfo(body: NimNode): seq[ViewInfo] =
  ## Analyze loop body to find views being accessed
  var viewTable: Table[string, ViewInfo]
  
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
              viewTable[name] = ViewInfo(name: viewSym, nameStr: name)
            viewTable[name].isWrite = true
          for i in 2..<n.len:
            analyzeNode(n[i])
        elif opName == "[]":
          let viewSym = extractViewSym(n[1])
          if viewSym != nil:
            let name = viewSym.strVal
            if name notin viewTable:
              viewTable[name] = ViewInfo(name: viewSym, nameStr: name)
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
  
  for name, info in viewTable:
    result.add info

#[ ============================================================================
   Expression Tree Analysis
   ============================================================================ ]#

proc analyzeExpr*(n: NimNode): ExprInfo =
  ## Analyze an expression AST and build an ExprInfo tree
  ## Identifies view accesses, operations, scalars, etc.
  
  case n.kind
  of nnkCall:
    if n[0].kind == nnkSym:
      let opName = n[0].strVal
      
      # View access: view[site]
      if opName == "[]" and n.len >= 3:
        let viewSym = extractViewSym(n[1])
        if viewSym != nil:
          result = ExprInfo(kind: ekSiteProxy, viewName: viewSym.strVal, viewNode: viewSym)
          return
      
      # Matrix multiplication: A * B
      if opName == "*" and n.len >= 3:
        let leftExpr = analyzeExpr(n[1])
        let rightExpr = analyzeExpr(n[2])
        # Both sides are view accesses -> matrix multiply
        if leftExpr.kind in {ekSiteProxy, ekMatMul, ekMatAdd, ekScalarMul} and
           rightExpr.kind in {ekSiteProxy, ekMatMul, ekMatAdd, ekScalarMul}:
          result = ExprInfo(kind: ekMatMul)
          new(result.left); result.left[] = leftExpr
          new(result.right); result.right[] = rightExpr
          return
        # Scalar * tensor
        elif leftExpr.kind == ekLiteral:
          result = ExprInfo(kind: ekScalarMul, scalar: leftExpr.scalar)
          new(result.right); result.right[] = rightExpr
          return
        elif rightExpr.kind == ekLiteral:
          result = ExprInfo(kind: ekScalarMul, scalar: rightExpr.scalar)
          new(result.left); result.left[] = leftExpr
          return
      
      # Matrix addition: A + B
      if opName == "+" and n.len >= 3:
        let leftExpr = analyzeExpr(n[1])
        let rightExpr = analyzeExpr(n[2])
        # Both sides are tensor expressions
        if leftExpr.kind in {ekSiteProxy, ekMatMul, ekMatAdd, ekScalarMul} and
           rightExpr.kind in {ekSiteProxy, ekMatMul, ekMatAdd, ekScalarMul}:
          result = ExprInfo(kind: ekMatAdd, isSubtract: false)
          new(result.left); result.left[] = leftExpr
          new(result.right); result.right[] = rightExpr
          return
        # Tensor + scalar
        elif leftExpr.kind != ekLiteral and rightExpr.kind == ekLiteral:
          result = ExprInfo(kind: ekScalarAdd, scalar: rightExpr.scalar)
          new(result.left); result.left[] = leftExpr
          return
        elif leftExpr.kind == ekLiteral:
          result = ExprInfo(kind: ekScalarAdd, scalar: leftExpr.scalar)
          new(result.right); result.right[] = rightExpr
          return
      
      # Matrix subtraction: A - B
      if opName == "-" and n.len >= 3:
        let leftExpr = analyzeExpr(n[1])
        let rightExpr = analyzeExpr(n[2])
        if leftExpr.kind in {ekSiteProxy, ekMatMul, ekMatAdd, ekScalarMul} and
           rightExpr.kind in {ekSiteProxy, ekMatMul, ekMatAdd, ekScalarMul}:
          result = ExprInfo(kind: ekMatAdd, isSubtract: true)
          new(result.left); result.left[] = leftExpr
          new(result.right); result.right[] = rightExpr
          return
    
    # Unknown call
    result = ExprInfo(kind: ekUnknown)
  
  of nnkIntLit, nnkInt8Lit, nnkInt16Lit, nnkInt32Lit, nnkInt64Lit,
     nnkFloatLit, nnkFloat32Lit, nnkFloat64Lit:
    result = ExprInfo(kind: ekLiteral, scalar: n)
  
  of nnkSym:
    # Could be a scalar variable
    result = ExprInfo(kind: ekLiteral, scalar: n)
  
  of nnkHiddenDeref:
    result = analyzeExpr(n[0])
  
  else:
    result = ExprInfo(kind: ekUnknown)

#[ ============================================================================
   Code Generation: Transform Expression Tree to Direct Memory Access
   ============================================================================ ]#

proc generateElementCode(expr: ExprInfo, siteVar, elemVar: NimNode, 
                         viewDataTable: Table[string, NimNode],
                         viewElemsTable: Table[string, NimNode]): NimNode =
  ## Generate Nim AST for computing a single element [elem] of the result
  ## at site [site]. This is element-wise for add/sub, but for matmul
  ## we need row/col decomposition.
  
  case expr.kind
  of ekSiteProxy:
    # Direct element access: viewData[site * elems + elem]
    let viewData = viewDataTable[expr.viewName]
    let viewElems = viewElemsTable[expr.viewName]
    result = quote do:
      `viewData`[`siteVar` * `viewElems` + `elemVar`]
  
  of ekMatAdd:
    # Element-wise add/subtract
    let leftCode = generateElementCode(expr.left[], siteVar, elemVar, viewDataTable, viewElemsTable)
    let rightCode = generateElementCode(expr.right[], siteVar, elemVar, viewDataTable, viewElemsTable)
    if expr.isSubtract:
      result = quote do:
        `leftCode` - `rightCode`
    else:
      result = quote do:
        `leftCode` + `rightCode`
  
  of ekScalarMul:
    let tensorCode = if expr.left != nil: 
                       generateElementCode(expr.left[], siteVar, elemVar, viewDataTable, viewElemsTable)
                     else:
                       generateElementCode(expr.right[], siteVar, elemVar, viewDataTable, viewElemsTable)
    let scalarNode = expr.scalar
    result = quote do:
      `scalarNode` * `tensorCode`
  
  of ekScalarAdd:
    let tensorCode = if expr.left != nil:
                       generateElementCode(expr.left[], siteVar, elemVar, viewDataTable, viewElemsTable)
                     else:
                       generateElementCode(expr.right[], siteVar, elemVar, viewDataTable, viewElemsTable)
    let scalarNode = expr.scalar
    result = quote do:
      `tensorCode` + `scalarNode`
  
  of ekLiteral:
    result = expr.scalar
  
  of ekMatMul, ekMatVec:
    # Matrix multiply needs special handling - this shouldn't be called directly
    # for matmul; we handle it at a higher level with row/col loops
    result = newLit(0.0)
  
  of ekUnknown:
    result = newLit(0.0)

proc hasMatMul(expr: ExprInfo): bool =
  ## Check if expression contains any matrix multiplication
  case expr.kind
  of ekMatMul, ekMatVec: return true
  of ekMatAdd, ekScalarMul, ekScalarAdd:
    if expr.left != nil and hasMatMul(expr.left[]): return true
    if expr.right != nil and hasMatMul(expr.right[]): return true
    return false
  else:
    return false

proc generateMatMulElement(expr: ExprInfo, siteVar, rowVar, colVar, innerDimVar: NimNode,
                           viewDataTable: Table[string, NimNode],
                           shapeTable: Table[string, NimNode]): NimNode =
  ## Generate code for a single element of matrix multiply result
  ## C[row, col] = sum_k(A[row, k] * B[k, col])
  
  case expr.kind
  of ekMatMul:
    # Get left and right matrix accesses
    let leftName = expr.left.viewName
    let rightName = expr.right.viewName
    let leftData = viewDataTable[leftName]
    let rightData = viewDataTable[rightName]
    let leftShape = shapeTable[leftName]
    let rightShape = shapeTable[rightName]
    
    # A[row, k] = leftData[site * leftElems + row * leftCols + k]
    # B[k, col] = rightData[site * rightElems + k * rightCols + col]
    result = quote do:
      block:
        var sum = typeof(`leftData`[0])(0)
        let leftCols = `leftShape`[1]
        let rightCols = `rightShape`[1]
        let leftElems = `leftShape`[0] * `leftShape`[1]
        let rightElems = `rightShape`[0] * `rightShape`[1]
        for k in 0..<leftCols:
          let leftIdx = `siteVar` * leftElems + `rowVar` * leftCols + k
          let rightIdx = `siteVar` * rightElems + k * rightCols + `colVar`
          sum += `leftData`[leftIdx] * `rightData`[rightIdx]
        sum
  
  of ekSiteProxy:
    # Direct element access for row,col
    let viewData = viewDataTable[expr.viewName]
    let viewShape = shapeTable[expr.viewName]
    result = quote do:
      block:
        let cols = `viewShape`[1]
        let elems = `viewShape`[0] * `viewShape`[1]
        `viewData`[`siteVar` * elems + `rowVar` * cols + `colVar`]
  
  of ekMatAdd:
    let leftCode = generateMatMulElement(expr.left[], siteVar, rowVar, colVar, innerDimVar, viewDataTable, shapeTable)
    let rightCode = generateMatMulElement(expr.right[], siteVar, rowVar, colVar, innerDimVar, viewDataTable, shapeTable)
    if expr.isSubtract:
      result = quote do:
        `leftCode` - `rightCode`
    else:
      result = quote do:
        `leftCode` + `rightCode`
  
  of ekScalarMul:
    let tensorCode = if expr.left != nil:
                       generateMatMulElement(expr.left[], siteVar, rowVar, colVar, innerDimVar, viewDataTable, shapeTable)
                     else:
                       generateMatMulElement(expr.right[], siteVar, rowVar, colVar, innerDimVar, viewDataTable, shapeTable)
    let scalarNode = expr.scalar
    result = quote do:
      `scalarNode` * `tensorCode`
  
  else:
    result = newLit(0.0)

#[ ============================================================================
   Statement Analysis
   ============================================================================ ]#

proc extractWriteInfo(n: NimNode): tuple[viewSym: NimNode, siteExpr: NimNode, rhsExpr: NimNode] =
  ## Extract the write view, site index, and RHS expression from an assignment
  ## Handles: view[site] = expr or `[]=`(view, site, expr)
  
  case n.kind
  of nnkCall:
    if n[0].kind == nnkSym and n[0].strVal == "[]=":
      # `[]=`(view, site, value)
      result.viewSym = extractViewSym(n[1])
      result.siteExpr = n[2]
      result.rhsExpr = n[3]
  of nnkAsgn:
    # view[site] = expr
    if n[0].kind == nnkBracketExpr:
      result.viewSym = extractViewSym(n[0][0])
      result.siteExpr = n[0][1]
      result.rhsExpr = n[1]
    elif n[0].kind == nnkCall and n[0][0].kind == nnkSym and n[0][0].strVal == "[]":
      result.viewSym = extractViewSym(n[0][1])
      result.siteExpr = n[0][2]
      result.rhsExpr = n[1]
  else:
    discard

#[ ============================================================================
   Echo Statement Detection
   ============================================================================ ]#

proc hasEchoStatement(n: NimNode): bool =
  ## Check if body contains echo/debugEcho (requires serial execution with CPU fallback)
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
  else:
    for child in n:
      if hasEchoStatement(child):
        return true
  return false

#[ ============================================================================
   Main Each Macro
   ============================================================================ ]#

macro each*(forLoop: ForLoopStmt): untyped =
  ## OpenMP parallel each loop for TensorFieldView
  ##
  ## This macro provides the `each` iterator for OpenMP-parallelized loops.
  ## Unlike the OpenCL backend which transforms expressions into kernel code,
  ## this version executes the operators directly - the operators in
  ## tensorview.nim and sitetensor.nim handle the actual memory operations.
  ##
  ## Usage:
  ##   for n in each 0..<view.numSites():
  ##     viewC[n] = viewA[n] + viewB[n]
  
  # Extract loop components
  let loopVar = forLoop[0]
  let loopRangeNode = forLoop[1][1]  # Skip 'each' wrapper
  let body = forLoop[2]
  
  # Check if we need serial execution (echo statements)
  let needsSerial = hasEchoStatement(body)
  
  if needsSerial:
    # Serial fallback for debugging with echo
    result = quote do:
      block:
        let rangeVal = `loopRangeNode`
        for `loopVar` in rangeVal:
          `body`
    return result
  
  # Execute the body directly - operators handle memory access
  # TODO: Add OpenMP parallelization with proper pragma placement
  if loopRangeNode.kind == nnkInfix and loopRangeNode[0].strVal == "..<":
    let startExpr = loopRangeNode[1]
    let endExpr = loopRangeNode[2]
    result = quote do:
      block:
        let startVal = `startExpr`
        let endVal = `endExpr`
        for `loopVar` in startVal..<endVal:
          `body`
  else:
    result = quote do:
      block:
        let rangeVal = `loopRangeNode`
        for `loopVar` in rangeVal:
          `body`

when isMainModule:
  import ../tensor/sitetensor
  
  initOpenMP()
  echo "OpenMP dispatch module loaded"
  echo "Max threads: ", getNumThreads()
