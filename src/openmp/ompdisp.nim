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

## SIMD-Vectorized Each Macro for TensorFieldView (OpenMP backend)
##
## This module provides the `each` iterator for SIMD-vectorized parallel loops
## on TensorFieldView objects. Uses OpenMP for thread parallelism and the
## SIMD infrastructure (SimdVecDyn, AoSoA layout) from simd/ for vectorization.
##
## The macro analyzes the loop body at compile time to detect operation patterns
## (copy, add, subtract, scalar multiply, matrix multiply, etc.) and generates
## SIMD-vectorized code that:
## - Iterates over vector groups (outer loop, OpenMP parallelized)
## - Loads contiguous AoSoA lanes as SimdVecDyn vectors
## - Performs arithmetic on full SIMD vectors
## - Stores results back to AoSoA layout
##
## Usage:
##   for n in each 0..<view.numSites():
##     viewC[n] = viewA[n] + viewB[n]   # Vectorized via SIMD load/add/store
##
## For LocalTensorField operations, see `all` in omplocal.nim.

import std/[macros, tables, strutils]

import ompbase
export ompbase

{.passC: "-fopenmp".}
{.passL: "-fopenmp".}

{.emit: """
#include <omp.h>
""".}

import ./ompwrap
import ../simd/simdtypes

#[ ============================================================================
   Echo Statement Detection
   ============================================================================ ]#

proc hasEchoStatement(n: NimNode): bool =
  ## Check if body contains echo/debugEcho (requires serial execution)
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
   Compile-Time Expression Analysis (shared with OpenCL/SYCL)
   ============================================================================ ]#

type
  OmpExprKind = enum
    oekSiteProxy,    ## view[n]
    oekMatMul,       ## A * B (matrix multiply)
    oekMatVec,       ## M * v (matrix-vector)
    oekMatAdd,       ## A + B or A - B
    oekScalarMul,    ## scalar * A
    oekScalarAdd,    ## scalar + A
    oekLiteral,      ## numeric literal
    oekUnknown

  OmpExprInfo = object
    kind: OmpExprKind
    viewName: string
    isSubtract: bool
    scalar: float64
    left, right: ref OmpExprInfo
    isNeighborAccess: bool  ## true if index comes from stencil.neighbor()

  OmpViewInfo = object
    name: NimNode
    nameStr: string
    isRead: bool
    isWrite: bool

  OmpDispatchKind = enum
    odkCopy,        ## dst[n] = src[n]
    odkAdd,         ## dst[n] = a[n] + b[n]
    odkSub,         ## dst[n] = a[n] - b[n]
    odkScalarMul,   ## dst[n] = s * a[n]
    odkScalarAdd,   ## dst[n] = a[n] + s
    odkMatMul,      ## dst[n] = a[n] * b[n] (matrix multiply)
    odkMatVec,      ## dst[n] = m[n] * v[n] (matrix-vector)
    odkUnknown      ## Fall back to per-site scalar loop

  OmpDispatch = object
    kind: OmpDispatchKind
    lhsView: string
    rhsViews: seq[string]
    scalar: float64
    hasStencil: bool

proc isViewAccess(n: NimNode, viewNames: seq[string]): bool =
  ## Check if node is a view[n] access
  if n.kind == nnkCall and n.len >= 2:
    if n[0].kind == nnkSym and n[0].strVal == "[]":
      if n[1].kind == nnkSym and n[1].strVal in viewNames:
        return true
  false

proc findViewName(n: NimNode): string =
  ## Extract view name from a site proxy expression
  case n.kind
  of nnkCall:
    if n[0].kind == nnkSym:
      if n[0].strVal == "[]" and n.len >= 2:
        if n[1].kind == nnkSym:
          return n[1].strVal
  of nnkSym:
    return n.strVal
  of nnkHiddenStdConv, nnkHiddenDeref, nnkConv:
    if n.len > 0:
      return findViewName(n[^1])
  else:
    discard
  ""

proc analyzeExpr(n: NimNode, viewNames: seq[string]): OmpExprInfo =
  ## Recursively analyze an expression to determine its operation kind
  case n.kind
  of nnkCall:
    let fnName = if n[0].kind == nnkSym: n[0].strVal else: ""
    
    if fnName == "[]" and n.len >= 2:
      # View access: view[n]
      if n[1].kind == nnkSym and n[1].strVal in viewNames:
        result.kind = oekSiteProxy
        result.viewName = n[1].strVal
        # Check if index is a stencil neighbor variable
        if n.len >= 3 and n[2].kind == nnkSym:
          let idxName = n[2].strVal
          if "nbr" in idxName.toLowerAscii or "neighbor" in idxName.toLowerAscii:
            result.isNeighborAccess = true
        return
    
    if fnName == "+" and n.len == 3:
      let leftExpr = analyzeExpr(n[1], viewNames)
      let rightExpr = analyzeExpr(n[2], viewNames)
      if leftExpr.kind == oekLiteral:
        result.kind = oekScalarAdd
        result.scalar = leftExpr.scalar
        result.left = new OmpExprInfo
        result.left[] = rightExpr
      elif rightExpr.kind == oekLiteral:
        result.kind = oekScalarAdd
        result.scalar = rightExpr.scalar
        result.left = new OmpExprInfo
        result.left[] = leftExpr
      else:
        result.kind = oekMatAdd
        result.left = new OmpExprInfo
        result.left[] = leftExpr
        result.right = new OmpExprInfo
        result.right[] = rightExpr
      return
    
    if fnName == "-" and n.len == 3:
      let leftExpr = analyzeExpr(n[1], viewNames)
      let rightExpr = analyzeExpr(n[2], viewNames)
      result.kind = oekMatAdd
      result.isSubtract = true
      result.left = new OmpExprInfo
      result.left[] = leftExpr
      result.right = new OmpExprInfo
      result.right[] = rightExpr
      return
    
    if fnName == "*" and n.len == 3:
      let leftExpr = analyzeExpr(n[1], viewNames)
      let rightExpr = analyzeExpr(n[2], viewNames)
      if leftExpr.kind == oekLiteral:
        result.kind = oekScalarMul
        result.scalar = leftExpr.scalar
        result.left = new OmpExprInfo
        result.left[] = rightExpr
      elif rightExpr.kind == oekLiteral:
        result.kind = oekScalarMul
        result.scalar = rightExpr.scalar
        result.left = new OmpExprInfo
        result.left[] = leftExpr
      else:
        result.kind = oekMatMul
        result.left = new OmpExprInfo
        result.left[] = leftExpr
        result.right = new OmpExprInfo
        result.right[] = rightExpr
      return
  
  of nnkInfix:
    let op = if n[0].kind in {nnkSym, nnkIdent}: n[0].strVal else: ""
    if op in ["+", "-", "*"]:
      # Convert to call-style and re-analyze
      var callNode = newNimNode(nnkCall)
      callNode.add(n[0])
      callNode.add(n[1])
      callNode.add(n[2])
      return analyzeExpr(callNode, viewNames)
  
  of nnkFloatLit..nnkFloat64Lit:
    result.kind = oekLiteral
    result.scalar = n.floatVal
    return
  
  of nnkIntLit..nnkInt64Lit:
    result.kind = oekLiteral
    result.scalar = n.intVal.float64
    return
  
  of nnkSym:
    if n.strVal in viewNames:
      result.kind = oekSiteProxy
      result.viewName = n.strVal
    else:
      # Could be a scalar variable
      result.kind = oekLiteral
      result.scalar = 0.0  # Will be resolved at runtime
    return
  
  of nnkHiddenStdConv, nnkHiddenDeref, nnkConv:
    if n.len > 0:
      return analyzeExpr(n[^1], viewNames)
  
  else:
    discard
  
  result.kind = oekUnknown

proc gatherViewInfo(body: NimNode, loopVar: NimNode): tuple[views: seq[OmpViewInfo], hasStencil: bool] =
  ## Walk the typed AST to find all TensorFieldView accesses and their read/write roles
  var viewTable: Table[string, OmpViewInfo]
  var hasStencil = false
  
  proc analyzeNode(n: NimNode, isLHS: bool = false) =
    case n.kind
    of nnkCall:
      if n[0].kind == nnkSym:
        let fnName = n[0].strVal
        if fnName == "[]=" and n.len >= 3:
          # Write access: view[n] = ...
          if n[1].kind == nnkSym:
            let viewName = n[1].strVal
            if viewName notin viewTable:
              viewTable[viewName] = OmpViewInfo(name: n[1], nameStr: viewName)
            viewTable[viewName].isWrite = true
          # Analyze RHS
          for i in 3..<n.len:
            analyzeNode(n[i], false)
          return
        elif fnName == "[]" and n.len >= 2:
          # Read access: view[n]
          if n[1].kind == nnkSym:
            let viewName = n[1].strVal
            if viewName notin viewTable:
              viewTable[viewName] = OmpViewInfo(name: n[1], nameStr: viewName)
            viewTable[viewName].isRead = true
          return
        elif fnName == "neighbor":
          hasStencil = true
      for child in n:
        analyzeNode(child, isLHS)
    of nnkLetSection, nnkVarSection:
      for def in n:
        if def.kind == nnkIdentDefs and def.len >= 3:
          analyzeNode(def[2], false)
    of nnkAsgn:
      analyzeNode(n[0], true)
      analyzeNode(n[1], false)
    else:
      for child in n:
        analyzeNode(child, isLHS)
  
  analyzeNode(body)
  
  for _, v in viewTable:
    result.views.add v
  result.hasStencil = hasStencil

proc determineDispatch(expr: OmpExprInfo, lhsView: string): OmpDispatch =
  ## Determine the dispatch operation from the expression tree
  result.lhsView = lhsView
  
  case expr.kind
  of oekSiteProxy:
    result.kind = odkCopy
    result.rhsViews = @[expr.viewName]
    result.hasStencil = expr.isNeighborAccess
  
  of oekMatAdd:
    if expr.left != nil and expr.right != nil:
      if expr.left.kind == oekSiteProxy and expr.right.kind == oekSiteProxy:
        if expr.isSubtract:
          result.kind = odkSub
        else:
          result.kind = odkAdd
        result.rhsViews = @[expr.left.viewName, expr.right.viewName]
      else:
        result.kind = odkUnknown
    else:
      result.kind = odkUnknown
  
  of oekScalarMul:
    if expr.left != nil and expr.left.kind == oekSiteProxy:
      result.kind = odkScalarMul
      result.scalar = expr.scalar
      result.rhsViews = @[expr.left.viewName]
    else:
      result.kind = odkUnknown
  
  of oekScalarAdd:
    if expr.left != nil and expr.left.kind == oekSiteProxy:
      result.kind = odkScalarAdd
      result.scalar = expr.scalar
      result.rhsViews = @[expr.left.viewName]
    else:
      result.kind = odkUnknown
  
  of oekMatMul:
    if expr.left != nil and expr.right != nil and
       expr.left.kind == oekSiteProxy and expr.right.kind == oekSiteProxy:
      result.kind = odkMatMul
      result.rhsViews = @[expr.left.viewName, expr.right.viewName]
    else:
      result.kind = odkUnknown
  
  of oekMatVec:
    if expr.left != nil and expr.right != nil and
       expr.left.kind == oekSiteProxy and expr.right.kind == oekSiteProxy:
      result.kind = odkMatVec
      result.rhsViews = @[expr.left.viewName, expr.right.viewName]
    else:
      result.kind = odkUnknown
  
  else:
    result.kind = odkUnknown

#[ ============================================================================
   Internal Typed Each Macro - SIMD dispatch
   ============================================================================ ]#

macro eachImpl*(loopVar: untyped, lo: typed, hi: typed, body: typed): untyped =
  ## Internal typed macro that receives full type information.
  ## Analyzes the expression pattern and generates SIMD-vectorized code
  ## using loadSimdVectorDyn/storeSimdVectorDyn on AoSoA data.
  
  let loopVarSym = loopVar
  let (info, hasStencil) = gatherViewInfo(body, loopVarSym)
  
  if info.len == 0 or hasEchoStatement(body) or hasStencil:
    # Fall back to scalar per-site loop for:
    # - No views found
    # - Echo statements (need serial)
    # - Stencil access (neighbor indices aren't vectorizable without gather)
    # Note: must assign to loopVarSym (not create new local) because the typed
    # body's symbols are already bound to the injected global variable.
    let siteIdxSym = genSym(nskForVar, "siteIdx")
    let assignStmt = newNimNode(nnkAsgn).add(loopVarSym, siteIdxSym)
    let loopBody = newStmtList(assignStmt, body)
    let rangeExpr = newNimNode(nnkInfix).add(ident"..<", lo, hi)
    let forLoop = newNimNode(nnkForStmt).add(siteIdxSym, rangeExpr, loopBody)
    result = newNimNode(nnkBlockStmt).add(newEmptyNode(), newStmtList(forLoop))
    return
  
  # Gather view names
  var viewNames: seq[string]
  for v in info:
    viewNames.add v.nameStr
  
  # Find output (LHS) view
  var lhsViewSym: NimNode = nil
  var lhsViewName = ""
  for v in info:
    if v.isWrite:
      lhsViewSym = v.name
      lhsViewName = v.nameStr
      break
  
  if lhsViewSym == nil:
    # No write detected, fall back to scalar per-site loop
    let siteIdxSym = genSym(nskForVar, "siteIdx")
    let assignStmt = newNimNode(nnkAsgn).add(loopVarSym, siteIdxSym)
    let loopBody = newStmtList(assignStmt, body)
    let rangeExpr = newNimNode(nnkInfix).add(ident"..<", lo, hi)
    let forLoop = newNimNode(nnkForStmt).add(siteIdxSym, rangeExpr, loopBody)
    result = newNimNode(nnkBlockStmt).add(newEmptyNode(), newStmtList(forLoop))
    return
  
  # Extract the RHS expression from the body
  var stmt: NimNode
  if body.kind == nnkStmtList and body.len > 0:
    stmt = body[0]
  else:
    stmt = body
  
  # Analyze the RHS expression
  var rhsExpr: OmpExprInfo
  if stmt.kind == nnkCall and stmt.len >= 4 and stmt[0].kind == nnkSym and stmt[0].strVal == "[]=":
    rhsExpr = analyzeExpr(stmt[3], viewNames)
  else:
    rhsExpr = OmpExprInfo(kind: oekUnknown)
  
  let dispatch = determineDispatch(rhsExpr, lhsViewName)
  
  # Find RHS view symbols
  var rhsView1Sym, rhsView2Sym: NimNode = nil
  for v in info:
    if dispatch.rhsViews.len >= 1 and v.nameStr == dispatch.rhsViews[0]:
      rhsView1Sym = v.name
    if dispatch.rhsViews.len >= 2 and v.nameStr == dispatch.rhsViews[1]:
      rhsView2Sym = v.name
  
  # Generate SIMD-vectorized code based on dispatch kind
  # Helper template for scalar fallback (sequential per-site loop)
  # Must assign to loopVarSym because the typed body references the injected global.
  template scalarFallback() =
    let siteIdxSym = genSym(nskForVar, "siteIdx")
    let assignStmt = newNimNode(nnkAsgn).add(loopVarSym, siteIdxSym)
    let loopBodyStmt = newStmtList(assignStmt, body)
    let rangeExpr = newNimNode(nnkInfix).add(ident"..<", lo, hi)
    let forLoop = newNimNode(nnkForStmt).add(siteIdxSym, rangeExpr, loopBodyStmt)
    result = newNimNode(nnkBlockStmt).add(newEmptyNode(), newStmtList(forLoop))
  
  case dispatch.kind
  of odkCopy:
    if rhsView1Sym == nil:
      scalarFallback()
      return
    
    result = quote do:
      block:
        let dstData = `lhsViewSym`.aosoaDataPtr
        let srcData = `rhsView1Sym`.aosoaDataPtr
        let layout = `lhsViewSym`.data.simdLayout
        let elemsPerSite = `lhsViewSym`.data.tensorElementsPerSite
        let nInner = layout.nSitesInner
        let nOuter = layout.nSitesOuter
        type ElemT = typeof(dstData[0])
        proc simdLoop(chunkStart, chunkEnd: int64, ctx: pointer) {.cdecl.} =
          for outer in chunkStart..<chunkEnd:
            let outerIdx = int(outer)
            for e in 0..<elemsPerSite:
              var vec = loadSimdVectorDyn[ElemT](
                srcData, outerIdx, e, elemsPerSite, nInner)
              storeSimdVectorDyn(vec, dstData, outerIdx, e, elemsPerSite, nInner)
        ompParallelForChunked(0'i64, int64(nOuter), simdLoop, nil)
  
  of odkAdd:
    if rhsView1Sym == nil or rhsView2Sym == nil:
      scalarFallback()
      return
    
    result = quote do:
      block:
        let dstData = `lhsViewSym`.aosoaDataPtr
        let srcAData = `rhsView1Sym`.aosoaDataPtr
        let srcBData = `rhsView2Sym`.aosoaDataPtr
        let layout = `lhsViewSym`.data.simdLayout
        let elemsPerSite = `lhsViewSym`.data.tensorElementsPerSite
        let nInner = layout.nSitesInner
        let nOuter = layout.nSitesOuter
        type ElemT = typeof(dstData[0])
        proc simdLoop(chunkStart, chunkEnd: int64, ctx: pointer) {.cdecl.} =
          for outer in chunkStart..<chunkEnd:
            let outerIdx = int(outer)
            for e in 0..<elemsPerSite:
              var vecA = loadSimdVectorDyn[ElemT](
                srcAData, outerIdx, e, elemsPerSite, nInner)
              var vecB = loadSimdVectorDyn[ElemT](
                srcBData, outerIdx, e, elemsPerSite, nInner)
              var vecR = vecA + vecB
              storeSimdVectorDyn(vecR, dstData, outerIdx, e, elemsPerSite, nInner)
        ompParallelForChunked(0'i64, int64(nOuter), simdLoop, nil)
  
  of odkSub:
    if rhsView1Sym == nil or rhsView2Sym == nil:
      scalarFallback()
      return
    
    result = quote do:
      block:
        let dstData = `lhsViewSym`.aosoaDataPtr
        let srcAData = `rhsView1Sym`.aosoaDataPtr
        let srcBData = `rhsView2Sym`.aosoaDataPtr
        let layout = `lhsViewSym`.data.simdLayout
        let elemsPerSite = `lhsViewSym`.data.tensorElementsPerSite
        let nInner = layout.nSitesInner
        let nOuter = layout.nSitesOuter
        type ElemT = typeof(dstData[0])
        proc simdLoop(chunkStart, chunkEnd: int64, ctx: pointer) {.cdecl.} =
          for outer in chunkStart..<chunkEnd:
            let outerIdx = int(outer)
            for e in 0..<elemsPerSite:
              var vecA = loadSimdVectorDyn[ElemT](
                srcAData, outerIdx, e, elemsPerSite, nInner)
              var vecB = loadSimdVectorDyn[ElemT](
                srcBData, outerIdx, e, elemsPerSite, nInner)
              var vecR = vecA - vecB
              storeSimdVectorDyn(vecR, dstData, outerIdx, e, elemsPerSite, nInner)
        ompParallelForChunked(0'i64, int64(nOuter), simdLoop, nil)
  
  of odkScalarMul:
    if rhsView1Sym == nil:
      scalarFallback()
      return
    
    let scalarLit = newLit(dispatch.scalar)
    result = quote do:
      block:
        let dstData = `lhsViewSym`.aosoaDataPtr
        let srcData = `rhsView1Sym`.aosoaDataPtr
        let layout = `lhsViewSym`.data.simdLayout
        let elemsPerSite = `lhsViewSym`.data.tensorElementsPerSite
        let nInner = layout.nSitesInner
        let nOuter = layout.nSitesOuter
        type ElemT = typeof(dstData[0])
        proc simdLoop(chunkStart, chunkEnd: int64, ctx: pointer) {.cdecl.} =
          for outer in chunkStart..<chunkEnd:
            let outerIdx = int(outer)
            for e in 0..<elemsPerSite:
              var vec = loadSimdVectorDyn[ElemT](
                srcData, outerIdx, e, elemsPerSite, nInner)
              var vecR = ElemT(`scalarLit`) * vec
              storeSimdVectorDyn(vecR, dstData, outerIdx, e, elemsPerSite, nInner)
        ompParallelForChunked(0'i64, int64(nOuter), simdLoop, nil)
  
  of odkScalarAdd:
    if rhsView1Sym == nil:
      scalarFallback()
      return
    
    let scalarLit = newLit(dispatch.scalar)
    result = quote do:
      block:
        let dstData = `lhsViewSym`.aosoaDataPtr
        let srcData = `rhsView1Sym`.aosoaDataPtr
        let layout = `lhsViewSym`.data.simdLayout
        let elemsPerSite = `lhsViewSym`.data.tensorElementsPerSite
        let nInner = layout.nSitesInner
        let nOuter = layout.nSitesOuter
        type ElemT = typeof(dstData[0])
        proc simdLoop(chunkStart, chunkEnd: int64, ctx: pointer) {.cdecl.} =
          for outer in chunkStart..<chunkEnd:
            let outerIdx = int(outer)
            for e in 0..<elemsPerSite:
              var vec = loadSimdVectorDyn[ElemT](
                srcData, outerIdx, e, elemsPerSite, nInner)
              var vecR = vec + ElemT(`scalarLit`)
              storeSimdVectorDyn(vecR, dstData, outerIdx, e, elemsPerSite, nInner)
        ompParallelForChunked(0'i64, int64(nOuter), simdLoop, nil)
  
  of odkMatMul:
    if rhsView1Sym == nil or rhsView2Sym == nil:
      scalarFallback()
      return
    
    # Matrix multiply: C[i,j] = sum_k A[i,k] * B[k,j]
    # SIMD vectorization is across sites (lanes), not across matrix elements
    result = quote do:
      block:
        let dstData = `lhsViewSym`.aosoaDataPtr
        let srcAData = `rhsView1Sym`.aosoaDataPtr
        let srcBData = `rhsView2Sym`.aosoaDataPtr
        let layout = `lhsViewSym`.data.simdLayout
        let nInner = layout.nSitesInner
        let nOuter = layout.nSitesOuter
        type ElemT = typeof(dstData[0])
        
        let outShape = `lhsViewSym`.shape
        let rows = outShape[0]
        let cols = if outShape.len > 1: outShape[1] else: 1
        let innerDim = if `rhsView1Sym`.shape.len > 1: `rhsView1Sym`.shape[1] else: 1
        let elemsA = `rhsView1Sym`.data.tensorElementsPerSite
        let elemsB = `rhsView2Sym`.data.tensorElementsPerSite
        let elemsC = `lhsViewSym`.data.tensorElementsPerSite
        proc simdLoop(chunkStart, chunkEnd: int64, ctx: pointer) {.cdecl.} =
          for outer in chunkStart..<chunkEnd:
            let outerIdx = int(outer)
            for i in 0..<rows:
              for j in 0..<cols:
                var acc = newSimdVecDyn[ElemT](nInner)
                for k in 0..<innerDim:
                  let aElemIdx = i * innerDim + k
                  let bElemIdx = k * cols + j
                  var vecA = loadSimdVectorDyn[ElemT](
                    srcAData, outerIdx, aElemIdx, elemsA, nInner)
                  var vecB = loadSimdVectorDyn[ElemT](
                    srcBData, outerIdx, bElemIdx, elemsB, nInner)
                  acc = acc + vecA * vecB
                let cElemIdx = i * cols + j
                storeSimdVectorDyn(acc, dstData, outerIdx, cElemIdx, elemsC, nInner)
        ompParallelForChunked(0'i64, int64(nOuter), simdLoop, nil)
  
  of odkMatVec:
    if rhsView1Sym == nil or rhsView2Sym == nil:
      scalarFallback()
      return
    
    # Matrix-vector: y[i] = sum_j M[i,j] * x[j]
    result = quote do:
      block:
        let dstData = `lhsViewSym`.aosoaDataPtr
        let srcMData = `rhsView1Sym`.aosoaDataPtr
        let srcVData = `rhsView2Sym`.aosoaDataPtr
        let layout = `lhsViewSym`.data.simdLayout
        let nInner = layout.nSitesInner
        let nOuter = layout.nSitesOuter
        type ElemT = typeof(dstData[0])
        
        let matRows = `rhsView1Sym`.shape[0]
        let matCols = if `rhsView1Sym`.shape.len > 1: `rhsView1Sym`.shape[1] else: 1
        let elemsM = `rhsView1Sym`.data.tensorElementsPerSite
        let elemsV = `rhsView2Sym`.data.tensorElementsPerSite
        let elemsOut = `lhsViewSym`.data.tensorElementsPerSite
        proc simdLoop(chunkStart, chunkEnd: int64, ctx: pointer) {.cdecl.} =
          for outer in chunkStart..<chunkEnd:
            let outerIdx = int(outer)
            for i in 0..<matRows:
              var acc = newSimdVecDyn[ElemT](nInner)
              for j in 0..<matCols:
                let mElemIdx = i * matCols + j
                var vecM = loadSimdVectorDyn[ElemT](
                  srcMData, outerIdx, mElemIdx, elemsM, nInner)
                var vecV = loadSimdVectorDyn[ElemT](
                  srcVData, outerIdx, j, elemsV, nInner)
                acc = acc + vecM * vecV
              storeSimdVectorDyn(acc, dstData, outerIdx, i, elemsOut, nInner)
        ompParallelForChunked(0'i64, int64(nOuter), simdLoop, nil)
  
  of odkUnknown:
    # Unknown pattern — fall back to scalar per-site loop
    scalarFallback()

#[ ============================================================================
   Public Each Macro
   ============================================================================ ]#

macro each*(forLoop: ForLoopStmt): untyped =
  ## SIMD-vectorized parallel each loop for TensorFieldView (OpenMP backend)
  ##
  ## Analyzes the loop body at compile time to detect operation patterns,
  ## then generates SIMD-vectorized code using the AoSoA layout and
  ## SimdVecDyn load/store/arithmetic from simd/.
  ##
  ## Recognized patterns (SIMD-vectorized):
  ##   viewC[n] = viewA[n]                     # Copy
  ##   viewC[n] = viewA[n] + viewB[n]          # Addition
  ##   viewC[n] = viewA[n] - viewB[n]          # Subtraction
  ##   viewC[n] = 2.0 * viewA[n]               # Scalar multiply
  ##   viewC[n] = viewA[n] + 3.0               # Scalar add
  ##   viewC[n] = viewA[n] * viewB[n]          # Matrix multiply
  ##
  ## Falls back to scalar per-site loop for:
  ##   - Echo/print statements
  ##   - Stencil neighbor access
  ##   - Complex/unrecognized expressions
  ##
  ## Usage:
  ##   for n in each 0..<view.numSites():
  ##     viewC[n] = viewA[n] + viewB[n]
  
  let loopVar = forLoop[0]
  let loopRangeNode = forLoop[1][1]  # Skip 'each' wrapper
  let body = forLoop[2]
  
  # Check for echo - needs serial execution
  let needsSerial = hasEchoStatement(body)
  
  if loopRangeNode.kind == nnkInfix and loopRangeNode[0].strVal == "..<":
    let startExpr = loopRangeNode[1]
    let endExpr = loopRangeNode[2]
    
    if needsSerial:
      result = quote do:
        block:
          for `loopVar` in `startExpr`..<`endExpr`:
            `body`
    else:
      result = quote do:
        block:
          var `loopVar` {.inject.}: int = 0
          eachImpl(`loopVar`, `startExpr`, `endExpr`, `body`)
  else:
    result = quote do:
      block:
        for `loopVar` in `loopRangeNode`:
          `body`

when isMainModule:
  import ../tensor/sitetensor
  
  initOpenMP()
  echo "OpenMP SIMD dispatch module loaded"
  echo "Max threads: ", getNumThreads()
