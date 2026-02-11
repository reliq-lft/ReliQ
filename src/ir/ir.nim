#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/ir/ir.nim
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

## Shared Intermediate Representation for ReliQ Backend Codegen
##
## This module contains the backend-agnostic intermediate representation (IR)
## shared by both the OpenCL and OpenMP backends. It includes:
##
## 1. **Type detection utilities** — compile-time type introspection for
##    TensorFieldView, GaugeFieldView, Complex types, etc.
##
## 2. **Kernel information gathering** — ``gatherInfo`` walks the typed AST
##    to discover views, gauge views, stencils, and let bindings.
##
## 3. **Runtime variable detection** — finds int/float symbols and dot-expr
##    accesses that must be passed as kernel parameters.
##
## 4. **AST utility functions** — echo detection, FLOP call extraction,
##    element-level write detection, AST sanitization.
##
## Both backends import this module and only specialize the final C code
## emission (OpenCL C vs OpenMP C with SIMD intrinsics).

import std/[macros, tables, strutils, sets]

const VectorWidth* {.intdefine.} = 8
const DebugKernels* {.booldefine.} = false

#[ ============================================================================
   Type Detection Utilities
   ============================================================================ ]#

proc typeContains*(n: NimNode, name: string): bool =
  case n.kind
  of nnkSym:
    if n.strVal == name: return true
  of nnkBracketExpr:
    for child in n:
      if typeContains(child, name): return true
  else:
    for child in n:
      if typeContains(child, name): return true
  return false

proc isGaugeFieldViewSym*(n: NimNode): bool =
  try:
    return typeContains(n.getTypeInst(), "GaugeFieldView")
  except: return false

proc isTensorFieldViewSym*(n: NimNode): bool =
  try:
    return typeContains(n.getTypeInst(), "TensorFieldView")
  except: return false

proc isComplexSym*(n: NimNode): bool =
  try:
    let ti = n.getTypeInst()
    return typeContains(ti, "Complex64") or typeContains(ti, "Complex32")
  except: return false

proc isComplex32Sym*(n: NimNode): bool =
  try:
    return typeContains(n.getTypeInst(), "Complex32")
  except: return false

proc isMatMulSym*(n: NimNode): bool =
  ## Backward-compatible alias — checks for MatrixSiteProxy (which
  ## MatMulResult is now an alias of).
  try:
    let ti = n.getTypeInst()
    return typeContains(ti, "MatrixSiteProxy")
  except: return false

proc isTensorSiteProxySym*(n: NimNode): bool =
  try:
    return typeContains(n.getTypeInst(), "TensorSiteProxy")
  except: return false

proc isMatrixTypedNode*(n: NimNode): bool =
  ## True when *n* carries a matrix-like site proxy type — any of
  ## TensorSiteProxy, MatrixSiteProxy, VectorSiteProxy, or their aliases.
  ## Used by gatherInfo to tag let/var bindings that need array storage
  ## in the generated C code.
  try:
    let ti = n.getTypeInst()
    return typeContains(ti, "TensorSiteProxy") or
           typeContains(ti, "MatrixSiteProxy") or
           typeContains(ti, "VectorSiteProxy")
  except: return false

proc isIntSym*(n: NimNode): bool =
  try:
    return n.getTypeInst().repr == "int"
  except: return false

proc isFloatSym*(n: NimNode): bool =
  try:
    let r = n.getTypeInst().repr
    return r == "float" or r == "float64"
  except: return false

proc isDotExprFloat*(n: NimNode): bool =
  try:
    let r = n.getType().repr
    return r == "float" or r == "float64"
  except: return false

proc isDotExprInt*(n: NimNode): bool =
  try:
    return n.getType().repr == "int"
  except: return false

proc extractGaugeFieldDim*(n: NimNode): int =
  try:
    let ti = n.getTypeInst()
    if ti.kind == nnkBracketExpr and ti.len >= 2:
      if ti[1].kind in {nnkIntLit..nnkInt64Lit}:
        return ti[1].intVal.int
  except: discard
  return 4

type ElementType* = enum
  etFloat64, etFloat32, etInt32, etInt64

proc detectElementType*(n: NimNode): ElementType =
  try:
    let ti = n.getTypeInst()
    let r = ti.repr
    if r.contains("float32") or r.contains("cfloat"): return etFloat32
    if r.contains("int32") or r.contains("cint"): return etInt32
    if r.contains("int64") or r.contains("clonglong"): return etInt64
  except: discard
  return etFloat64

#[ ============================================================================
   Kernel Information Types
   ============================================================================ ]#

type
  ViewEntry* = object
    name*: string
    nimSym*: NimNode
    isRead*: bool
    isWrite*: bool
    isComplex*: bool
    scalarType*: ElementType

  GaugeViewEntry* = object
    name*: string
    nimSym*: NimNode
    dim*: int
    isComplex*: bool
    scalarType*: ElementType

  StencilEntry* = object
    name*: string
    nimSym*: NimNode

  LetBindingKind* = enum
    lbkStencilFwd, lbkStencilBwd, lbkStencilNeighbor, lbkStencilCorner, lbkMatMul, lbkOther

  LetBinding* = object
    varName*: string
    kind*: LetBindingKind
    stencilName*: string
    dirExpr*: string
    pointExpr*: string
    # Corner-specific fields (for lbkStencilCorner)
    signExprA*: string
    dirExprA*: string
    signExprB*: string
    dirExprB*: string
    nDim*: int  # lattice dimensionality D

  VarBinding* = object
    varName*: string
    isProxy*: bool  # True if this is a TensorSiteProxy var

  KernelInfo* = object
    views*: seq[ViewEntry]
    gaugeViews*: seq[GaugeViewEntry]
    stencils*: seq[StencilEntry]
    letBindings*: seq[LetBinding]
    varBindings*: seq[VarBinding]
    loopVar*: NimNode
    loopVarStr*: string
    isComplex*: bool
    hasStencil*: bool
    scalarType*: ElementType

proc getLetBinding*(info: KernelInfo, name: string): LetBinding =
  for lb in info.letBindings:
    if lb.varName == name: return lb
  return LetBinding(varName: name, kind: lbkOther)

#[ ============================================================================
   AST Analysis: gatherInfo
   ============================================================================ ]#

proc gatherInfo*(body: NimNode, loopVar: NimNode): KernelInfo =
  result.loopVar = loopVar
  result.loopVarStr = loopVar.strVal

  var viewTab = initTable[string, ViewEntry]()
  var gaugeTab = initTable[string, GaugeViewEntry]()
  var stencilTab = initTable[string, StencilEntry]()
  var letBindNames: HashSet[string]
  var hasStencilFlag = false
  var letBindingsLocal: seq[LetBinding]

  proc addView(sym: NimNode, name: string, rd, wr: bool) =
    if name notin viewTab:
      viewTab[name] = ViewEntry(name: name, nimSym: sym, isRead: rd, isWrite: wr,
                                 isComplex: isComplexSym(sym),
                                 scalarType: detectElementType(sym))
    else:
      if rd: viewTab[name].isRead = true
      if wr: viewTab[name].isWrite = true

  proc addStencil(sym: NimNode, name: string) =
    if name notin stencilTab:
      stencilTab[name] = StencilEntry(name: name, nimSym: sym)
    hasStencilFlag = true

  proc extractGaugeSym(n: NimNode): NimNode =
    ## From HiddenDeref(Call("[]", HiddenAddr(vu), mu)) extract vu
    if n.kind == nnkHiddenDeref and n[0].kind == nnkCall:
      let ic = n[0]
      if ic[0].kind == nnkSym and ic[0].strVal == "[]" and ic.len >= 3:
        let arg = ic[1]
        if arg.kind == nnkHiddenAddr and arg[0].kind == nnkSym:
          return arg[0]
        elif arg.kind == nnkSym:
          return arg
    return nil

  proc walk(n: NimNode) =
    case n.kind
    of nnkCall:
      if n.len >= 2 and n[0].kind == nnkSym:
        let fn = n[0].strVal
        case fn
        of "[]=":
          let target = n[1]
          if target.kind == nnkSym and isTensorFieldViewSym(target):
            addView(target, target.strVal, false, true)
          elif target.kind == nnkCall and target[0].kind == nnkSym and target[0].strVal == "[]":
            # Element-level write: view[n][i,j] = val
            if target.len >= 3:
              let inner = target[1]
              if inner.kind == nnkSym and isTensorFieldViewSym(inner):
                addView(inner, inner.strVal, false, true)
          for i in 1..<n.len: walk(n[i])
        of "[]":
          if n.len >= 3:
            let gs = extractGaugeSym(n[1])
            if gs != nil and isGaugeFieldViewSym(gs):
              let gn = gs.strVal
              if gn notin gaugeTab:
                gaugeTab[gn] = GaugeViewEntry(name: gn, nimSym: gs,
                  dim: extractGaugeFieldDim(gs), isComplex: isComplexSym(gs),
                  scalarType: detectElementType(gs))
              walk(n[2])
              return
            let sym = if n[1].kind == nnkSym: n[1]
                      elif n[1].kind == nnkHiddenDeref and n[1][0].kind == nnkSym: n[1][0]
                      elif n[1].kind == nnkHiddenAddr and n[1][0].kind == nnkSym: n[1][0]
                      else: nil
            if sym != nil and isTensorFieldViewSym(sym):
              addView(sym, sym.strVal, true, false)
            for i in 1..<n.len: walk(n[i])
        of "fwd", "bwd":
          if n.len >= 4 and n[1].kind == nnkSym:
            addStencil(n[1], n[1].strVal)
          for i in 1..<n.len: walk(n[i])
        of "corner":
          if n.len >= 7 and n[1].kind == nnkSym:
            addStencil(n[1], n[1].strVal)
          for i in 1..<n.len: walk(n[i])
        of "neighbor":
          if n.len >= 4 and n[1].kind == nnkSym:
            addStencil(n[1], n[1].strVal)
          for i in 1..<n.len: walk(n[i])
        else:
          for i in 1..<n.len: walk(n[i])

    of nnkInfix:
      if n.len >= 3 and n[0].kind == nnkSym and n[0].strVal == "+=":
        let lhs = n[1]
        if lhs.kind == nnkCall and lhs[0].kind == nnkSym and lhs[0].strVal == "[]":
          if lhs.len >= 3:
            let viewSym = lhs[1]
            if viewSym.kind == nnkSym and isTensorFieldViewSym(viewSym):
              addView(viewSym, viewSym.strVal, true, true)
      for child in n: walk(child)

    of nnkLetSection:
      for idefs in n:
        if idefs.kind == nnkIdentDefs and idefs.len >= 3:
          let vsym = idefs[0]
          let val = idefs[2]
          letBindNames.incl vsym.strVal
          if val.kind == nnkCall and val.len >= 4 and val[0].kind == nnkSym:
            let fn = val[0].strVal
            if fn == "fwd" or fn == "bwd":
              addStencil(val[1], val[1].strVal)
              var dirStr = ""
              if val[3].kind == nnkSym: dirStr = val[3].strVal
              elif val[3].kind in {nnkIntLit..nnkInt64Lit}: dirStr = $val[3].intVal
              letBindingsLocal.add LetBinding(
                varName: vsym.strVal,
                kind: if fn == "fwd": lbkStencilFwd else: lbkStencilBwd,
                stencilName: val[1].strVal,
                dirExpr: dirStr)
            elif fn == "corner":
              addStencil(val[1], val[1].strVal)
              proc extractIntExpr(node: NimNode): string =
                if node.kind in {nnkIntLit..nnkInt64Lit}: $node.intVal
                elif node.kind == nnkSym: node.strVal
                elif node.kind == nnkPrefix and node[0].strVal == "-":
                  "(-" & extractIntExpr(node[1]) & ")"
                else: "0"
              var nDimVal = 4
              let stencilType = val[1].getType()
              if stencilType.kind == nnkBracketExpr and stencilType.len >= 2:
                if stencilType[1].kind == nnkIntLit:
                  nDimVal = stencilType[1].intVal.int
              letBindingsLocal.add LetBinding(
                varName: vsym.strVal,
                kind: lbkStencilCorner,
                stencilName: val[1].strVal,
                signExprA: extractIntExpr(val[3]),
                dirExprA: extractIntExpr(val[4]),
                signExprB: extractIntExpr(val[5]),
                dirExprB: extractIntExpr(val[6]),
                nDim: nDimVal)
            elif fn == "neighbor":
              addStencil(val[1], val[1].strVal)
              var ptStr = "0"
              if val.len >= 4:
                if val[3].kind in {nnkIntLit..nnkInt64Lit}: ptStr = $val[3].intVal
                elif val[3].kind == nnkSym: ptStr = val[3].strVal
              letBindingsLocal.add LetBinding(
                varName: vsym.strVal,
                kind: lbkStencilNeighbor,
                stencilName: val[1].strVal,
                pointExpr: ptStr)
            else:
              var isMat = false
              try: isMat = isMatrixTypedNode(vsym)
              except: discard
              letBindingsLocal.add LetBinding(
                varName: vsym.strVal,
                kind: if isMat: lbkMatMul else: lbkOther)
              walk(val)
          else:
            var isMat = false
            try: isMat = isMatrixTypedNode(vsym)
            except: discard
            letBindingsLocal.add LetBinding(
              varName: vsym.strVal,
              kind: if isMat: lbkMatMul else: lbkOther)
            walk(val)

    of nnkVarSection:
      for idefs in n:
        if idefs.kind == nnkIdentDefs and idefs.len >= 3:
          let vsym = idefs[0]
          let val = idefs[2]
          letBindNames.incl vsym.strVal
          var isMat = false
          try: isMat = isMatrixTypedNode(vsym)
          except: discard
          letBindingsLocal.add LetBinding(
            varName: vsym.strVal,
            kind: if isMat: lbkMatMul else: lbkOther)
          walk(val)

    of nnkAsgn:
      if n.len >= 2:
        walk(n[1])

    of nnkForStmt:
      # Register for-loop variable to prevent it from being picked up
      # as a runtime int capture variable
      if n.len >= 3 and n[0].kind == nnkSym:
        letBindNames.incl n[0].strVal
        letBindingsLocal.add LetBinding(
          varName: n[0].strVal,
          kind: lbkOther)
      for child in n: walk(child)

    of nnkHiddenCallConv:
      if n.len >= 2: walk(n[1])
    else:
      for child in n: walk(child)

  walk(body)

  result.hasStencil = hasStencilFlag
  result.letBindings = letBindingsLocal
  for _, v in viewTab: result.views.add v
  for _, gv in gaugeTab: result.gaugeViews.add gv
  for _, s in stencilTab: result.stencils.add s
  for v in result.views:
    if v.isComplex: result.isComplex = true
  for gv in result.gaugeViews:
    if gv.isComplex: result.isComplex = true
  result.scalarType = etFloat64
  for v in result.views:
    result.scalarType = v.scalarType
    break
  for gv in result.gaugeViews:
    result.scalarType = gv.scalarType
    break

#[ ============================================================================
   Runtime Variable Detection
   ============================================================================ ]#

proc findRuntimeIntVars*(body: NimNode, info: KernelInfo): seq[tuple[name: string, sym: NimNode]] =
  var seen: HashSet[string]
  var viewNames, gaugeNames, stencilNames, letNames: HashSet[string]
  var found: seq[tuple[name: string, sym: NimNode]]

  for v in info.views: viewNames.incl v.name
  for gv in info.gaugeViews: gaugeNames.incl gv.name
  for s in info.stencils: stencilNames.incl s.name
  for lb in info.letBindings: letNames.incl lb.varName

  proc scan(n: NimNode) =
    case n.kind
    of nnkDotExpr:
      # Only recurse into the object (first child), skip the field sym (second child)
      # to avoid picking up struct field symbols. The dot pair itself is handled
      # by findRuntimeDotIntVars.
      if n.len >= 1: scan(n[0])
    of nnkSym:
      let name = n.strVal
      if name != info.loopVarStr and name notin seen and
         name notin viewNames and name notin gaugeNames and
         name notin stencilNames and name notin letNames:
        if isIntSym(n):
          seen.incl name
          found.add (name, n)
    else:
      for child in n: scan(child)

  scan(body)
  return found

proc findRuntimeFloatVars*(body: NimNode, info: KernelInfo): seq[tuple[name: string, sym: NimNode]] =
  var seen: HashSet[string]
  var viewNames, gaugeNames, stencilNames, letNames: HashSet[string]
  var found: seq[tuple[name: string, sym: NimNode]]

  for v in info.views: viewNames.incl v.name
  for gv in info.gaugeViews: gaugeNames.incl gv.name
  for s in info.stencils: stencilNames.incl s.name
  for lb in info.letBindings: letNames.incl lb.varName

  proc scan(n: NimNode) =
    case n.kind
    of nnkDotExpr:
      # Only recurse into the object (first child), skip the field sym (second child)
      # to avoid picking up struct field symbols. The dot pair itself is handled
      # by findRuntimeDotFloatVars.
      if n.len >= 1: scan(n[0])
    of nnkSym:
      let name = n.strVal
      if name != info.loopVarStr and name notin seen and
         name notin viewNames and name notin gaugeNames and
         name notin stencilNames and name notin letNames:
        if isFloatSym(n):
          seen.incl name
          found.add (name, n)
    else:
      for child in n: scan(child)

  scan(body)
  return found

proc findRuntimeDotFloatVars*(body: NimNode, info: KernelInfo): seq[tuple[name: string, dotNode: NimNode]] =
  var seen: HashSet[string]
  var found: seq[tuple[name: string, dotNode: NimNode]]

  proc scan(n: NimNode) =
    case n.kind
    of nnkDotExpr:
      if n.len >= 2 and n[0].kind == nnkSym and n[1].kind == nnkSym:
        if isDotExprFloat(n):
          let name = n[0].strVal & "_" & n[1].strVal
          if name notin seen:
            seen.incl name
            found.add (name, n)
          return
      for child in n: scan(child)
    else:
      for child in n: scan(child)

  scan(body)
  return found

proc findRuntimeDotIntVars*(body: NimNode, info: KernelInfo): seq[tuple[name: string, dotNode: NimNode]] =
  var seen: HashSet[string]
  var found: seq[tuple[name: string, dotNode: NimNode]]

  proc scan(n: NimNode) =
    case n.kind
    of nnkDotExpr:
      if n.len >= 2 and n[0].kind == nnkSym and n[1].kind == nnkSym:
        if isDotExprInt(n):
          let name = n[0].strVal & "_" & n[1].strVal
          if name notin seen:
            seen.incl name
            found.add (name, n)
          return
      for child in n: scan(child)
    else:
      for child in n: scan(child)

  scan(body)
  return found

#[ ============================================================================
   AST Utility Functions
   ============================================================================ ]#

proc hasEchoStatement*(n: NimNode): bool =
  case n.kind
  of nnkCall:
    if n[0].kind == nnkSym and n[0].strVal in ["echo", "debugEcho"]: return true
    for child in n:
      if hasEchoStatement(child): return true
  of nnkCommand:
    if n[0].kind in {nnkIdent, nnkSym} and n[0].strVal in ["echo", "debugEcho"]: return true
    for child in n:
      if hasEchoStatement(child): return true
  else:
    for child in n:
      if hasEchoStatement(child): return true
  return false

proc isElementLevelWrite*(stmt: NimNode): bool =
  if stmt.kind != nnkCall or stmt.len < 4: return false
  if stmt[0].kind != nnkSym or stmt[0].strVal != "[]=": return false
  let a = stmt[1]
  return a.kind == nnkCall and a.len >= 2 and a[0].kind == nnkSym and a[0].strVal == "[]"

proc isAddFlopCall*(n: NimNode): bool =
  n.kind == nnkCall and n.len >= 3 and
    n[0].kind == nnkSym and n[0].strVal == "addFLOPImpl"

proc ind*(depth: int): string = "  ".repeat(depth)

proc sanitizeFieldSyms*(n: NimNode): NimNode =
  case n.kind
  of nnkDotExpr:
    let lhs = sanitizeFieldSyms(n[0])
    let rhs = n[1]
    if rhs.kind == nnkSym and rhs.symKind == nskField:
      result = nnkDotExpr.newTree(lhs, ident(rhs.strVal))
    else:
      result = nnkDotExpr.newTree(lhs, sanitizeFieldSyms(rhs))
  of nnkSym:
    result = n
  of nnkLiterals:
    result = n
  else:
    result = copyNimNode(n)
    for child in n:
      result.add sanitizeFieldSyms(child)

proc untypeAst*(n: NimNode): NimNode =
  case n.kind
  of nnkSym:
    result = ident(n.strVal)
  of nnkOpenSymChoice, nnkClosedSymChoice:
    result = ident(n[0].strVal)
  of nnkLiterals:
    result = n
  of nnkDotExpr:
    result = nnkDotExpr.newTree(untypeAst(n[0]), untypeAst(n[1]))
  else:
    result = copyNimNode(n)
    for child in n:
      result.add untypeAst(child)

type
  FlopCallEntry* = object
    flopsExpr*: NimNode
    condExpr*: NimNode

proc findFlopCalls*(body: NimNode): seq[FlopCallEntry] =
  var entries: seq[FlopCallEntry]

  proc scan(n: NimNode, cond: NimNode) =
    case n.kind
    of nnkCall:
      if isAddFlopCall(n):
        entries.add FlopCallEntry(flopsExpr: n[2], condExpr: cond)
        return
      for child in n: scan(child, cond)
    of nnkIfStmt:
      for branch in n:
        if branch.kind == nnkElifBranch:
          let branchCond = branch[0]
          let branchBody = branch[1]
          let effectiveCond = if cond == nil: branchCond
                              else: nnkInfix.newTree(ident"and", cond, branchCond)
          if branchBody.kind == nnkStmtList:
            for child in branchBody: scan(child, effectiveCond)
          else:
            scan(branchBody, effectiveCond)
        elif branch.kind == nnkElse:
          if branch.len > 0:
            let elseBody = branch[0]
            if elseBody.kind == nnkStmtList:
              for child in elseBody: scan(child, cond)
            else:
              scan(elseBody, cond)
    of nnkStmtList:
      for child in n: scan(child, cond)
    else:
      for child in n: scan(child, cond)

  scan(body, nil)
  return entries

#[ ============================================================================
   Site Reference — Resolved view[site] access
   ============================================================================ ]#

type SiteRef* = object
  dataVar*: string
  groupVar*: string
  laneVar*: string
  elemsVar*: string
  isGauge*: bool
  gaugeName*: string
  dirExpr*: string
  gaugeDim*: int

#[ ============================================================================
   Code Generation Context (backend-agnostic base)
   ============================================================================ ]#

type
  CodeCtx* = object
    loopVarStr*: string
    isComplex*: bool
    scalarType*: string   # "double", "float", "int", "long long"
    elemType*: string     # compound type for complex, scalar for real
    vecType*: string      # OpenCL: "double2"/"float2"; OpenMP: unused
    viewNames*: HashSet[string]
    gaugeNames*: HashSet[string]
    stencilNames*: HashSet[string]
    letNames*: HashSet[string]
    info*: KernelInfo
    tmpIdx*: int

proc nextTmp*(ctx: var CodeCtx): string =
  result = "_t" & $ctx.tmpIdx
  ctx.tmpIdx += 1

#[ ============================================================================
   AoSoA Indexing (shared between backends)
   ============================================================================ ]#

proc aosoaIdx*(dataVar, groupVar, laneVar, elemsVar, elemExpr: string): string =
  ## Scalar element access: data[group * (VW * elems) + elem * VW + lane]
  let vw = $VectorWidth
  dataVar & "[" & groupVar & " * (" & vw & " * " & elemsVar & ") + (" & elemExpr & ") * " & vw & " + " & laneVar & "]"

proc aosoaBase*(dataVar, groupVar, elemsVar, elemExpr: string): string =
  ## Base address for VW-wide SIMD load/store
  let vw = $VectorWidth
  "&" & dataVar & "[" & groupVar & " * (" & vw & " * " & elemsVar & ") + (" & elemExpr & ") * " & vw & "]"

#[ ============================================================================
   Site Reference Resolution (shared logic)
   ============================================================================ ]#

proc resolveSiteRef*(viewExpr, siteExpr: NimNode, ctx: var CodeCtx): SiteRef =
  var groupVar = "group"
  var laneVar = "lane"

  block resolveGL:
    let sn = siteExpr
    if sn.kind == nnkHiddenCallConv and sn.len >= 2 and sn[1].kind == nnkSym:
      let nm = sn[1].strVal
      groupVar = nm & "_group"
      laneVar = nm & "_lane"
      break resolveGL
    if sn.kind == nnkSym:
      let nm = sn.strVal
      if nm == ctx.loopVarStr:
        groupVar = "group"
        laneVar = "lane"
        break resolveGL
      let lb = ctx.info.getLetBinding(nm)
      if lb.kind in {lbkStencilFwd, lbkStencilBwd, lbkStencilNeighbor, lbkStencilCorner}:
        groupVar = nm & "_group"
        laneVar = nm & "_lane"
        break resolveGL

  # Check for GaugeFieldView double-index
  if viewExpr.kind == nnkHiddenDeref and viewExpr[0].kind == nnkCall:
    let ic = viewExpr[0]
    if ic[0].kind == nnkSym and ic[0].strVal == "[]" and ic.len >= 3:
      var gs: NimNode = nil
      if ic[1].kind == nnkHiddenAddr and ic[1][0].kind == nnkSym:
        gs = ic[1][0]
      elif ic[1].kind == nnkSym:
        gs = ic[1]
      if gs != nil:
        let gn = gs.strVal
        var isGauge = false
        for gv in ctx.info.gaugeViews:
          if gv.name == gn: isGauge = true
        if isGauge:
          var dirCode = ""
          let dirNode = ic[2]
          if dirNode.kind == nnkSym: dirCode = dirNode.strVal
          elif dirNode.kind in {nnkIntLit..nnkInt64Lit}: dirCode = $dirNode.intVal
          else: dirCode = "0"
          var dim = 4
          for gv in ctx.info.gaugeViews:
            if gv.name == gn: dim = gv.dim
          return SiteRef(isGauge: true, gaugeName: gn, dirExpr: dirCode,
                         gaugeDim: dim, groupVar: groupVar, laneVar: laneVar,
                         elemsVar: gn & "_elems")

  # Regular TensorFieldView
  var vn = ""
  if viewExpr.kind == nnkSym:
    vn = viewExpr.strVal
  elif viewExpr.kind == nnkHiddenDeref and viewExpr[0].kind == nnkSym:
    vn = viewExpr[0].strVal
  elif viewExpr.kind == nnkHiddenAddr and viewExpr[0].kind == nnkSym:
    vn = viewExpr[0].strVal
  else:
    vn = "unknown"

  return SiteRef(dataVar: vn & "_data", groupVar: groupVar, laneVar: laneVar,
                 elemsVar: vn & "_elems")

#[ ============================================================================
   Scalar Expression Transpilation (shared)
   ============================================================================ ]#

proc transpileScalar*(n: NimNode, ctx: var CodeCtx): string =
  case n.kind
  of nnkSym:
    return n.strVal
  of nnkDotExpr:
    if n.len >= 2 and n[0].kind == nnkSym and n[1].kind == nnkSym:
      return n[0].strVal & "_" & n[1].strVal
    return "0"
  of nnkIntLit..nnkInt64Lit:
    return $n.intVal
  of nnkFloatLit..nnkFloat64Lit:
    return $n.floatVal
  of nnkHiddenStdConv, nnkConv, nnkHiddenDeref, nnkHiddenAddr, nnkHiddenSubConv:
    if n.len > 0: return transpileScalar(n[^1], ctx)
  of nnkStmtListExpr:
    for i in countdown(n.len - 1, 0):
      if n[i].kind != nnkEmpty:
        return transpileScalar(n[i], ctx)
  of nnkPrefix:
    if n.len >= 2 and n[0].kind == nnkSym:
      let op = n[0].strVal
      if op == "not":
        return "!(" & transpileScalar(n[1], ctx) & ")"
      else:
        return op & "(" & transpileScalar(n[1], ctx) & ")"
  of nnkCall:
    if n[0].kind == nnkSym:
      let fn = n[0].strVal
      # Element read on local matrix temp: m[i] or m[i,j] or m[i,j,k]
      if fn == "[]" and n.len >= 3:
        let target = n[1]
        if target.kind == nnkSym:
          var isMat = false
          try: isMat = isMatrixTypedNode(target)
          except: discard
          if isMat:
            let name = target.strVal
            if n.len == 3:
              # 1D: m[idx]
              let idx = transpileScalar(n[2], ctx)
              return name & "[" & idx & "]"
            elif n.len == 4:
              # 2D: m[i,j]
              let row = transpileScalar(n[2], ctx)
              let col = transpileScalar(n[3], ctx)
              return name & "[(" & row & ")*NC+(" & col & ")]"
            elif n.len >= 5:
              # 3D: m[i,j,k]
              let i = transpileScalar(n[2], ctx)
              let j = transpileScalar(n[3], ctx)
              let k = transpileScalar(n[4], ctx)
              return name & "[(" & i & ")*NC*NC+(" & j & ")*NC+(" & k & ")]"
      # conj — complex conjugate of scalar element
      if fn == "conj" and n.len == 2:
        let inner = transpileScalar(n[1], ctx)
        if ctx.isComplex:
          return "cconj(" & inner & ")"
        else:
          return inner
      var args: seq[string]
      for i in 1..<n.len: args.add transpileScalar(n[i], ctx)
      return n[0].strVal & "(" & args.join(", ") & ")"
  of nnkInfix:
    if n.len >= 3 and n[0].kind == nnkSym:
      return "(" & transpileScalar(n[1], ctx) & " " & n[0].strVal & " " & transpileScalar(n[2], ctx) & ")"
  else: discard
  return "0"

#[ ============================================================================
   Matrix Expression Result
   ============================================================================ ]#

type MatResult* = tuple[code: string, elems: string]

#[ ============================================================================
   Custom Site Operation Registry
   ============================================================================

   This system lets users define custom operations on TensorSiteProxy that
   the transpiler can recognise and emit backend-specific C code for.

   Usage overview (user-facing):

     # 1. Register the operation with C code templates
     registerSiteOp("myInverse"):
       arity = 1                       # number of matrix arguments
       resultElems = "NC*NC"           # element count of result
       codeTemplate = """
         // Compute inverse in-place: Gauss–Jordan on $elemType target[$resultElems]
         for (int _i = 0; _i < NC; _i++)
           for (int _j = 0; _j < NC; _j++)
             $target[_i*NC+_j] = $arg0[_i*NC+_j];
         // ... user-supplied inversion kernel ...
       """

     # 2. Declare the phantom proc (in user code or a library module)
     proc myInverse*[L, T](a: TensorSiteProxy[L, T]): MatrixSiteProxy[L, T] =
       raise newException(Defect, "myInverse phantom")

     # 3. Use inside `each`:
     #   for n in each view.all:
     #     out[n] = myInverse(mat[n])
     #
     # The transpiler sees the `myInverse` call, looks up the registered
     # C template, substitutes $target/$arg0, and emits the inline code.

   The templates support these placeholders (replaced at emit time):
     $target      — name of the C target array
     $arg0..$arg3 — names of the C arrays holding evaluated arguments
     $argElems0.. — element-count expressions for each argument
     $elemType    — backend element type ("double", "double2", "simd_v", etc.)
     $resultElems — same as the `resultElems` field
     $NC          — colour-matrix dimension (replaced by literal or macro)
     $d           — current indentation depth (integer)
]#

type
  CustomSiteOp* = object
    name*: string         ## Nim proc name as it appears in the typed AST
    arity*: int           ## Number of matrix-valued arguments (1..4)
    resultElems*: string  ## C expression for output element count, e.g. "NC*NC"
    codeTemplate*: string ## C code template with $-placeholders

# Compile-time registry — populated by registerSiteOp, queried by backends
var customSiteOps* {.compileTime.}: seq[CustomSiteOp]

proc lookupCustomOp*(name: string): int {.compileTime.} =
  ## Returns index into customSiteOps, or -1 if not found
  for i, op in customSiteOps:
    if op.name == name: return i
  return -1

proc instantiateTemplate*(tmpl: string, target: string, args: seq[string],
                           argElems: seq[string], elemType, resultElems: string,
                           depth: int): string {.compileTime.} =
  ## Replace $-placeholders in a custom op code template with concrete values
  result = tmpl
  result = result.replace("$target", target)
  result = result.replace("$elemType", elemType)
  result = result.replace("$resultElems", resultElems)
  result = result.replace("$NC", "NC")
  result = result.replace("$d", $depth)
  for i in 0..<min(args.len, 4):
    result = result.replace("$arg" & $i, args[i])
  for i in 0..<min(argElems.len, 4):
    result = result.replace("$argElems" & $i, argElems[i])
  # Indent each line
  let prefix = ind(depth)
  var lines = result.split('\n')
  var indented: seq[string]
  for line in lines:
    if line.strip().len > 0:
      indented.add(prefix & line.strip() & "\n")
  result = indented.join("")

macro registerSiteOp*(name: static[string], arity: static[int],
                       resultElems: static[string],
                       codeTemplate: static[string]): untyped =
  ## Register a custom site operation for the transpiler.
  ##
  ## Example:
  ##   registerSiteOp("myInverse", 1, "NC*NC", """
  ##     for (int _i = 0; _i < NC; _i++)
  ##       for (int _j = 0; _j < NC; _j++)
  ##         $target[_i*NC+_j] = $arg0[_i*NC+_j];
  ##   """)
  customSiteOps.add CustomSiteOp(
    name: name, arity: arity,
    resultElems: resultElems, codeTemplate: codeTemplate)
  result = newStmtList()
