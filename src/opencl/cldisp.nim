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
## It uses a general-purpose AST-to-OpenCL-C transpiler that handles:
## - AoSoA memory layout for coalesced GPU access
## - Matrix/vector operations (matmul, matadd, matvec, scalar ops)
## - Complex arithmetic (Complex64/Complex32 via double2/float2)
## - Stencil neighbor access patterns (including runtime direction indices)
## - GaugeFieldView multi-direction array access
## - Let bindings for intermediate results
## - Adjoint (conjugate transpose) operations
## - In-place accumulation (+=)
## - Element-level writes (view[n][i,j] = val)
## - Multi-device dispatch

import std/[macros, tables, strutils, sets]

import clbase
export clbase

const VectorWidth* {.intdefine.} = 8
const DebugKernels* {.booldefine.} = false
const UseWorkGroups* {.booldefine.} = false

#[ ============================================================================
   Type Detection Utilities
   ============================================================================ ]#

proc typeContains(n: NimNode, name: string): bool =
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

proc isGaugeFieldViewSym(n: NimNode): bool =
  try:
    return typeContains(n.getTypeInst(), "GaugeFieldView")
  except: return false

proc isTensorFieldViewSym(n: NimNode): bool =
  try:
    return typeContains(n.getTypeInst(), "TensorFieldView")
  except: return false

proc isComplexSym(n: NimNode): bool =
  try:
    let ti = n.getTypeInst()
    return typeContains(ti, "Complex64") or typeContains(ti, "Complex32")
  except: return false

proc isComplex32Sym(n: NimNode): bool =
  try:
    return typeContains(n.getTypeInst(), "Complex32")
  except: return false

proc isMatMulSym(n: NimNode): bool =
  try:
    return typeContains(n.getTypeInst(), "MatMulResult")
  except: return false

proc isIntSym(n: NimNode): bool =
  try:
    return n.getTypeInst().repr == "int"
  except: return false

proc extractGaugeFieldDim(n: NimNode): int =
  try:
    let ti = n.getTypeInst()
    if ti.kind == nnkBracketExpr and ti.len >= 2:
      if ti[1].kind in {nnkIntLit..nnkInt64Lit}:
        return ti[1].intVal.int
  except: discard
  return 4

type ElementType* = enum
  etFloat64, etFloat32, etInt32, etInt64

proc detectElementType(n: NimNode): ElementType =
  ## Detect the scalar element type from a view's Nim type
  try:
    let ti = n.getTypeInst()
    let r = ti.repr
    if r.contains("float32") or r.contains("cfloat"): return etFloat32
    if r.contains("int32") or r.contains("cint"): return etInt32
    if r.contains("int64") or r.contains("clonglong"): return etInt64
  except: discard
  return etFloat64

proc elementTypeToOpenCL*(et: ElementType): string =
  case et
  of etFloat32: "float"
  of etFloat64: "double"
  of etInt32: "int"
  of etInt64: "long"

#[ ============================================================================
   Kernel Information Gathering
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
    lbkStencilFwd, lbkStencilBwd, lbkStencilNeighbor, lbkMatMul, lbkOther

  LetBinding* = object
    varName*: string
    kind*: LetBindingKind
    stencilName*: string
    dirExpr*: string
    pointExpr*: string

  KernelInfo* = object
    views*: seq[ViewEntry]
    gaugeViews*: seq[GaugeViewEntry]
    stencils*: seq[StencilEntry]
    letBindings*: seq[LetBinding]
    loopVar*: NimNode
    loopVarStr*: string
    isComplex*: bool
    hasStencil*: bool
    scalarType*: ElementType

proc getLetBinding*(info: KernelInfo, name: string): LetBinding =
  for lb in info.letBindings:
    if lb.varName == name: return lb
  return LetBinding(varName: name, kind: lbkOther)

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
          # Assignment target
          let target = n[1]
          if target.kind == nnkSym and isTensorFieldViewSym(target):
            addView(target, target.strVal, false, true)
          elif target.kind == nnkCall and target[0].kind == nnkSym and target[0].strVal == "[]":
            # Element-level: view[n][i,j] = val
            if target[1].kind == nnkSym and isTensorFieldViewSym(target[1]):
              addView(target[1], target[1].strVal, false, true)
          for i in 1..<n.len: walk(n[i])
        of "[]":
          if n.len >= 3:
            # Check for GaugeFieldView double-index
            let gs = extractGaugeSym(n[1])
            if gs != nil and isGaugeFieldViewSym(gs):
              let gn = gs.strVal
              if gn notin gaugeTab:
                gaugeTab[gn] = GaugeViewEntry(name: gn, nimSym: gs,
                  dim: extractGaugeFieldDim(gs), isComplex: isComplexSym(gs),
                  scalarType: detectElementType(gs))
              walk(n[2])
              return
            # Regular view read
            let sym = if n[1].kind == nnkSym: n[1]
                      elif n[1].kind == nnkHiddenDeref and n[1][0].kind == nnkSym: n[1][0]
                      else: nil
            if sym != nil and isTensorFieldViewSym(sym):
              addView(sym, sym.strVal, true, false)
            for i in 1..<n.len: walk(n[i])
        of "fwd", "bwd":
          if n.len >= 4 and n[1].kind == nnkSym:
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
        # += target is read+write
        let lhs = n[1]
        if lhs.kind == nnkCall and lhs[0].kind == nnkSym and lhs[0].strVal == "[]":
          if lhs[1].kind == nnkSym and isTensorFieldViewSym(lhs[1]):
            addView(lhs[1], lhs[1].strVal, true, true)
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
              # Check if result is MatMulResult
              var isMatmul = false
              try: isMatmul = isMatMulSym(vsym)
              except: discard
              letBindingsLocal.add LetBinding(
                varName: vsym.strVal,
                kind: if isMatmul: lbkMatMul else: lbkOther)
              walk(val)
          else:
            var isMatmul = false
            try: isMatmul = isMatMulSym(vsym)
            except: discard
            letBindingsLocal.add LetBinding(
              varName: vsym.strVal,
              kind: if isMatmul: lbkMatMul else: lbkOther)
            walk(val)

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
  # Determine scalar type from views (first view wins, all should agree)
  result.scalarType = etFloat64  # default
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
  ## Find int-typed symbols that are not the loop var, views, stencils, or let bindings
  var seen: HashSet[string]
  var viewNames: HashSet[string]
  var gaugeNames: HashSet[string]
  var stencilNames: HashSet[string]
  var letNames: HashSet[string]
  var found: seq[tuple[name: string, sym: NimNode]]

  for v in info.views: viewNames.incl v.name
  for gv in info.gaugeViews: gaugeNames.incl gv.name
  for s in info.stencils: stencilNames.incl s.name
  for lb in info.letBindings: letNames.incl lb.varName

  proc scan(n: NimNode) =
    case n.kind
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

#[ ============================================================================
   OpenCL C Code Generation — General AST Transpiler
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

proc ind*(depth: int): string = "  ".repeat(depth)

type
  CodeCtx* = object
    loopVarStr*: string
    isComplex*: bool
    scalarType*: string  # "double" or "float"
    vecType*: string     # "double2" or "float2"
    elemType*: string    # vecType for complex, scalarType for real
    viewNames*: HashSet[string]
    gaugeNames*: HashSet[string]
    stencilNames*: HashSet[string]
    letNames*: HashSet[string]
    info*: KernelInfo
    tmpIdx*: int

proc newCodeCtx*(info: KernelInfo): CodeCtx =
  result.loopVarStr = info.loopVarStr
  result.isComplex = info.isComplex
  result.scalarType = elementTypeToOpenCL(info.scalarType)
  result.vecType = if info.scalarType == etFloat32: "float2" else: "double2"
  result.elemType = if info.isComplex: result.vecType else: result.scalarType
  for v in info.views: result.viewNames.incl v.name
  for gv in info.gaugeViews: result.gaugeNames.incl gv.name
  for s in info.stencils: result.stencilNames.incl s.name
  for lb in info.letBindings: result.letNames.incl lb.varName
  result.info = info

proc nextTmp*(ctx: var CodeCtx): string =
  result = "_t" & $ctx.tmpIdx
  ctx.tmpIdx += 1

#[ --- AoSoA Element Access --- ]#

proc aosoaIdx*(dataVar, groupVar, laneVar, elemsVar, elemExpr: string): string =
  ## Generate AoSoA index expression: data[group * (VW * elems) + elem * VW + lane]
  let vw = $VectorWidth
  dataVar & "[" & groupVar & " * (" & vw & " * " & elemsVar & ") + (" & elemExpr & ") * " & vw & " + " & laneVar & "]"

type SiteRef* = object
  ## Resolved site reference: which data buffer, what group/lane, what elems var
  dataVar*: string    # e.g. "vplaq_data" or needs runtime switch for gauge
  groupVar*: string   # e.g. "group" or "fwdMu_group"
  laneVar*: string    # e.g. "lane" or "fwdMu_lane"
  elemsVar*: string   # e.g. "vplaq_elems"
  isGauge*: bool      # True if this needs runtime direction selection
  gaugeName*: string  # For gauge: base name
  dirExpr*: string    # For gauge: direction expression (C code)
  gaugeDim*: int      # For gauge: number of directions

proc resolveSiteRef*(viewExpr, siteExpr: NimNode, ctx: var CodeCtx): SiteRef =
  ## Resolve a view[site] access to data buffer + group/lane + elems info
  
  # --- Determine group/lane from site expression ---
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
      if lb.kind in {lbkStencilFwd, lbkStencilBwd, lbkStencilNeighbor}:
        groupVar = nm & "_group"
        laneVar = nm & "_lane"
        break resolveGL
  
  # --- Check for GaugeFieldView double-index ---
  # AST: Call("[]", HiddenDeref(Call("[]", HiddenAddr(vu), mu)), n)
  # Here viewExpr = HiddenDeref(Call("[]", HiddenAddr(vu), mu))
  if viewExpr.kind == nnkHiddenDeref and viewExpr[0].kind == nnkCall:
    let ic = viewExpr[0]
    if ic[0].kind == nnkSym and ic[0].strVal == "[]" and ic.len >= 3:
      var gs: NimNode = nil
      if ic[1].kind == nnkHiddenAddr and ic[1][0].kind == nnkSym:
        gs = ic[1][0]
      elif ic[1].kind == nnkSym:
        gs = ic[1]
      if gs != nil and gs.strVal in ctx.gaugeNames:
        let gn = gs.strVal
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
  
  # --- Regular TensorFieldView ---
  var vn = ""
  if viewExpr.kind == nnkSym:
    vn = viewExpr.strVal
  elif viewExpr.kind == nnkHiddenDeref and viewExpr[0].kind == nnkSym:
    vn = viewExpr[0].strVal
  else:
    vn = "unknown"
  
  return SiteRef(dataVar: vn & "_data", groupVar: groupVar, laneVar: laneVar,
                 elemsVar: vn & "_elems")

#[ --- Matrix expression code generation ---
   These procs generate C code that stores a matrix expression into a local array.
   The target array must already be declared.
   Each proc returns (code, elemsExpr) where elemsExpr is a C expression for
   the number of elements in the result (e.g. "NC*NC" for matrices, "NC" for vectors). ]#

type MatResult* = tuple[code: string, elems: string]

# Forward declarations
proc emitMatExpr*(target: string, n: NimNode, ctx: var CodeCtx, d: int): MatResult
proc emitLoadView*(target: string, sr: SiteRef, ctx: var CodeCtx, d: int): MatResult

proc emitLoadView*(target: string, sr: SiteRef, ctx: var CodeCtx, d: int): MatResult =
  ## Generate code to load all elements from a view site into `target` array.
  ## Uses the view's own _elems count (correct for both matrices and vectors).
  var s = ""
  let p = ind(d)
  let elems = sr.elemsVar  # e.g. "vplaq_elems" or "vu_elems"
  if sr.isGauge:
    # Runtime direction selection
    for di in 0..<sr.gaugeDim:
      let cond = if di == 0: "if" else: "else if"
      s &= p & cond & " (" & sr.dirExpr & " == " & $di & ") {\n"
      s &= p & "  for (int _e = 0; _e < " & elems & "; _e++)\n"
      s &= p & "    " & target & "[_e] = " &
           aosoaIdx(sr.gaugeName & "_" & $di & "_data", sr.groupVar, sr.laneVar, elems, "_e") & ";\n"
      s &= p & "}\n"
  else:
    s &= p & "for (int _e = 0; _e < " & elems & "; _e++)\n"
    s &= p & "  " & target & "[_e] = " &
         aosoaIdx(sr.dataVar, sr.groupVar, sr.laneVar, elems, "_e") & ";\n"
  return (s, elems)

proc emitMatMul(target, lhs, rhs, lhsElems, rhsElems: string, ctx: var CodeCtx, d: int): MatResult =
  ## Generate code: target = lhs * rhs (generalized matrix multiply)
  ## Handles mat*mat, mat*vec, vec*vec by computing cols from elems/NC.
  ## lhs is treated as (NC × lhs_cols), rhs as (lhs_cols × rhs_cols).
  ## Result has NC * rhs_cols elements.
  var s = ""
  let p = ind(d)
  let lcVar = ctx.nextTmp() & "_lc"
  let rcVar = ctx.nextTmp() & "_rc"
  s &= p & "const int " & lcVar & " = " & lhsElems & " / NC;\n"
  s &= p & "const int " & rcVar & " = " & rhsElems & " / NC;\n"
  s &= p & "for (int _i = 0; _i < NC; _i++) {\n"
  s &= p & "  for (int _j = 0; _j < " & rcVar & "; _j++) {\n"
  if ctx.isComplex:
    s &= p & "    " & ctx.vecType & " _s = (" & ctx.vecType & ")(0.0, 0.0);\n"
  else:
    s &= p & "    " & ctx.scalarType & " _s = 0.0;\n"
  s &= p & "    for (int _k = 0; _k < " & lcVar & "; _k++)\n"
  if ctx.isComplex:
    s &= p & "      _s += cmul(" & lhs & "[_i*" & lcVar & "+_k], " & rhs & "[_k*" & rcVar & "+_j]);\n"
  else:
    s &= p & "      _s += " & lhs & "[_i*" & lcVar & "+_k] * " & rhs & "[_k*" & rcVar & "+_j];\n"
  s &= p & "    " & target & "[_i*" & rcVar & "+_j] = _s;\n"
  s &= p & "  }\n"
  s &= p & "}\n"
  let resultElems = "NC * " & rcVar
  return (s, resultElems)

proc emitAdjoint(target, src, srcElems: string, ctx: var CodeCtx, d: int): MatResult =
  ## Generate code: target = adjoint(src) = conjugate transpose
  ## For NC×NC matrices: target[j*NC+i] = conj(src[i*NC+j])
  ## srcElems tells us how many elements src has.
  var s = ""
  let p = ind(d)
  let colsVar = ctx.nextTmp() & "_ac"
  s &= p & "const int " & colsVar & " = " & srcElems & " / NC;\n"
  s &= p & "for (int _i = 0; _i < NC; _i++)\n"
  s &= p & "  for (int _j = 0; _j < " & colsVar & "; _j++)\n"
  if ctx.isComplex:
    s &= p & "    " & target & "[_j*NC+_i] = cconj(" & src & "[_i*" & colsVar & "+_j]);\n"
  else:
    s &= p & "    " & target & "[_j*NC+_i] = " & src & "[_i*" & colsVar & "+_j];\n"
  # adjoint swaps rows/cols: result has colsVar * NC elements = srcElems (for square) 
  # For NC×M → M×NC: result elems = M * NC = srcElems
  return (s, srcElems)

proc emitMatExpr*(target: string, n: NimNode, ctx: var CodeCtx, d: int): MatResult =
  ## Generate C code that evaluates a matrix-valued expression and stores result in `target`.
  ## `target` is a pre-declared local array of size NC*NC (max possible).
  ## Returns (generated C code, element count expression).
  
  case n.kind
  of nnkCall:
    if n[0].kind == nnkSym:
      let fn = n[0].strVal
      
      if fn == "[]" and n.len >= 3:
        # view[site] — load from view buffer
        let sr = resolveSiteRef(n[1], n[2], ctx)
        return emitLoadView(target, sr, ctx, d)
      
      if fn == "adjoint" and n.len >= 2:
        # adjoint(expr) — evaluate inner, then conjugate transpose
        let tmp = ctx.nextTmp()
        let p = ind(d)
        var s = p & ctx.elemType & " " & tmp & "[NC*NC];\n"
        let inner = emitMatExpr(tmp, n[1], ctx, d)
        s &= inner.code
        let adj = emitAdjoint(target, tmp, inner.elems, ctx, d)
        s &= adj.code
        return (s, adj.elems)
    
    return (ind(d) & "// unhandled call: " & n[0].strVal & "\n", "NC*NC")
  
  of nnkSym:
    # Reference to a let-bound matrix temp
    let name = n.strVal
    let p = ind(d)
    let elemsName = name & "_elems"
    return (p & "for (int _e = 0; _e < " & elemsName & "; _e++) " & target & "[_e] = " & name & "[_e];\n",
            elemsName)
  
  of nnkIntLit..nnkInt64Lit:
    # Scalar integer literal — broadcast to all elements
    let p = ind(d)
    let val = $n.intVal
    return (p & "for (int _e = 0; _e < NC*NC; _e++) " & target & "[_e] = " & val & ";\n", "NC*NC")
  
  of nnkFloatLit..nnkFloat64Lit:
    # Scalar float literal — broadcast to all elements
    let p = ind(d)
    let val = $n.floatVal
    return (p & "for (int _e = 0; _e < NC*NC; _e++) " & target & "[_e] = " & val & ";\n", "NC*NC")
  
  of nnkInfix:
    if n.len >= 3 and n[0].kind == nnkSym:
      let op = n[0].strVal
      
      if op == "*":
        # Check if this is a matrix multiply (result type is MatMulResult)
        var isMM = false
        try: isMM = isMatMulSym(n)
        except: discard
        
        if isMM:
          let tmpL = ctx.nextTmp()
          let tmpR = ctx.nextTmp()
          let p = ind(d)
          var s = p & ctx.elemType & " " & tmpL & "[NC*NC];\n"
          s &= p & ctx.elemType & " " & tmpR & "[NC*NC];\n"
          let lRes = emitMatExpr(tmpL, n[1], ctx, d)
          s &= lRes.code
          let rRes = emitMatExpr(tmpR, n[2], ctx, d)
          s &= rRes.code
          let mmRes = emitMatMul(target, tmpL, tmpR, lRes.elems, rRes.elems, ctx, d)
          s &= mmRes.code
          return (s, mmRes.elems)
        else:
          # Element-wise scalar multiply
          let tmpL = ctx.nextTmp()
          let tmpR = ctx.nextTmp()
          let p = ind(d)
          var s = p & ctx.elemType & " " & tmpL & "[NC*NC];\n"
          s &= p & ctx.elemType & " " & tmpR & "[NC*NC];\n"
          let lRes = emitMatExpr(tmpL, n[1], ctx, d)
          s &= lRes.code
          let rRes = emitMatExpr(tmpR, n[2], ctx, d)
          s &= rRes.code
          # Use the output elems (should be same for both operands)
          let outElems = lRes.elems
          s &= p & "for (int _e = 0; _e < " & outElems & "; _e++)\n"
          if ctx.isComplex:
            s &= p & "  " & target & "[_e] = cmul(" & tmpL & "[_e], " & tmpR & "[_e]);\n"
          else:
            s &= p & "  " & target & "[_e] = " & tmpL & "[_e] * " & tmpR & "[_e];\n"
          return (s, outElems)
      
      if op == "+" or op == "-":
        let tmpL = ctx.nextTmp()
        let tmpR = ctx.nextTmp()
        let p = ind(d)
        var s = p & ctx.elemType & " " & tmpL & "[NC*NC];\n"
        s &= p & ctx.elemType & " " & tmpR & "[NC*NC];\n"
        let lRes = emitMatExpr(tmpL, n[1], ctx, d)
        s &= lRes.code
        let rRes = emitMatExpr(tmpR, n[2], ctx, d)
        s &= rRes.code
        let outElems = lRes.elems
        s &= p & "for (int _e = 0; _e < " & outElems & "; _e++)\n"
        s &= p & "  " & target & "[_e] = " & tmpL & "[_e] " & op & " " & tmpR & "[_e];\n"
        return (s, outElems)
  
  of nnkHiddenCallConv:
    if n.len >= 2:
      return emitMatExpr(target, n[1], ctx, d)
  
  of nnkHiddenDeref, nnkHiddenAddr, nnkHiddenStdConv, nnkConv:
    if n.len > 0:
      return emitMatExpr(target, n[^1], ctx, d)
  
  else:
    discard
  
  return (ind(d) & "// unhandled expr kind: " & $n.kind & "\n", "NC*NC")

#[ --- Scalar expression transpilation (for element-level writes and scalars) --- ]#

proc transpileScalar*(n: NimNode, ctx: var CodeCtx): string =
  ## Transpile a scalar expression to C code
  case n.kind
  of nnkSym:
    return n.strVal
  of nnkIntLit..nnkInt64Lit:
    return $n.intVal
  of nnkFloatLit..nnkFloat64Lit:
    return $n.floatVal
  of nnkHiddenStdConv, nnkConv, nnkHiddenDeref, nnkHiddenAddr:
    if n.len > 0: return transpileScalar(n[^1], ctx)
  of nnkCall:
    if n[0].kind == nnkSym:
      var args: seq[string]
      for i in 1..<n.len: args.add transpileScalar(n[i], ctx)
      return n[0].strVal & "(" & args.join(", ") & ")"
  of nnkInfix:
    if n.len >= 3 and n[0].kind == nnkSym:
      return "(" & transpileScalar(n[1], ctx) & " " & n[0].strVal & " " & transpileScalar(n[2], ctx) & ")"
  else: discard
  return "0"

#[ ============================================================================
   Kernel Source Assembly
   ============================================================================ ]#

proc generateKernelSource(kernelName: string, body: NimNode, info: KernelInfo): string =
  var ctx = newCodeCtx(info)
  let vw = $VectorWidth
  var src = ""

  # FP64 extension (only needed for double precision)
  if info.scalarType in {etFloat64, etInt64}:
    src &= "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n"
  # NC is a compile-time constant (replaced at runtime before kernel build)
  src &= "#define NC {NC_VALUE}\n\n"

  # Complex helpers
  if info.isComplex:
    let vt = ctx.vecType
    src &= "inline " & vt & " cmul(" & vt & " a, " & vt & " b) {\n"
    src &= "  return (" & vt & ")(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);\n"
    src &= "}\n"
    src &= "inline " & vt & " cconj(" & vt & " a) {\n"
    src &= "  return (" & vt & ")(a.x, -a.y);\n"
    src &= "}\n\n"

  # Build parameter list
  var params: seq[string]
  # View buffers
  for v in info.views:
    params.add "__global " & ctx.elemType & "* " & v.name & "_data"
  # Gauge view buffers (D per gauge view)
  for gv in info.gaugeViews:
    for d in 0..<gv.dim:
      params.add "__global " & ctx.elemType & "* " & gv.name & "_" & $d & "_data"
  # Per-view elems counts
  for v in info.views:
    params.add "const int " & v.name & "_elems"
  for gv in info.gaugeViews:
    params.add "const int " & gv.name & "_elems"
  # Standard params
  params.add "const int numSites"
  # NC is a #define, not a param (OpenCL C doesn't support VLAs)
  # Stencil params
  for s in info.stencils:
    params.add "__global const int* " & s.name & "_offsets"
    params.add "const int " & s.name & "_npts"
  # Runtime int vars
  let runtimeVars = findRuntimeIntVars(body, info)
  for rv in runtimeVars:
    params.add "const int " & rv.name

  src &= "__kernel void " & kernelName & "(\n"
  src &= "    " & params.join(",\n    ")
  src &= "\n) {\n"

  # Work-item setup
  src &= "  const int " & ctx.loopVarStr & " = get_global_id(0);\n"
  src &= "  if (" & ctx.loopVarStr & " >= numSites) return;\n\n"
  src &= "  const int VW = " & vw & ";\n"
  src &= "  const int group = " & ctx.loopVarStr & " / VW;\n"
  src &= "  const int lane = " & ctx.loopVarStr & " % VW;\n\n"

  # Process each statement
  var stmts: seq[NimNode]
  if body.kind == nnkStmtList:
    for child in body: stmts.add child
  else:
    stmts.add body

  for stmt in stmts:
    case stmt.kind
    of nnkLetSection:
      for idefs in stmt:
        if idefs.kind == nnkIdentDefs and idefs.len >= 3:
          let vn = idefs[0].strVal
          let val = idefs[2]
          let lb = info.getLetBinding(vn)

          case lb.kind
          of lbkStencilFwd:
            src &= "  // fwd neighbor: " & vn & "\n"
            src &= "  int " & vn & "_idx = " & lb.stencilName & "_offsets[" &
                   ctx.loopVarStr & " * " & lb.stencilName & "_npts + 2 * " & lb.dirExpr & "];\n"
            src &= "  int " & vn & "_group = " & vn & "_idx / VW;\n"
            src &= "  int " & vn & "_lane = " & vn & "_idx % VW;\n\n"
          of lbkStencilBwd:
            src &= "  // bwd neighbor: " & vn & "\n"
            src &= "  int " & vn & "_idx = " & lb.stencilName & "_offsets[" &
                   ctx.loopVarStr & " * " & lb.stencilName & "_npts + 2 * " & lb.dirExpr & " + 1];\n"
            src &= "  int " & vn & "_group = " & vn & "_idx / VW;\n"
            src &= "  int " & vn & "_lane = " & vn & "_idx % VW;\n\n"
          of lbkStencilNeighbor:
            src &= "  // stencil neighbor: " & vn & "\n"
            src &= "  int " & vn & "_idx = " & lb.stencilName & "_offsets[" &
                   ctx.loopVarStr & " * " & lb.stencilName & "_npts + " & lb.pointExpr & "];\n"
            src &= "  int " & vn & "_group = " & vn & "_idx / VW;\n"
            src &= "  int " & vn & "_lane = " & vn & "_idx % VW;\n\n"
          of lbkMatMul:
            # Matrix-valued let binding: let ta = expr
            src &= "  // matrix temp: " & vn & "\n"
            src &= "  " & ctx.elemType & " " & vn & "[NC*NC];\n"
            let matRes = emitMatExpr(vn, val, ctx, 1)
            src &= matRes.code
            src &= "  const int " & vn & "_elems = " & matRes.elems & ";\n"
            src &= "\n"
          of lbkOther:
            let code = transpileScalar(val, ctx)
            src &= "  " & ctx.elemType & " " & vn & " = " & code & ";\n"

    of nnkInfix:
      if stmt.len >= 3 and stmt[0].kind == nnkSym and stmt[0].strVal == "+=":
        # vplaq[n] += expr
        let lhs = stmt[1]  # Call("[]", view, site)
        let rhs = stmt[2]
        if lhs.kind == nnkCall and lhs[0].kind == nnkSym and lhs[0].strVal == "[]":
          let sr = resolveSiteRef(lhs[1], lhs[2], ctx)
          let tmp = ctx.nextTmp()
          src &= "  { // +=\n"
          src &= "    " & ctx.elemType & " " & tmp & "[NC*NC];\n"
          let matRes = emitMatExpr(tmp, rhs, ctx, 2)
          src &= matRes.code
          let storeElems = sr.elemsVar
          if sr.isGauge:
            for di in 0..<sr.gaugeDim:
              let cond = if di == 0: "if" else: "else if"
              src &= "    " & cond & " (" & sr.dirExpr & " == " & $di & ") {\n"
              src &= "      for (int _e = 0; _e < " & storeElems & "; _e++)\n"
              src &= "        " & aosoaIdx(sr.gaugeName & "_" & $di & "_data", sr.groupVar, sr.laneVar, sr.elemsVar, "_e") & " += " & tmp & "[_e];\n"
              src &= "    }\n"
          else:
            src &= "    for (int _e = 0; _e < " & storeElems & "; _e++)\n"
            src &= "      " & aosoaIdx(sr.dataVar, sr.groupVar, sr.laneVar, sr.elemsVar, "_e") & " += " & tmp & "[_e];\n"
          src &= "  }\n"

    of nnkCall:
      if stmt.len >= 2 and stmt[0].kind == nnkSym and stmt[0].strVal == "[]=":
        # view[n] = expr
        if isElementLevelWrite(stmt):
          # Element-level write: view[n][i,j] = val
          let innerCall = stmt[1]
          var viewName = "output"
          if innerCall[1].kind == nnkSym:
            viewName = innerCall[1].strVal

          if stmt.len == 4:
            # 1D: view[n][idx] = val
            let idxCode = transpileScalar(stmt[2], ctx)
            let valCode = transpileScalar(stmt[3], ctx)
            src &= "  " & aosoaIdx(viewName & "_data", "group", "lane", viewName & "_elems", idxCode) & " = " & valCode & ";\n"
          elif stmt.len >= 5:
            # 2D: view[n][row, col] = val
            let rowCode = transpileScalar(stmt[2], ctx)
            let colCode = transpileScalar(stmt[3], ctx)
            let valCode = transpileScalar(stmt[4], ctx)
            let flatIdx = "(" & rowCode & ")*NC+(" & colCode & ")"
            src &= "  " & aosoaIdx(viewName & "_data", "group", "lane", viewName & "_elems", flatIdx) & " = " & valCode & ";\n"
        else:
          # Tensor-level: view[n] = matrix_expr
          let viewNode = stmt[1]
          let siteNode = stmt[2]
          let rhsNode = stmt[3]
          let sr = resolveSiteRef(viewNode, siteNode, ctx)
          
          # Generate the RHS matrix expr into a temp
          let tmp = ctx.nextTmp()
          src &= "  { // assign\n"
          src &= "    " & ctx.elemType & " " & tmp & "[NC*NC];\n"
          let matRes = emitMatExpr(tmp, rhsNode, ctx, 2)
          src &= matRes.code
          let storeElems = sr.elemsVar
          if sr.isGauge:
            for di in 0..<sr.gaugeDim:
              let cond = if di == 0: "if" else: "else if"
              src &= "    " & cond & " (" & sr.dirExpr & " == " & $di & ") {\n"
              src &= "      for (int _e = 0; _e < " & storeElems & "; _e++)\n"
              src &= "        " & aosoaIdx(sr.gaugeName & "_" & $di & "_data", sr.groupVar, sr.laneVar, sr.elemsVar, "_e") & " = " & tmp & "[_e];\n"
              src &= "    }\n"
          else:
            src &= "    for (int _e = 0; _e < " & storeElems & "; _e++)\n"
            src &= "      " & aosoaIdx(sr.dataVar, sr.groupVar, sr.laneVar, sr.elemsVar, "_e") & " = " & tmp & "[_e];\n"
          src &= "  }\n"

    else:
      src &= "  // skipped: " & $stmt.kind & "\n"

  src &= "}\n"
  return src

#[ ============================================================================
   The `each` Macro — Main Entry Point
   ============================================================================ ]#

macro eachImpl*(loopVar: untyped, lo: typed, hi: typed, body: typed): untyped =
  when DebugKernels:
    echo "=== AST TREE REPR ==="
    echo body.treeRepr
    echo "=== END AST TREE REPR ==="

  let loopVarSym = loopVar
  let info = gatherInfo(body, loopVarSym)

  # CPU fallback for echo/debugEcho
  if hasEchoStatement(body):
    result = quote do:
      block:
        for cpuIdx in `lo`..<`hi`:
          `loopVarSym` = cpuIdx
          `body`
    return result

  if info.views.len == 0 and info.gaugeViews.len == 0:
    error("No TensorFieldView found in loop body. The each macro requires at least one view access.")

  # Generate kernel source
  let kernelName = "tfv_kernel_" & $body.lineInfoObj.line
  let kernelSource = generateKernelSource(kernelName, body, info)
  let kernelNameLit = newLit(kernelName)
  let kernelSourceLit = newLit(kernelSource)

  # Find a view sym for shape/sites info
  var outViewSym: NimNode = nil
  for v in info.views:
    if v.isWrite:
      outViewSym = v.nimSym; break
  if outViewSym == nil and info.views.len > 0:
    outViewSym = info.views[0].nimSym

  # For shape: if only gauge views, use field[0]
  var shapeExpr: NimNode
  if outViewSym != nil:
    shapeExpr = outViewSym
  elif info.gaugeViews.len > 0:
    shapeExpr = nnkBracketExpr.newTree(
      nnkDotExpr.newTree(info.gaugeViews[0].nimSym, ident"field"), newLit(0))
  else:
    error("No view found for shape info")
    return

  var sitesExpr: NimNode
  if outViewSym != nil:
    sitesExpr = outViewSym
  elif info.gaugeViews.len > 0:
    sitesExpr = nnkBracketExpr.newTree(
      nnkDotExpr.newTree(info.gaugeViews[0].nimSym, ident"field"), newLit(0))
  else:
    sitesExpr = shapeExpr

  let kernelSym = genSym(nskLet, "kernel")
  let programSym = genSym(nskLet, "program")
  let devIdxSym = genSym(nskForVar, "devIdx")

  # --- Build setArg statements ---
  var setArgsStmts = newStmtList()
  var argIndex = 0

  # Regular view buffers
  for v in info.views:
    let idx = newLit(argIndex)
    let buf = nnkBracketExpr.newTree(
      nnkDotExpr.newTree(nnkDotExpr.newTree(v.nimSym, ident"data"), ident"buffers"),
      devIdxSym)
    setArgsStmts.add nnkCall.newTree(
      nnkDotExpr.newTree(kernelSym, ident"setArg"), buf, idx)
    argIndex += 1

  # Gauge view buffers
  for gv in info.gaugeViews:
    for d in 0..<gv.dim:
      let idx = newLit(argIndex)
      let buf = nnkBracketExpr.newTree(
        nnkDotExpr.newTree(
          nnkDotExpr.newTree(
            nnkBracketExpr.newTree(
              nnkDotExpr.newTree(gv.nimSym, ident"field"), newLit(d)),
            ident"data"),
          ident"buffers"),
        devIdxSym)
      setArgsStmts.add nnkCall.newTree(
        nnkDotExpr.newTree(kernelSym, ident"setArg"), buf, idx)
      argIndex += 1

  # Per-view elems
  var elemsStmts = newStmtList()
  for v in info.views:
    let idx = newLit(argIndex)
    let sym = v.nimSym
    let tmp = genSym(nskVar, "elems")
    if info.isComplex:
      elemsStmts.add quote do:
        var `tmp` = `sym`.data.tensorElementsPerSite.int32
        `kernelSym`.setArg(`tmp`, `idx`)
    else:
      elemsStmts.add quote do:
        var `tmp` = `sym`.data.elementsPerSite.int32
        `kernelSym`.setArg(`tmp`, `idx`)
    argIndex += 1

  for gv in info.gaugeViews:
    let idx = newLit(argIndex)
    let gsym = gv.nimSym
    let tmp = genSym(nskVar, "gelems")
    if info.isComplex:
      elemsStmts.add quote do:
        var `tmp` = `gsym`.field[0].data.tensorElementsPerSite.int32
        `kernelSym`.setArg(`tmp`, `idx`)
    else:
      elemsStmts.add quote do:
        var `tmp` = `gsym`.field[0].data.elementsPerSite.int32
        `kernelSym`.setArg(`tmp`, `idx`)
    argIndex += 1

  # numSites (NC is a #define, not an arg)
  let numSitesIdx = newLit(argIndex)
  argIndex += 1

  # Stencil args
  var stencilSetup = newStmtList()
  var stencilArgs = newStmtList()
  var stencilCleanup = newStmtList()

  for s in info.stencils:
    let ssym = s.nimSym
    let offIdx = newLit(argIndex)
    let npIdx = newLit(argIndex + 1)
    argIndex += 2

    let sbuf = genSym(nskVar, "sbuf")
    let scopy = genSym(nskVar, "scopy")

    stencilSetup.add quote do:
      var `scopy` = newSeq[int32](`ssym`.offsets.len)
      for i in 0..<`ssym`.offsets.len:
        `scopy`[i] = `ssym`.offsets[i]
      var `sbuf` = gpuBufferLike(clContext, `scopy`, MEM_READ_ONLY)
      clQueues[0].write(`scopy`, `sbuf`)
      check clwrap.finish(clQueues[0])

    stencilArgs.add quote do:
      `kernelSym`.setArg(`sbuf`, `offIdx`)
      var np = `ssym`.nPoints.int32
      `kernelSym`.setArg(np, `npIdx`)

    stencilCleanup.add quote do:
      release(`sbuf`)

  # Runtime int variable args
  let runtimeVars = findRuntimeIntVars(body, info)
  var runtimeArgStmts = newStmtList()
  for rv in runtimeVars:
    let idx = newLit(argIndex)
    let rvsym = rv.sym
    let tmp = genSym(nskVar, "rv")
    runtimeArgStmts.add quote do:
      var `tmp` = `rvsym`.int32
      `kernelSym`.setArg(`tmp`, `idx`)
    argIndex += 1

  result = quote do:
    block:
      let outShape = `shapeExpr`.shape
      let outRank = outShape.len
      let ncDim = if outRank >= 1: outShape[0] else: 1
      let finalKernelSource = `kernelSourceLit`.replace("{NC_VALUE}", $ncDim)

      when DebugKernels:
        echo "=== Generated OpenCL Kernel ==="
        echo finalKernelSource
        echo "================================"

      let `programSym` = createAndBuild(clContext, finalKernelSource, clDevices)
      let `kernelSym` = `programSym`.createKernel(`kernelNameLit`)

      let numDevices = clQueues.len
      let sitesPerDev = `sitesExpr`.data.sitesPerDevice

      `stencilSetup`

      for `devIdxSym` in 0..<numDevices:
        let devSites = sitesPerDev[`devIdxSym`]
        if devSites > 0:
          `setArgsStmts`
          `elemsStmts`

          var nsArg = devSites.int32
          `kernelSym`.setArg(nsArg, `numSitesIdx`)

          `stencilArgs`
          `runtimeArgStmts`

          when UseWorkGroups:
            clQueues[`devIdxSym`].run(`kernelSym`, devSites, VectorWidth)
          else:
            clQueues[`devIdxSym`].run(`kernelSym`, devSites)

      for devIdx in 0..<numDevices:
        check clwrap.finish(clQueues[devIdx])

      release(`kernelSym`)
      release(`programSym`)

      `stencilCleanup`

macro each*(x: ForLoopStmt): untyped =
  expectLen(x, 3)
  let loopVarNode = x[0]
  let callNode = x[1]
  let bodyNode = x[2]
  let iterNode = callNode[1]

  var lo, hi: NimNode
  if iterNode.kind == nnkInfix:
    lo = iterNode[1]
    hi = iterNode[2]
  elif iterNode.kind == nnkDotExpr and iterNode[1].eqIdent("all"):
    lo = newLit(0)
    hi = newCall(ident"numSites", iterNode[0])
  else:
    error("each requires a range expression like 0..<N or obj.all")

  result = quote do:
    block:
      var `loopVarNode` {.inject.}: int = 0
      eachImpl(`loopVarNode`, `lo`, `hi`, `bodyNode`)
