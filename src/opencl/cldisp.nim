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

import ../ir/ir
export ir

const UseWorkGroups* {.booldefine.} = false

proc elementTypeToOpenCL*(et: ElementType): string =
  case et
  of etFloat32: "float"
  of etFloat64: "double"
  of etInt32: "int"
  of etInt64: "long"

#[ ============================================================================
   OpenCL C Code Generation Context
   ============================================================================ ]#

type
  ClCodeCtx* = object
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

proc newClCodeCtx*(info: KernelInfo): ClCodeCtx =
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

proc nextTmp*(ctx: var ClCodeCtx): string =
  result = "_t" & $ctx.tmpIdx
  ctx.tmpIdx += 1

#[ --- AoSoA and SiteRef: reuse from ir.nim, adapt for ClCodeCtx --- ]#

proc clResolveSiteRef*(viewExpr, siteExpr: NimNode, ctx: var ClCodeCtx): SiteRef =
  ## Resolve a view[site] access via the shared IR resolver
  var irCtx = CodeCtx(loopVarStr: ctx.loopVarStr, isComplex: ctx.isComplex,
                       scalarType: ctx.scalarType, elemType: ctx.elemType,
                       vecType: ctx.vecType,
                       viewNames: ctx.viewNames, gaugeNames: ctx.gaugeNames,
                       stencilNames: ctx.stencilNames, letNames: ctx.letNames,
                       info: ctx.info, tmpIdx: ctx.tmpIdx)
  result = resolveSiteRef(viewExpr, siteExpr, irCtx)
  ctx.tmpIdx = irCtx.tmpIdx

proc clTranspileScalar*(n: NimNode, ctx: var ClCodeCtx): string =
  ## Transpile a scalar expression via the shared IR transpiler
  var irCtx = CodeCtx(loopVarStr: ctx.loopVarStr, isComplex: ctx.isComplex,
                       scalarType: ctx.scalarType, elemType: ctx.elemType,
                       vecType: ctx.vecType,
                       viewNames: ctx.viewNames, gaugeNames: ctx.gaugeNames,
                       stencilNames: ctx.stencilNames, letNames: ctx.letNames,
                       info: ctx.info, tmpIdx: ctx.tmpIdx)
  result = transpileScalar(n, irCtx)
  ctx.tmpIdx = irCtx.tmpIdx

#[ --- Matrix expression code generation ---
   These procs generate C code that stores a matrix expression into a local array.
   The target array must already be declared.
   Each proc returns (code, elemsExpr) where elemsExpr is a C expression for
   the number of elements in the result (e.g. "NC*NC" for matrices, "NC" for vectors). ]#

type MatResult* = tuple[code: string, elems: string]

# Forward declarations
proc emitMatExpr*(target: string, n: NimNode, ctx: var ClCodeCtx, d: int): MatResult
proc emitLoadView*(target: string, sr: SiteRef, ctx: var ClCodeCtx, d: int): MatResult
proc clTranspileStmt(stmt: NimNode, ctx: var ClCodeCtx, info: KernelInfo, d: int): string

proc emitLoadView*(target: string, sr: SiteRef, ctx: var ClCodeCtx, d: int): MatResult =
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

proc emitMatMul(target, lhs, rhs, lhsElems, rhsElems: string, ctx: var ClCodeCtx, d: int): MatResult =
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

proc emitAdjoint(target, src, srcElems: string, ctx: var ClCodeCtx, d: int): MatResult =
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

proc emitMatExpr*(target: string, n: NimNode, ctx: var ClCodeCtx, d: int): MatResult =
  ## Generate C code that evaluates a matrix-valued expression and stores result in `target`.
  ## `target` is a pre-declared local array of size NC*NC (max possible).
  ## Returns (generated C code, element count expression).
  
  case n.kind
  of nnkCall:
    if n[0].kind == nnkSym:
      let fn = n[0].strVal
      
      if fn == "[]" and n.len >= 3:
        # Check if this is an element read on a local matrix temp
        let refNode = n[1]
        if refNode.kind == nnkSym:
          var isMat = false
          try: isMat = isMatrixTypedNode(refNode)
          except: discard
          if isMat:
            # Element read: m[i] or m[i,j] → scalar result
            let name = refNode.strVal
            let p = ind(d)
            if n.len == 3:
              let idx = clTranspileScalar(n[2], ctx)
              return (p & target & "[0] = " & name & "[" & idx & "];\n", "1")
            elif n.len == 4:
              let row = clTranspileScalar(n[2], ctx)
              let col = clTranspileScalar(n[3], ctx)
              return (p & target & "[0] = " & name & "[(" & row & ")*NC+(" & col & ")];\n", "1")
        # view[site] — load from view buffer
        let sr = clResolveSiteRef(n[1], n[2], ctx)
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

      if fn == "trace" and n.len >= 2:
        # trace(matexpr) — evaluate inner matrix, sum diagonal → scalar
        let tmp = ctx.nextTmp()
        let p = ind(d)
        var s = p & ctx.elemType & " " & tmp & "[NC*NC];\n"
        let inner = emitMatExpr(tmp, n[1], ctx, d)
        s &= inner.code
        # Sum diagonal: target[0] = sum of tmp[i*NC+i] for i in 0..NC-1
        if ctx.isComplex:
          s &= p & target & "[0] = (" & ctx.elemType & ")(0, 0);\n"
          s &= p & "for (int _i = 0; _i < NC; _i++)\n"
          s &= p & "  " & target & "[0] += " & tmp & "[_i*NC+_i];\n"
        else:
          s &= p & target & "[0] = 0;\n"
          s &= p & "for (int _i = 0; _i < NC; _i++)\n"
          s &= p & "  " & target & "[0] += " & tmp & "[_i*NC+_i];\n"
        return (s, "1")

      if fn == "identity" or fn == "siteIdentity":
        # identity()/siteIdentity() — generate NC×NC identity matrix
        let p = ind(d)
        var s = ""
        s &= p & "for (int _e = 0; _e < NC*NC; _e++) " & target & "[_e] = "
        if ctx.isComplex:
          s &= "(" & ctx.elemType & ")(0.0, 0.0);\n"
          s &= p & "for (int _i = 0; _i < NC; _i++) " & target & "[_i*NC+_i] = (" & ctx.elemType & ")(1.0, 0.0);\n"
        else:
          s &= "0.0;\n"
          s &= p & "for (int _i = 0; _i < NC; _i++) " & target & "[_i*NC+_i] = 1.0;\n"
        return (s, "NC*NC")

      # --- Custom site operation dispatch ---
      let opIdx = lookupCustomOp(fn)
      if opIdx >= 0:
        let op = customSiteOps[opIdx]
        let p = ind(d)
        var s = ""
        var args: seq[string]
        var argElems: seq[string]
        for i in 1..op.arity:
          let argTmp = ctx.nextTmp()
          args.add argTmp
          s &= p & ctx.elemType & " " & argTmp & "[NC*NC];\n"
          let argRes = emitMatExpr(argTmp, n[i], ctx, d)
          s &= argRes.code
          argElems.add argRes.elems
        s &= instantiateTemplate(op.codeTemplate, target, args, argElems,
                                  ctx.elemType, op.resultElems, d)
        return (s, op.resultElems)
    
    return (ind(d) & "// unhandled call: " & n[0].strVal & "\n", "NC*NC")
  
  of nnkSym:
    let name = n.strVal
    let p = ind(d)
    # Check if this is a matrix-typed let binding (has _elems companion)
    var isMatrix = false
    try: isMatrix = isMatrixTypedNode(n)
    except: discard
    if isMatrix:
      # Reference to a let-bound matrix temp or var proxy
      let elemsName = name & "_elems"
      return (p & "for (int _e = 0; _e < " & elemsName & "; _e++) " & target & "[_e] = " & name & "[_e];\n",
              elemsName)
    else:
      # Scalar sym (float/int kernel parameter or local scalar) — broadcast
      if ctx.isComplex:
        return (p & target & "[0] = (" & ctx.elemType & ")(" & name & ", 0);\n", "1")
      else:
        return (p & target & "[0] = " & name & ";\n", "1")
  
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
        # Detect matrix multiply by checking both operands are matrix-typed
        var lhsIsMat = false
        var rhsIsMat = false
        try: lhsIsMat = isMatrixTypedNode(n[1])
        except: discard
        try: rhsIsMat = isMatrixTypedNode(n[2])
        except: discard
        
        if lhsIsMat and rhsIsMat:
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

  of nnkStmtListExpr:
    # Template expansion: process statements, then emit result expression
    if n.len >= 2:
      var s = ""
      for ci in 0..<n.len - 1:
        s &= clTranspileStmt(n[ci], ctx, ctx.info, d)
      let resultExpr = n[n.len - 1]
      let matRes = emitMatExpr(target, resultExpr, ctx, d)
      s &= matRes.code
      return (s, matRes.elems)
    elif n.len == 1:
      return emitMatExpr(target, n[0], ctx, d)
  
  else: discard
  
  return (ind(d) & "// unhandled expr kind: " & $n.kind & "\n", "NC*NC")

#[ ============================================================================
   Recursive Statement Dispatch
   ============================================================================ ]#

proc clTranspileStmt(stmt: NimNode, ctx: var ClCodeCtx, info: KernelInfo, d: int): string =
  ## Transpile a single statement inside an ``each`` loop body to OpenCL C.
  ## `d` is the indentation depth.  Called recursively for nested blocks
  ## (if-bodies, for-bodies, etc.).
  let p = ind(d)
  case stmt.kind
  of nnkLetSection:
    var s = ""
    for idefs in stmt:
      if idefs.kind == nnkIdentDefs and idefs.len >= 3:
        let vn = idefs[0].strVal
        let val = idefs[2]
        let lb = info.getLetBinding(vn)

        case lb.kind
        of lbkStencilFwd:
          s &= p & "// fwd neighbor: " & vn & "\n"
          s &= p & "int " & vn & "_idx = " & lb.stencilName & "_offsets[" &
                 ctx.loopVarStr & " * " & lb.stencilName & "_npts + 2 * " & lb.dirExpr & "];\n"
          s &= p & "int " & vn & "_group = " & vn & "_idx / VW;\n"
          s &= p & "int " & vn & "_lane = " & vn & "_idx % VW;\n\n"
        of lbkStencilBwd:
          s &= p & "// bwd neighbor: " & vn & "\n"
          s &= p & "int " & vn & "_idx = " & lb.stencilName & "_offsets[" &
                 ctx.loopVarStr & " * " & lb.stencilName & "_npts + 2 * " & lb.dirExpr & " + 1];\n"
          s &= p & "int " & vn & "_group = " & vn & "_idx / VW;\n"
          s &= p & "int " & vn & "_lane = " & vn & "_idx % VW;\n\n"
        of lbkStencilNeighbor:
          s &= p & "// stencil neighbor: " & vn & "\n"
          s &= p & "int " & vn & "_idx = " & lb.stencilName & "_offsets[" &
                 ctx.loopVarStr & " * " & lb.stencilName & "_npts + " & lb.pointExpr & "];\n"
          s &= p & "int " & vn & "_group = " & vn & "_idx / VW;\n"
          s &= p & "int " & vn & "_lane = " & vn & "_idx % VW;\n\n"
        of lbkStencilCorner:
          let nd = $lb.nDim
          let sA = lb.signExprA
          let dA = lb.dirExprA
          let sB = lb.signExprB
          let dB = lb.dirExprB
          let prefix = "_c_" & vn
          s &= p & "// corner neighbor: " & vn & " (" & sA & "*" & dA & ", " & sB & "*" & dB & ")\n"
          s &= p & "int " & prefix & "_a = (" & dA & " < " & dB & ") ? (" & dA & ") : (" & dB & ");\n"
          s &= p & "int " & prefix & "_b = (" & dA & " < " & dB & ") ? (" & dB & ") : (" & dA & ");\n"
          s &= p & "int " & prefix & "_pair = " & prefix & "_a * (2*" & nd & " - 1 - " & prefix & "_a) / 2 + " & prefix & "_b - " & prefix & "_a - 1;\n"
          s &= p & "int " & prefix & "_sA = (" & dA & " <= " & dB & ") ? (" & sA & ") : (" & sB & ");\n"
          s &= p & "int " & prefix & "_sB = (" & dA & " <= " & dB & ") ? (" & sB & ") : (" & sA & ");\n"
          s &= p & "int " & prefix & "_si = (" & prefix & "_sA > 0 ? 0 : 2) + (" & prefix & "_sB > 0 ? 0 : 1);\n"
          s &= p & "int " & vn & "_idx = " & lb.stencilName & "_offsets[" &
                 ctx.loopVarStr & " * " & lb.stencilName & "_npts + 2*" & nd & " + " & prefix & "_pair * 4 + " & prefix & "_si];\n"
          s &= p & "int " & vn & "_group = " & vn & "_idx / VW;\n"
          s &= p & "int " & vn & "_lane = " & vn & "_idx % VW;\n\n"
        of lbkMatMul:
          s &= p & "// matrix temp: " & vn & "\n"
          s &= p & ctx.elemType & " " & vn & "[NC*NC];\n"
          if val.kind == nnkStmtListExpr and val.len >= 2:
            # Template expansion: process statements, then copy result
            for ci in 0..<val.len - 1:
              s &= clTranspileStmt(val[ci], ctx, info, d)
            let resultExpr = val[val.len - 1]
            let matRes = emitMatExpr(vn, resultExpr, ctx, d)
            s &= matRes.code
            s &= p & "const int " & vn & "_elems = " & matRes.elems & ";\n\n"
          elif val.kind != nnkEmpty:
            let matRes = emitMatExpr(vn, val, ctx, d)
            s &= matRes.code
            s &= p & "const int " & vn & "_elems = " & matRes.elems & ";\n\n"
          else:
            s &= p & "const int " & vn & "_elems = NC*NC;\n\n"
        of lbkOther:
          let code = clTranspileScalar(val, ctx)
          s &= p & ctx.elemType & " " & vn & " = " & code & ";\n"
    return s

  of nnkVarSection:
    var s = ""
    for idefs in stmt:
      if idefs.kind == nnkIdentDefs and idefs.len >= 3:
        let vn = idefs[0].strVal
        let val = idefs[2]
        var isProxy = false
        try: isProxy = isTensorSiteProxySym(idefs[0])
        except: discard
        if isProxy:
          s &= p & "// var matrix: " & vn & "\n"
          s &= p & ctx.elemType & " " & vn & "[NC*NC];\n"
          s &= p & "const int " & vn & "_elems = NC*NC;\n"
          if val.kind == nnkStmtListExpr and val.len >= 2:
            # Template expansion: process statements, then copy result
            for ci in 0..<val.len - 1:
              s &= clTranspileStmt(val[ci], ctx, info, d)
            let resultExpr = val[val.len - 1]
            let matRes = emitMatExpr(vn, resultExpr, ctx, d)
            s &= matRes.code
          elif val.kind != nnkEmpty:
            let matRes = emitMatExpr(vn, val, ctx, d)
            s &= matRes.code
          s &= "\n"
        else:
          let code = clTranspileScalar(val, ctx)
          s &= p & ctx.elemType & " " & vn & " = " & code & ";\n"
    return s

  of nnkAsgn:
    if stmt.len >= 2:
      let lhs = stmt[0]
      let rhs = stmt[1]
      if lhs.kind == nnkSym:
        let vn = lhs.strVal
        var isProxy = false
        try: isProxy = isTensorSiteProxySym(lhs)
        except: discard
        if isProxy:
          var s = p & "{ // assign to var matrix: " & vn & "\n"
          let matRes = emitMatExpr(vn, rhs, ctx, d+1)
          s &= matRes.code
          s &= p & "}\n"
          return s
        else:
          let code = clTranspileScalar(rhs, ctx)
          return p & vn & " = " & code & ";\n"
    return ""

  of nnkInfix:
    if stmt.len >= 3 and stmt[0].kind == nnkSym and stmt[0].strVal == "+=":
      let lhs = stmt[1]
      let rhs = stmt[2]
      if lhs.kind == nnkCall and lhs[0].kind == nnkSym and lhs[0].strVal == "[]":
        let sr = clResolveSiteRef(lhs[1], lhs[2], ctx)
        let tmp = ctx.nextTmp()
        var s = p & "{ // +=\n"
        s &= ind(d+1) & ctx.elemType & " " & tmp & "[NC*NC];\n"
        let matRes = emitMatExpr(tmp, rhs, ctx, d+1)
        s &= matRes.code
        let storeElems = sr.elemsVar
        if sr.isGauge:
          for di in 0..<sr.gaugeDim:
            let cond = if di == 0: "if" else: "else if"
            s &= ind(d+1) & cond & " (" & sr.dirExpr & " == " & $di & ") {\n"
            s &= ind(d+2) & "for (int _e = 0; _e < " & storeElems & "; _e++)\n"
            s &= ind(d+3) & aosoaIdx(sr.gaugeName & "_" & $di & "_data", sr.groupVar, sr.laneVar, sr.elemsVar, "_e") & " += " & tmp & "[_e];\n"
            s &= ind(d+1) & "}\n"
        else:
          s &= ind(d+1) & "for (int _e = 0; _e < " & storeElems & "; _e++)\n"
          s &= ind(d+2) & aosoaIdx(sr.dataVar, sr.groupVar, sr.laneVar, sr.elemsVar, "_e") & " += " & tmp & "[_e];\n"
        s &= p & "}\n"
        return s
    return ""

  of nnkCall:
    if isAddFlopCall(stmt):
      return ""  # Skip addFLOPImpl — handled on host side
    elif stmt.len >= 2 and stmt[0].kind == nnkSym and stmt[0].strVal == "[]=":
      if isElementLevelWrite(stmt):
        # Element-level write: view[n][i,j] = val
        let innerCall = stmt[1]
        var viewName = "output"
        if innerCall[1].kind == nnkSym:
          viewName = innerCall[1].strVal

        if stmt.len == 4:
          let idxCode = clTranspileScalar(stmt[2], ctx)
          let valCode = clTranspileScalar(stmt[3], ctx)
          return p & aosoaIdx(viewName & "_data", "group", "lane", viewName & "_elems", idxCode) & " = " & valCode & ";\n"
        elif stmt.len >= 5:
          let rowCode = clTranspileScalar(stmt[2], ctx)
          let colCode = clTranspileScalar(stmt[3], ctx)
          let valCode = clTranspileScalar(stmt[4], ctx)
          let flatIdx = "(" & rowCode & ")*NC+(" & colCode & ")"
          return p & aosoaIdx(viewName & "_data", "group", "lane", viewName & "_elems", flatIdx) & " = " & valCode & ";\n"
      else:
        # Check for local var element write: localVar[i,j] = val
        # AST: nnkCall("[]=", localVarSym, i, j, val)  — no inner [] call
        let target = stmt[1]
        if target.kind == nnkSym:
          let vn = target.strVal
          var isProxy = false
          try: isProxy = isTensorSiteProxySym(target)
          except: discard
          if isProxy:
            if stmt.len == 4:
              # 1D: localVar[idx] = val
              let idxCode = clTranspileScalar(stmt[2], ctx)
              let valCode = clTranspileScalar(stmt[3], ctx)
              return p & vn & "[" & idxCode & "] = " & valCode & ";\n"
            elif stmt.len >= 5:
              # 2D: localVar[row, col] = val
              let rowCode = clTranspileScalar(stmt[2], ctx)
              let colCode = clTranspileScalar(stmt[3], ctx)
              let valCode = clTranspileScalar(stmt[4], ctx)
              return p & vn & "[(" & rowCode & ")*NC+(" & colCode & ")] = " & valCode & ";\n"
            return ""

        # Tensor-level: view[n] = matrix_expr
        let viewNode = stmt[1]
        let siteNode = stmt[2]
        let rhsNode = stmt[3]
        let sr = clResolveSiteRef(viewNode, siteNode, ctx)
        let tmp = ctx.nextTmp()
        var s = p & "{ // assign\n"
        s &= ind(d+1) & ctx.elemType & " " & tmp & "[NC*NC];\n"
        let matRes = emitMatExpr(tmp, rhsNode, ctx, d+1)
        s &= matRes.code
        let storeElems = sr.elemsVar
        if sr.isGauge:
          for di in 0..<sr.gaugeDim:
            let cond = if di == 0: "if" else: "else if"
            s &= ind(d+1) & cond & " (" & sr.dirExpr & " == " & $di & ") {\n"
            s &= ind(d+2) & "for (int _e = 0; _e < " & storeElems & "; _e++)\n"
            s &= ind(d+3) & aosoaIdx(sr.gaugeName & "_" & $di & "_data", sr.groupVar, sr.laneVar, sr.elemsVar, "_e") & " = " & tmp & "[_e];\n"
            s &= ind(d+1) & "}\n"
        else:
          s &= ind(d+1) & "for (int _e = 0; _e < " & storeElems & "; _e++)\n"
          s &= ind(d+2) & aosoaIdx(sr.dataVar, sr.groupVar, sr.laneVar, sr.elemsVar, "_e") & " = " & tmp & "[_e];\n"
        s &= p & "}\n"
        return s
    return ""

  of nnkIfStmt:
    var s = ""
    for branch in stmt:
      if branch.kind == nnkElifBranch:
        let cond = clTranspileScalar(branch[0], ctx)
        s &= p & "if (" & cond & ") {\n"
        var innerStmts: seq[NimNode]
        let bodyStmt = branch[1]
        if bodyStmt.kind == nnkStmtList:
          for child in bodyStmt: innerStmts.add child
        else:
          innerStmts.add bodyStmt
        for inner in innerStmts:
          s &= clTranspileStmt(inner, ctx, info, d+1)
        s &= p & "}\n"
      elif branch.kind == nnkElse:
        s &= p & "else {\n"
        var innerStmts: seq[NimNode]
        let bodyStmt = branch[0]
        if bodyStmt.kind == nnkStmtList:
          for child in bodyStmt: innerStmts.add child
        else:
          innerStmts.add bodyStmt
        for inner in innerStmts:
          s &= clTranspileStmt(inner, ctx, info, d+1)
        s &= p & "}\n"
    return s

  of nnkForStmt:
    # for i in 0..<NC  →  for (int i = 0; i < NC; i++)
    # AST: nnkForStmt(loopVarSym, rangeExpr, bodyStmtList)
    if stmt.len >= 3:
      let loopVarNode = stmt[0]
      let rangeNode = stmt[1]
      let forBody = stmt[2]
      let iterVar = if loopVarNode.kind == nnkSym: loopVarNode.strVal else: "i"

      var loExpr = "0"
      var hiExpr = "NC"
      # Parse range: Call("..<", lo, hi) or Call("countup", lo, hi)
      if rangeNode.kind == nnkCall and rangeNode.len >= 3:
        if rangeNode[0].kind == nnkSym:
          let fn = rangeNode[0].strVal
          loExpr = clTranspileScalar(rangeNode[1], ctx)
          if fn == "..<":
            hiExpr = clTranspileScalar(rangeNode[2], ctx)
          elif fn == "countup":
            hiExpr = "(" & clTranspileScalar(rangeNode[2], ctx) & "+1)"
      elif rangeNode.kind == nnkInfix and rangeNode.len >= 3:
        if rangeNode[0].kind == nnkSym and rangeNode[0].strVal == "..<":
          loExpr = clTranspileScalar(rangeNode[1], ctx)
          hiExpr = clTranspileScalar(rangeNode[2], ctx)

      var s = p & "for (int " & iterVar & " = " & loExpr & "; " & iterVar & " < " & hiExpr & "; " & iterVar & "++) {\n"
      var innerStmts: seq[NimNode]
      if forBody.kind == nnkStmtList:
        for child in forBody: innerStmts.add child
      else:
        innerStmts.add forBody
      for inner in innerStmts:
        s &= clTranspileStmt(inner, ctx, info, d+1)
      s &= p & "}\n"
      return s
    return ""

  of nnkDiscardStmt:
    return ""

  else:
    return p & "// skipped: " & $stmt.kind & "\n"

#[ ============================================================================
   Kernel Source Assembly
   ============================================================================ ]#

proc generateKernelSource(kernelName: string, body: NimNode, info: KernelInfo): string =
  var ctx = newClCodeCtx(info)
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
  # Runtime float vars (scalar coefficients like cp, cr)
  let runtimeFloatVars = findRuntimeFloatVars(body, info)
  for rv in runtimeFloatVars:
    params.add "const " & ctx.scalarType & " " & rv.name
  # Runtime dot-accessed float vars (e.g. c.cp -> c_cp)
  let runtimeDotFloatVars = findRuntimeDotFloatVars(body, info)
  for rv in runtimeDotFloatVars:
    params.add "const " & ctx.scalarType & " " & rv.name
  # Runtime dot-accessed int vars (e.g. c.someInt -> c_someInt)
  let runtimeDotIntVars = findRuntimeDotIntVars(body, info)
  for rv in runtimeDotIntVars:
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

  # Process each statement using recursive dispatch
  var stmts: seq[NimNode]
  if body.kind == nnkStmtList:
    for child in body: stmts.add child
  else:
    stmts.add body

  for stmt in stmts:
    src &= clTranspileStmt(stmt, ctx, info, 1)

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

    stencilSetup.add quote do:
      var `sbuf` = getOrUploadStencil(clContext, clQueues[0], `ssym`.offsets)

    stencilArgs.add quote do:
      `kernelSym`.setArg(`sbuf`, `offIdx`)
      var np = `ssym`.nPoints.int32
      `kernelSym`.setArg(np, `npIdx`)

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

  # Runtime float variable args (scalar coefficients like cp, cr)
  let runtimeFloatVars = findRuntimeFloatVars(body, info)
  var runtimeFloatArgStmts = newStmtList()
  for rv in runtimeFloatVars:
    let idx = newLit(argIndex)
    let rvsym = rv.sym
    let tmp = genSym(nskVar, "rfv")
    runtimeFloatArgStmts.add quote do:
      var `tmp` = `rvsym`.float64
      `kernelSym`.setArg(`tmp`, `idx`)
    argIndex += 1

  # Runtime dot-accessed float variable args (e.g. c.cp -> c_cp)
  let runtimeDotFloatVars = findRuntimeDotFloatVars(body, info)
  var runtimeDotFloatArgStmts = newStmtList()
  for rv in runtimeDotFloatVars:
    let idx = newLit(argIndex)
    let dotNode = rv.dotNode
    let tmp = genSym(nskVar, "rdfv")
    # Rebuild dot expr from scratch: objSym.fieldIdent (avoid skField symbol)
    let freshDot = nnkDotExpr.newTree(dotNode[0], ident(dotNode[1].strVal))
    let convExpr = newCall(ident"float64", freshDot)
    let varSection = nnkVarSection.newTree(
      nnkIdentDefs.newTree(tmp, newEmptyNode(), convExpr))
    let setArgCall = newCall(
      nnkDotExpr.newTree(kernelSym, ident"setArg"), tmp, idx)
    runtimeDotFloatArgStmts.add varSection
    runtimeDotFloatArgStmts.add setArgCall
    argIndex += 1

  # Runtime dot-accessed int variable args (e.g. c.someInt -> c_someInt)
  let runtimeDotIntVars = findRuntimeDotIntVars(body, info)
  var runtimeDotIntArgStmts = newStmtList()
  for rv in runtimeDotIntVars:
    let idx = newLit(argIndex)
    let dotNode = rv.dotNode
    let tmp = genSym(nskVar, "rdiv")
    # Rebuild dot expr from scratch: objSym.fieldIdent (avoid skField symbol)
    let freshDot = nnkDotExpr.newTree(dotNode[0], ident(dotNode[1].strVal))
    let convExpr = newCall(ident"int32", freshDot)
    let varSection = nnkVarSection.newTree(
      nnkIdentDefs.newTree(tmp, newEmptyNode(), convExpr))
    let setArgCall = newCall(
      nnkDotExpr.newTree(kernelSym, ident"setArg"), tmp, idx)
    runtimeDotIntArgStmts.add varSection
    runtimeDotIntArgStmts.add setArgCall
    argIndex += 1

  # Build an expression to determine NC at runtime:
  # Use the first gauge view's field shape if available, otherwise the output view
  var ncShapeExpr: NimNode
  if info.gaugeViews.len > 0:
    ncShapeExpr = nnkBracketExpr.newTree(
      nnkDotExpr.newTree(info.gaugeViews[0].nimSym, ident"field"), newLit(0))
  elif outViewSym != nil:
    ncShapeExpr = outViewSym
  else:
    ncShapeExpr = shapeExpr

  # --- Host-side FLOP accumulation (built before quote do, spliced in) ---
  var flopBlock = newStmtList()   # empty = no-op when spliced
  let flopEntries = findFlopCalls(body)
  if flopEntries.len > 0:
    # Build the inner statements: compute total sites, then addFLOPImpl calls
    var innerStmts = newStmtList()

    # var flopTotalSites {.inject.} = 0
    # for flopX {.inject.} in sitesExpr.data.sitesPerDevice: flopTotalSites += flopX
    let sitesPerDevExpr = nnkDotExpr.newTree(
      nnkDotExpr.newTree(sitesExpr, ident"data"), ident"sitesPerDevice")
    innerStmts.add quote do:
      var flopTotalSites {.inject.} = 0
      for flopX {.inject.} in `sitesPerDevExpr`:
        flopTotalSites += flopX

    # Emit addFLOPImpl calls for each entry
    for entry in flopEntries:
      let flopsRepr = repr(entry.flopsExpr)
      let flops = parseExpr(flopsRepr)
      if entry.condExpr == nil:
        innerStmts.add quote do:
          addFLOPImpl(profiler, `flops` * flopTotalSites)
      else:
        let condRepr = repr(entry.condExpr)
        let cond = parseExpr(condRepr)
        innerStmts.add quote do:
          if `cond`:
            addFLOPImpl(profiler, `flops` * flopTotalSites)

    # Wrap everything in when ProfileMode == 1:
    flopBlock = nnkWhenStmt.newTree(
      nnkElifBranch.newTree(
        nnkInfix.newTree(ident"==", ident"ProfileMode", newLit(1)),
        innerStmts))

  result = quote do:
    block:
      # NC must be the gauge field matrix dimension, not the output scalar field
      let ncShape = `ncShapeExpr`.shape
      let ncDim = if ncShape.len >= 1: ncShape[0] else: 1
      let finalKernelSource = `kernelSourceLit`.replace("{NC_VALUE}", $ncDim)

      when DebugKernels:
        echo "=== Generated OpenCL Kernel ==="
        echo finalKernelSource
        echo "================================"

      let `kernelSym` = getOrCompile(clContext, finalKernelSource, clDevices, `kernelNameLit`)

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
          `runtimeFloatArgStmts`
          `runtimeDotFloatArgStmts`
          `runtimeDotIntArgStmts`

          when UseWorkGroups:
            clQueues[`devIdxSym`].run(`kernelSym`, devSites, VectorWidth)
          else:
            clQueues[`devIdxSym`].run(`kernelSym`, devSites)

      for `devIdxSym` in 0..<numDevices:
        check clwrap.flush(clQueues[`devIdxSym`])

      `flopBlock`

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
