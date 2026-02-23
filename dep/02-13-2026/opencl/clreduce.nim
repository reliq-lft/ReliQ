#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/opencl/clreduce.nim
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

## Parallel Reduce Macro for TensorFieldView (OpenCL backend) — GPU Codegen
##
## This module provides the ``reduce`` macro for parallel reduction loops
## on TensorFieldView objects using OpenCL. The macro follows the same
## AST-to-OpenCL-C transpiler approach as ``cldisp.nim``'s ``each`` macro:
##
## 1. A per-work-item kernel computes each site's scalar contribution
##    and writes it to a ``__global double* partials`` buffer.
## 2. The host reads back the partials and sums them.
## 3. ``GA_Dgop`` then sums across MPI ranks.
##
## This is fully GPU-parallel — no serial CPU fallback.
##
## Usage:
##   var traceSum = 0.0
##   for n in reduce view.all:
##     traceSum += trace(view[n]).re
##
## After the loop, ``traceSum`` contains the globally-reduced result
## across all MPI ranks.

import std/[macros, strutils, sets]

import clbase
import cldisp
export clbase

#[ ============================================================================
   Helper: Find += accumulation variable in typed AST
   ============================================================================ ]#

proc findAccumTarget(body: NimNode): NimNode =
  case body.kind
  of nnkInfix:
    if body.len >= 3 and body[0].kind in {nnkSym, nnkIdent}:
      if body[0].strVal == "+=":
        return body[1]
  of nnkCall:
    if body.len >= 3 and body[0].kind == nnkSym and body[0].strVal == "+=":
      return body[1]
  else:
    discard
  for child in body:
    let found = findAccumTarget(child)
    if found != nil:
      return found
  return nil

proc unwrapSym(n: NimNode): NimNode =
  var cur = n
  while cur.kind in {nnkHiddenAddr, nnkHiddenDeref, nnkAddr, nnkDerefExpr,
                      nnkHiddenStdConv, nnkHiddenSubConv, nnkConv}:
    cur = cur[0]
  return cur

#[ ============================================================================
   Reduce RHS → OpenCL C Transpiler
   ============================================================================ ]#

proc transpileReduceRhs(rhs: NimNode, ctx: var ClCodeCtx, d: int): string =
  ## Generate OpenCL C code that computes a scalar and adds it to ``accum``.
  ## Fully general — delegates to ``emitMatExpr`` for any matrix expression.
  let p = ind(d)

  # --- Pattern: re(view[n]) or im(view[n]) — proc-style accessor on scalar field ---
  if rhs.kind == nnkCall and rhs[0].kind == nnkSym and rhs[0].strVal in ["re", "im"]:
    let inner = rhs[1]
    let accessor = if rhs[0].strVal == "re": ".x" else: ".y"
    # re(view[n]) — load element 0 of scalar field
    if inner.kind == nnkCall and inner[0].kind == nnkSym and inner[0].strVal == "[]":
      let sr = clResolveSiteRef(inner[1], inner[2], ctx)
      var s = ""
      let elems = sr.elemsVar
      if ctx.isComplex:
        s &= p & "accum += " & aosoaIdx(sr.dataVar, sr.groupVar, sr.laneVar, elems, "0") & accessor & ";\n"
      else:
        s &= p & "accum += " & aosoaIdx(sr.dataVar, sr.groupVar, sr.laneVar, elems, "0") & ";\n"
      return s

  # --- Pattern: expr.re or expr.im ---
  if rhs.kind == nnkDotExpr and rhs.len >= 2 and rhs[1].kind == nnkSym:
    let field = rhs[1].strVal
    if field in ["re", "im"]:
      let inner = rhs[0]
      let accessor = if field == "re": ".x" else: ".y"  # double2 uses .x/.y

      # trace(something).re/.im
      if inner.kind == nnkCall and inner[0].kind == nnkSym and inner[0].strVal == "trace":
        let traceArg = inner[1]
        # trace(view[n]).re — direct AoSoA load of diagonal
        if traceArg.kind == nnkCall and traceArg[0].kind == nnkSym and traceArg[0].strVal == "[]":
          let sr = clResolveSiteRef(traceArg[1], traceArg[2], ctx)
          var s = ""
          if sr.isGauge:
            let elems = sr.elemsVar
            for di in 0..<sr.gaugeDim:
              let cond = if di == 0: "if" else: "else if"
              s &= p & cond & " (" & sr.dirExpr & " == " & $di & ") {\n"
              s &= p & "  for (int _i = 0; _i < NC; _i++)\n"
              if ctx.isComplex:
                s &= p & "    accum += " & aosoaIdx(sr.gaugeName & "_" & $di & "_data", sr.groupVar, sr.laneVar, elems, "_i*NC+_i") & accessor & ";\n"
              else:
                s &= p & "    accum += " & aosoaIdx(sr.gaugeName & "_" & $di & "_data", sr.groupVar, sr.laneVar, elems, "_i*NC+_i") & ";\n"
              s &= p & "}\n"
          else:
            let elems = sr.elemsVar
            s &= p & "for (int _i = 0; _i < NC; _i++)\n"
            if ctx.isComplex:
              s &= p & "  accum += " & aosoaIdx(sr.dataVar, sr.groupVar, sr.laneVar, elems, "_i*NC+_i") & accessor & ";\n"
            else:
              s &= p & "  accum += " & aosoaIdx(sr.dataVar, sr.groupVar, sr.laneVar, elems, "_i*NC+_i") & ";\n"
          return s

        # trace(matexpr).re/.im — compute matrix, then trace
        else:
          var s = ""
          let tmp = ctx.nextTmp()
          s &= p & ctx.elemType & " " & tmp & "[NC*NC];\n"
          let matRes = emitMatExpr(tmp, traceArg, ctx, d)
          s &= matRes.code
          s &= p & "for (int _i = 0; _i < NC; _i++)\n"
          if ctx.isComplex:
            s &= p & "  accum += " & tmp & "[_i*NC+_i]" & accessor & ";\n"
          else:
            s &= p & "  accum += " & tmp & "[_i*NC+_i];\n"
          return s

      # view[n].re/.im — direct access to scalar field element 0
      if inner.kind == nnkCall and inner[0].kind == nnkSym and inner[0].strVal == "[]":
        let sr = clResolveSiteRef(inner[1], inner[2], ctx)
        var s = ""
        let elems = sr.elemsVar
        if ctx.isComplex:
          s &= p & "accum += " & aosoaIdx(sr.dataVar, sr.groupVar, sr.laneVar, elems, "0") & accessor & ";\n"
        else:
          s &= p & "accum += " & aosoaIdx(sr.dataVar, sr.groupVar, sr.laneVar, elems, "0") & ";\n"
        return s

  # --- Pattern: trace(something) without .re/.im ---
  if rhs.kind == nnkCall and rhs[0].kind == nnkSym and rhs[0].strVal == "trace":
    let traceArg = rhs[1]
    if traceArg.kind == nnkCall and traceArg[0].kind == nnkSym and traceArg[0].strVal == "[]":
      let sr = clResolveSiteRef(traceArg[1], traceArg[2], ctx)
      var s = ""
      if sr.isGauge:
        let elems = sr.elemsVar
        for di in 0..<sr.gaugeDim:
          let cond = if di == 0: "if" else: "else if"
          s &= p & cond & " (" & sr.dirExpr & " == " & $di & ") {\n"
          s &= p & "  for (int _i = 0; _i < NC; _i++)\n"
          if ctx.isComplex:
            s &= p & "    accum += " & aosoaIdx(sr.gaugeName & "_" & $di & "_data", sr.groupVar, sr.laneVar, elems, "_i*NC+_i") & ".x;\n"
          else:
            s &= p & "    accum += " & aosoaIdx(sr.gaugeName & "_" & $di & "_data", sr.groupVar, sr.laneVar, elems, "_i*NC+_i") & ";\n"
          s &= p & "}\n"
        return s
      else:
        let elems = sr.elemsVar
        s &= p & "for (int _i = 0; _i < NC; _i++)\n"
        if ctx.isComplex:
          s &= p & "  accum += " & aosoaIdx(sr.dataVar, sr.groupVar, sr.laneVar, elems, "_i*NC+_i") & ".x;\n"
        else:
          s &= p & "  accum += " & aosoaIdx(sr.dataVar, sr.groupVar, sr.laneVar, elems, "_i*NC+_i") & ";\n"
        return s

    # trace(matexpr) — compute matrix, then trace
    var s = ""
    let tmp = ctx.nextTmp()
    s &= p & ctx.elemType & " " & tmp & "[NC*NC];\n"
    let matRes = emitMatExpr(tmp, traceArg, ctx, d)
    s &= matRes.code
    s &= p & "for (int _i = 0; _i < NC; _i++)\n"
    if ctx.isComplex:
      s &= p & "  accum += " & tmp & "[_i*NC+_i].x;\n"
    else:
      s &= p & "  accum += " & tmp & "[_i*NC+_i];\n"
    return s

  # --- Pattern: literal scalars ---
  if rhs.kind in {nnkIntLit..nnkInt64Lit}:
    return p & "accum += " & $rhs.intVal & ";\n"
  if rhs.kind in {nnkFloatLit..nnkFloat64Lit}:
    return p & "accum += " & $rhs.floatVal & ";\n"

  # --- Pattern: arithmetic combinations (*, +, -, /) ---
  if rhs.kind == nnkInfix and rhs.len >= 3 and rhs[0].kind == nnkSym:
    let op = rhs[0].strVal
    if op in ["*", "+", "-", "/"]:
      let lCode = clTranspileScalar(rhs[1], ctx)
      let rCode = clTranspileScalar(rhs[2], ctx)
      return p & "accum += (" & lCode & " " & op & " " & rCode & ");\n"

  # --- Unwrap hidden conversions ---
  if rhs.kind in {nnkHiddenStdConv, nnkConv, nnkHiddenDeref, nnkHiddenAddr, nnkHiddenSubConv}:
    if rhs.len > 0:
      return transpileReduceRhs(rhs[^1], ctx, d)

  if rhs.kind == nnkHiddenCallConv:
    if rhs.len >= 2:
      return transpileReduceRhs(rhs[1], ctx, d)

  # --- Fallback: transpile as scalar expression ---
  let code = clTranspileScalar(rhs, ctx)
  return p & "accum += " & code & ";\n"

#[ ============================================================================
   Generate Reduce OpenCL Kernel Source
   ============================================================================ ]#

proc generateReduceKernel(body: NimNode, info: KernelInfo, kernelName: string): string =
  ## Generate an OpenCL kernel with work-group-level reduction.
  ## Each work-group reduces its elements into a single partial sum
  ## using ``__local`` memory, so the host reads back only
  ## ``numWorkGroups`` values instead of ``numSites``.
  var ctx = newClCodeCtx(info)
  let vw = $VectorWidth

  var src = ""

  # FP64 extension (needed for double precision)
  if info.scalarType in {etFloat64, etInt64}:
    src &= "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n"
  src &= "#define NC {NC_VALUE}\n"
  src &= "#define WG_SIZE 256\n\n"

  # Complex helpers for OpenCL
  if info.isComplex:
    let vt = ctx.vecType  # "double2" or "float2"
    src &= "inline " & vt & " cmul(" & vt & " a, " & vt & " b) {\n"
    src &= "  return (" & vt & ")(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);\n"
    src &= "}\n"
    src &= "inline " & vt & " cconj(" & vt & " a) {\n"
    src &= "  return (" & vt & ")(a.x, -a.y);\n"
    src &= "}\n\n"

  # Build parameter list
  var params: seq[string]
  params.add "__global " & ctx.scalarType & "* partials"
  for v in info.views:
    params.add "__global " & ctx.elemType & "* " & v.name & "_data"
  for gv in info.gaugeViews:
    for d in 0..<gv.dim:
      params.add "__global " & ctx.elemType & "* " & gv.name & "_" & $d & "_data"
  for v in info.views:
    params.add "const int " & v.name & "_elems"
  for gv in info.gaugeViews:
    params.add "const int " & gv.name & "_elems"
  params.add "const int numSites"
  for s in info.stencils:
    params.add "__global const int* " & s.name & "_offsets"
    params.add "const int " & s.name & "_npts"
  let runtimeVars = findRuntimeIntVars(body, info)
  for rv in runtimeVars:
    params.add "const int " & rv.name

  src &= "__kernel void " & kernelName & "(\n"
  src &= "    " & params.join(",\n    ")
  src &= "\n) {\n"

  # Work-item setup
  src &= "  const int " & ctx.loopVarStr & " = get_global_id(0);\n"
  src &= "  const int VW = " & vw & ";\n"
  src &= "  const int group = " & ctx.loopVarStr & " / VW;\n"
  src &= "  const int lane = " & ctx.loopVarStr & " % VW;\n\n"
  src &= "  " & ctx.scalarType & " accum = 0;\n\n"

  # Guard: only compute for valid sites
  src &= "  if (" & ctx.loopVarStr & " < numSites) {\n"

  # Process each statement in the reduce body
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
            src &= "  int " & vn & "_idx = " & lb.stencilName & "_offsets[" &
                   ctx.loopVarStr & " * " & lb.stencilName & "_npts + 2 * " & lb.dirExpr & "];\n"
            src &= "  int " & vn & "_group = " & vn & "_idx / VW;\n"
            src &= "  int " & vn & "_lane = " & vn & "_idx % VW;\n"
          of lbkStencilBwd:
            src &= "  int " & vn & "_idx = " & lb.stencilName & "_offsets[" &
                   ctx.loopVarStr & " * " & lb.stencilName & "_npts + 2 * " & lb.dirExpr & " + 1];\n"
            src &= "  int " & vn & "_group = " & vn & "_idx / VW;\n"
            src &= "  int " & vn & "_lane = " & vn & "_idx % VW;\n"
          of lbkStencilNeighbor:
            src &= "  int " & vn & "_idx = " & lb.stencilName & "_offsets[" &
                   ctx.loopVarStr & " * " & lb.stencilName & "_npts + " & lb.pointExpr & "];\n"
            src &= "  int " & vn & "_group = " & vn & "_idx / VW;\n"
            src &= "  int " & vn & "_lane = " & vn & "_idx % VW;\n"
          of lbkStencilCorner:
            let nd = $lb.nDim
            let sA = lb.signExprA
            let dA = lb.dirExprA
            let sB = lb.signExprB
            let dB = lb.dirExprB
            let prefix = "_c_" & vn
            src &= "  int " & prefix & "_a = (" & dA & " < " & dB & ") ? (" & dA & ") : (" & dB & ");\n"
            src &= "  int " & prefix & "_b = (" & dA & " < " & dB & ") ? (" & dB & ") : (" & dA & ");\n"
            src &= "  int " & prefix & "_pair = " & prefix & "_a * (2*" & nd & " - 1 - " & prefix & "_a) / 2 + " & prefix & "_b - " & prefix & "_a - 1;\n"
            src &= "  int " & prefix & "_sA = (" & dA & " <= " & dB & ") ? (" & sA & ") : (" & sB & ");\n"
            src &= "  int " & prefix & "_sB = (" & dA & " <= " & dB & ") ? (" & sB & ") : (" & sA & ");\n"
            src &= "  int " & prefix & "_si = (" & prefix & "_sA > 0 ? 0 : 2) + (" & prefix & "_sB > 0 ? 0 : 1);\n"
            src &= "  int " & vn & "_idx = " & lb.stencilName & "_offsets[" &
                   ctx.loopVarStr & " * " & lb.stencilName & "_npts + 2*" & nd & " + " & prefix & "_pair * 4 + " & prefix & "_si];\n"
            src &= "  int " & vn & "_group = " & vn & "_idx / VW;\n"
            src &= "  int " & vn & "_lane = " & vn & "_idx % VW;\n"
          of lbkMatMul:
            src &= "  " & ctx.elemType & " " & vn & "[NC*NC];\n"
            let matRes = emitMatExpr(vn, val, ctx, 1)
            src &= matRes.code
            src &= "  const int " & vn & "_elems = " & matRes.elems & ";\n"
          of lbkOther:
            let code = clTranspileScalar(val, ctx)
            src &= "  " & ctx.scalarType & " " & vn & " = " & code & ";\n"
    of nnkCall:
      # Skip addFLOPImpl calls — handled on the host side
      if isAddFlopCall(stmt):
        discard
      else:
        discard  # other calls not yet handled in reduce kernel
    of nnkInfix:
      if stmt.len >= 3 and stmt[0].kind == nnkSym and stmt[0].strVal == "+=":
        let rhs = stmt[2]
        src &= "  {\n"
        src &= transpileReduceRhs(rhs, ctx, 2)
        src &= "  }\n"
    else:
      discard

  src &= "  } // end if (n < numSites)\n\n"

  # Work-group reduction using __local memory
  src &= "  // Work-group reduction\n"
  src &= "  __local " & ctx.scalarType & " scratch[WG_SIZE];\n"
  src &= "  const int lid = get_local_id(0);\n"
  src &= "  scratch[lid] = accum;\n"
  src &= "  barrier(CLK_LOCAL_MEM_FENCE);\n\n"
  src &= "  for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {\n"
  src &= "    if (lid < stride) scratch[lid] += scratch[lid + stride];\n"
  src &= "    barrier(CLK_LOCAL_MEM_FENCE);\n"
  src &= "  }\n\n"
  src &= "  if (lid == 0) partials[get_group_id(0)] = scratch[0];\n"
  src &= "}\n"
  return src

#[ ============================================================================
   Typed Reduce Implementation — GPU Codegen
   ============================================================================ ]#

macro reduceImpl*(loopVar: untyped, lo: typed, hi: typed, body: typed): untyped =
  ## Internal typed macro for OpenCL reduce dispatch.
  ##
  ## Generates a per-work-item OpenCL kernel that computes each site's
  ## scalar contribution and writes it to ``__global double* partials``.
  ## The host reads back the partials, sums them, and calls ``GA_Dgop``
  ## to sum across MPI ranks.

  let loopVarSym = loopVar
  let info = gatherInfo(body, loopVarSym)

  let accumNode = findAccumTarget(body)
  let accumSym = if accumNode != nil: unwrapSym(accumNode) else: nil

  if accumNode == nil:
    error("reduce loop body must contain a += to an accumulation variable")

  # CPU fallback for echo/debugEcho
  if hasEchoStatement(body):
    let siteIdxSym = genSym(nskForVar, "siteIdx")
    let assignStmt = newNimNode(nnkAsgn).add(loopVarSym, siteIdxSym)
    let loopBody = newStmtList(assignStmt, body)
    let rangeExpr = newNimNode(nnkInfix).add(ident"..<", lo, hi)
    let forLoop = newNimNode(nnkForStmt).add(siteIdxSym, rangeExpr, loopBody)
    result = newNimNode(nnkBlockStmt).add(newEmptyNode(), newStmtList(
      quote do:
        let savedAccum = `accumSym`,
      forLoop,
      quote do:
        if gaIsLive:
          var deltaAccum = `accumSym` - savedAccum
          GA_Dgop(addr deltaAccum, 1, "+")
          `accumSym` = savedAccum + deltaAccum
    ))
    return result

  let kernelName = "tfv_reduce_" & $body.lineInfoObj.line
  let kernelSource = generateReduceKernel(body, info, kernelName)
  let kernelNameLit = newLit(kernelName)
  let kernelSourceLit = newLit(kernelSource)

  # Find a TensorFieldView sym for shape/sites info.
  # Only use regular views here; GaugeFieldView doesn't have .shape/.numSites.
  var shapeViewSym: NimNode = nil
  for v in info.views:
    shapeViewSym = v.nimSym; break

  var sitesExpr: NimNode
  if shapeViewSym != nil:
    sitesExpr = shapeViewSym
  elif info.gaugeViews.len > 0:
    sitesExpr = nnkBracketExpr.newTree(
      nnkDotExpr.newTree(info.gaugeViews[0].nimSym, ident"field"), newLit(0))
  else:
    error("No view found for sites info")

  var shapeExpr: NimNode
  if shapeViewSym != nil:
    shapeExpr = shapeViewSym
  elif info.gaugeViews.len > 0:
    shapeExpr = nnkBracketExpr.newTree(
      nnkDotExpr.newTree(info.gaugeViews[0].nimSym, ident"field"), newLit(0))
  else:
    shapeExpr = sitesExpr

  # NC shape: use gauge field matrix dimension if available
  var ncShapeExpr: NimNode
  if info.gaugeViews.len > 0:
    ncShapeExpr = nnkBracketExpr.newTree(
      nnkDotExpr.newTree(info.gaugeViews[0].nimSym, ident"field"), newLit(0))
  elif shapeViewSym != nil:
    ncShapeExpr = shapeViewSym
  else:
    ncShapeExpr = shapeExpr

  let kernelSym = genSym(nskLet, "kernel")
  let devIdxSym = genSym(nskForVar, "devIdx")

  # --- Build setArg statements ---
  var setArgsStmts = newStmtList()
  var argIndex = 0

  # arg 0: partials buffer (set below)
  let partialsIdx = newLit(argIndex)
  argIndex += 1

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

  # numSites
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

  # --- Host-side FLOP accumulation (built before quote do, spliced in) ---
  var flopBlock = newStmtList()   # empty = no-op when spliced
  let flopEntries = findFlopCalls(body)
  if flopEntries.len > 0:
    var innerStmts = newStmtList()

    # Compute total sites from sitesPerDevice
    let sitesPerDevExpr = nnkDotExpr.newTree(
      nnkDotExpr.newTree(sitesExpr, ident"data"), ident"sitesPerDevice")
    innerStmts.add quote do:
      var flopTotalSites {.inject.} = 0
      for flopX {.inject.} in `sitesPerDevExpr`:
        flopTotalSites += flopX

    # Emit addFLOPImpl calls for each entry
    for entry in flopEntries:
      let flops = untypeAst(entry.flopsExpr)
      if entry.condExpr == nil:
        innerStmts.add quote do:
          addFLOPImpl(profiler, `flops` * flopTotalSites)
      else:
        let cond = untypeAst(entry.condExpr)
        innerStmts.add quote do:
          if `cond`:
            addFLOPImpl(profiler, `flops` * flopTotalSites)

    # Wrap in when ProfileMode == 1:
    flopBlock = nnkWhenStmt.newTree(
      nnkElifBranch.newTree(
        nnkInfix.newTree(ident"==", ident"ProfileMode", newLit(1)),
        innerStmts))

  result = quote do:
    block:
      let savedAccum = `accumSym`
      let ncShape = `ncShapeExpr`.shape
      let ncDim = if ncShape.len >= 1: ncShape[0] else: 1
      let finalKernelSource = `kernelSourceLit`.replace("{NC_VALUE}", $ncDim)

      when DebugKernels:
        echo "=== Generated OpenCL Reduce Kernel ==="
        echo finalKernelSource
        echo "======================================="

      let `kernelSym` = getOrCompile(clContext, finalKernelSource, clDevices, `kernelNameLit`)

      let numDevices = clQueues.len
      let sitesPerDev = `sitesExpr`.data.sitesPerDevice
      let totalSites = `sitesExpr`.numSites()

      # Work-group reduction: each work-group of 256 produces one partial sum
      const reduceWgSize = 256
      let numWorkGroups = (totalSites + reduceWgSize - 1) div reduceWgSize
      let globalWorkSize = numWorkGroups * reduceWgSize  # rounded up

      # Cached partials buffer — sized for numWorkGroups, not totalSites
      var partialsBuf = getOrAllocPartials(clContext, numWorkGroups)
      var partialsHostPtr = getOrAllocPartialsHost(numWorkGroups)

      `stencilSetup`

      for `devIdxSym` in 0..<numDevices:
        let devSites = sitesPerDev[`devIdxSym`]
        if devSites > 0:
          `kernelSym`.setArg(partialsBuf, `partialsIdx`)
          `setArgsStmts`
          `elemsStmts`

          var nsArg = devSites.int32
          `kernelSym`.setArg(nsArg, `numSitesIdx`)

          `stencilArgs`
          `runtimeArgStmts`

          clQueues[`devIdxSym`].run(`kernelSym`, globalWorkSize, reduceWgSize)

      for devIdx in 0..<numDevices:
        check clwrap.finish(clQueues[devIdx])

      # Read back only numWorkGroups partials (not totalSites)
      clQueues[0].read(partialsHostPtr[], partialsBuf)

      var localSum = 0.0
      for i in 0..<numWorkGroups:
        localSum += partialsHostPtr[][i]
      `accumSym` += localSum

      `stencilCleanup`

      # MPI allreduce — only allreduce the delta from this reduce call
      if gaIsLive:
        var deltaAccum = `accumSym` - savedAccum
        GA_Dgop(addr deltaAccum, 1, "+")
        `accumSym` = savedAccum + deltaAccum

      `flopBlock`

#[ ============================================================================
   Public Reduce Macro (ForLoopStmt)
   ============================================================================ ]#

macro reduce*(forLoop: ForLoopStmt): untyped =
  ## OpenCL parallel reduce loop for TensorFieldView.
  ##
  ## Generates a per-work-item OpenCL kernel that computes scalar
  ## contributions, reads back the partials buffer, and sums on the
  ## host. ``GA_Dgop`` then sums across MPI ranks.
  ##
  ## Usage:
  ##   var traceSum = 0.0
  ##   for n in reduce view.all:
  ##     traceSum += trace(view[n]).re
  ##
  ## After the loop, ``traceSum`` holds the MPI-global sum.

  let loopVar = forLoop[0]
  let loopRangeNode = forLoop[1][1]  # Skip 'reduce' wrapper
  let body = forLoop[2]

  # Parse the range expression
  var isRange = false
  var startExpr, endExpr: NimNode

  if loopRangeNode.kind == nnkInfix:
    let opNode = loopRangeNode[0]
    let opStr = if opNode.kind in {nnkIdent, nnkSym, nnkOpenSymChoice, nnkClosedSymChoice}:
                  (if opNode.kind in {nnkOpenSymChoice, nnkClosedSymChoice}: opNode[0].strVal else: opNode.strVal)
                else: ""
    if opStr == "..<" or opStr == "..":
      isRange = true
      startExpr = loopRangeNode[1]
      endExpr = loopRangeNode[2]
  elif loopRangeNode.kind == nnkDotExpr and loopRangeNode[1].eqIdent("all"):
    isRange = true
    startExpr = newLit(0)
    endExpr = newCall(ident"numSites", loopRangeNode[0])

  if isRange:
    # Two-stage pattern: inject loopVar, call typed reduceImpl
    result = quote do:
      block:
        var `loopVar` {.inject.}: int = 0
        reduceImpl(`loopVar`, `startExpr`, `endExpr`, `body`)
  else:
    # Fallback: sequential reduce for non-range iterators
    result = quote do:
      block:
        for `loopVar` in `loopRangeNode`:
          `body`
