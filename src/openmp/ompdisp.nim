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

## OpenMP Dispatch Module for TensorFieldView — C Codegen Approach
##
## This module provides the ``each`` macro for parallel loops on
## ``TensorFieldView`` objects.  Unlike the OpenCL backend which JIT-
## compiles kernel strings at runtime, this module emits complete C
## functions at file scope via ``{.emit.}`` and calls them from Nim.
##
## The strategy mirrors ``cldisp.nim``: the typed macro AST is walked by
## a recursive transpiler that generates C code strings.  The resulting
## C function is self-contained (no Nim gotos, no closure captures) and
## contains the ``#pragma omp parallel for`` directive.
##
## Supported patterns (identical to OpenCL):
## - Tensor assign / copy:  ``viewC[n] = viewA[n]``
## - Element-wise +/-:     ``viewC[n] = viewA[n] + viewB[n]``
## - Scalar × tensor:      ``viewC[n] = 2.0 * viewA[n]``
## - Matrix multiply:      ``viewC[n] = viewA[n] * viewB[n]``
## - Mat-vec multiply:     ``viewC[n] = viewA[n] * viewV[n]``
## - Adjoint:              ``viewC[n] += ta * tb.adjoint()``
## - Stencil neighbors:    ``let fwd = stencil.fwd(n, mu)``
## - Gauge field access:   ``vu[mu][n]``
## - In-place accumulate:  ``vplaq[n] += expr``
## - Element-level write:  ``view[n][i,j] = val``
## - Echo (serial fallback)

import std/[macros, tables, strutils, sets, sequtils, compilesettings, os]

import ompbase
export ompbase

import ../ir/ir
export ir

{.passC: "-fopenmp".}
{.passL: "-fopenmp".}

{.emit: """
#include <omp.h>
#include <string.h>
""".}

# Path to the portable SIMD header, resolved at compile time relative to this source file
const simdHeaderPath* = currentSourcePath().parentDir() / "simd_intrinsics.h"

#[ ============================================================================
   OpenMP-Specific Utilities
   ============================================================================ ]#

proc elementTypeToC*(et: ElementType): string =
  case et
  of etFloat32: "float"
  of etFloat64: "double"
  of etInt32: "int"
  of etInt64: "long long"

## Emit a flat-index expression used when lane varies inside a VW-wide inner loop.
proc aosoaFlatIdx*(dataVar, groupVar, elemsVar, elemExpr: string, lane: string): string =
  let vw = $VectorWidth
  dataVar & "[" & groupVar & " * (" & vw & " * " & elemsVar & ") + (" & elemExpr & ") * " & vw & " + " & lane & "]"

#[ ============================================================================
   OpenMP C Code Generation Context
   ============================================================================ ]#

type
  OmpCodeCtx* = object
    loopVarStr*: string
    isComplex*: bool
    scalarType*: string  # "double", "float", "int", "long long"
    elemType*: string    # compound type for complex, scalar for real
    info*: KernelInfo
    tmpIdx*: int

proc newOmpCodeCtx*(info: KernelInfo): OmpCodeCtx =
  result.loopVarStr = info.loopVarStr
  result.isComplex = info.isComplex
  result.scalarType = elementTypeToC(info.scalarType)
  if info.isComplex:
    result.elemType = "NComplex"
  else:
    result.elemType = result.scalarType
  result.info = info

proc nextTmp*(ctx: var OmpCodeCtx): string =
  result = "_t" & $ctx.tmpIdx
  ctx.tmpIdx += 1

proc simdUseMacro*(ctx: OmpCodeCtx): string =
  ## Returns the #define name to select the right SIMD type in simd_intrinsics.h
  case ctx.scalarType
  of "float": "SIMD_USE_FLOAT"
  of "int": "SIMD_USE_INT32"
  of "long long": "SIMD_USE_INT64"
  else: "SIMD_USE_DOUBLE"

proc cPtrType*(ctx: OmpCodeCtx): string =
  ## Returns the C pointer type for this element type
  ctx.scalarType & "* "

#[ --- OmpCodeCtx adapters for shared IR functions --- ]#

proc ompResolveSiteRef*(viewExpr, siteExpr: NimNode, ctx: var OmpCodeCtx): SiteRef =
  ## Resolve a view[site] access via the shared IR resolver
  var irCtx = CodeCtx(loopVarStr: ctx.loopVarStr, isComplex: ctx.isComplex,
                       scalarType: ctx.scalarType, elemType: ctx.elemType,
                       info: ctx.info, tmpIdx: ctx.tmpIdx)
  result = resolveSiteRef(viewExpr, siteExpr, irCtx)
  ctx.tmpIdx = irCtx.tmpIdx

#[ ============================================================================
   C Code Generation — Matrix Expression Transpiler
   ============================================================================ ]#

type MatResult* = tuple[code: string, elems: string]

proc emitMatExpr*(target: string, n: NimNode, ctx: var OmpCodeCtx, d: int): MatResult
proc ompTranspileScalar*(n: NimNode, ctx: var OmpCodeCtx): string
proc ompTranspileStmt(stmt: NimNode, ctx: var OmpCodeCtx, info: KernelInfo, d: int): string

proc emitLoadView*(target: string, sr: SiteRef, ctx: var OmpCodeCtx, d: int): MatResult =
  ## Load tensor elements as SIMD vectors from AoSoA layout.
  ## For contiguous access (same group), uses simd_load_d.
  ## For stencil neighbors (different group per lane), uses simd_gather_d.
  var s = ""
  let p = ind(d)
  let elems = sr.elemsVar
  let isStencilAccess = sr.groupVar != "group"  # stencil neighbors have per-lane group/lane

  if sr.isGauge:
    for di in 0..<sr.gaugeDim:
      let cond = if di == 0: "if" else: "else if"
      s &= p & cond & " (" & sr.dirExpr & " == " & $di & ") {\n"
      if isStencilAccess:
        # Stencil: gather from per-lane neighbor indices
        let idxArray = sr.groupVar.replace("_group", "_indices")
        s &= p & "  for (int _e = 0; _e < " & elems & "; _e++)\n"
        s &= p & "    " & target & "[_e] = simd_gather(" & sr.gaugeName & "_" & $di & "_data, " & idxArray & ", _e, " & elems & ");\n"
      else:
        s &= p & "  for (int _e = 0; _e < " & elems & "; _e++)\n"
        s &= p & "    " & target & "[_e] = simd_load(" &
             aosoaBase(sr.gaugeName & "_" & $di & "_data", sr.groupVar, elems, "_e") & ");\n"
      s &= p & "}\n"
  else:
    if isStencilAccess:
      # Stencil: gather from per-lane neighbor indices
      let idxArray = sr.groupVar.replace("_group", "_indices")
      s &= p & "for (int _e = 0; _e < " & elems & "; _e++)\n"
      s &= p & "  " & target & "[_e] = simd_gather(" & sr.dataVar & ", " & idxArray & ", _e, " & elems & ");\n"
    else:
      s &= p & "for (int _e = 0; _e < " & elems & "; _e++)\n"
      s &= p & "  " & target & "[_e] = simd_load(" &
           aosoaBase(sr.dataVar, sr.groupVar, elems, "_e") & ");\n"
  return (s, elems)

proc emitMatMul*(target, lhs, rhs, lhsElems, rhsElems: string, ctx: var OmpCodeCtx, d: int): MatResult =
  ## SIMD matrix multiply: all temporaries are simd_v vectors.
  ## For complex: data is stored as (re, im) pairs of VW-wide doubles.
  ##   With elems = elementsPerSite = 2*NC*NC, the tensor column count is elems/(2*NC).
  ##   Element [i,j] has re at flat index 2*(i*cols+j) and im at 2*(i*cols+j)+1.
  ##   Complex matmul uses simd_cmadd_d.
  ## For real: standard i,j,k loop with simd_fmadd_d.
  ##   With elems = elementsPerSite = NC*NC, column count is elems/NC.
  var s = ""
  let p = ind(d)
  let lcVar = ctx.nextTmp() & "_lc"
  let rcVar = ctx.nextTmp() & "_rc"
  if ctx.isComplex:
    # Complex: elems = elementsPerSite = 2*NC*NC doubles.
    # Tensor column count = elems / (2*NC).
    s &= p & "const int " & lcVar & " = " & lhsElems & " / (2*NC);\n"
    s &= p & "const int " & rcVar & " = " & rhsElems & " / (2*NC);\n"
    s &= p & "for (int _i = 0; _i < NC; _i++) {\n"
    s &= p & "  for (int _j = 0; _j < " & rcVar & "; _j++) {\n"
    s &= p & "    simd_v _sre = simd_setzero();\n"
    s &= p & "    simd_v _sim = simd_setzero();\n"
    s &= p & "    for (int _k = 0; _k < " & lcVar & "; _k++) {\n"
    # a[i,k]: re at flat index 2*(i*lcVar+k), im at 2*(i*lcVar+k)+1
    # b[k,j]: re at flat index 2*(k*rcVar+j), im at 2*(k*rcVar+j)+1
    s &= p & "      simd_cmadd(&_sre, &_sim,\n"
    s &= p & "        " & lhs & "[2*(_i*" & lcVar & "+_k)], " & lhs & "[2*(_i*" & lcVar & "+_k)+1],\n"
    s &= p & "        " & rhs & "[2*(_k*" & rcVar & "+_j)], " & rhs & "[2*(_k*" & rcVar & "+_j)+1]);\n"
    s &= p & "    }\n"
    s &= p & "    " & target & "[2*(_i*" & rcVar & "+_j)] = _sre;\n"
    s &= p & "    " & target & "[2*(_i*" & rcVar & "+_j)+1] = _sim;\n"
    s &= p & "  }\n"
    s &= p & "}\n"
  else:
    s &= p & "const int " & lcVar & " = " & lhsElems & " / NC;\n"
    s &= p & "const int " & rcVar & " = " & rhsElems & " / NC;\n"
    s &= p & "for (int _i = 0; _i < NC; _i++) {\n"
    s &= p & "  for (int _j = 0; _j < " & rcVar & "; _j++) {\n"
    s &= p & "    simd_v _s = simd_setzero();\n"
    s &= p & "    for (int _k = 0; _k < " & lcVar & "; _k++)\n"
    s &= p & "      _s = simd_fmadd(" & lhs & "[_i*" & lcVar & "+_k], " & rhs & "[_k*" & rcVar & "+_j], _s);\n"
    s &= p & "    " & target & "[_i*" & rcVar & "+_j] = _s;\n"
    s &= p & "  }\n"
    s &= p & "}\n"
  let resultElems = if ctx.isComplex: "2 * NC * " & rcVar else: "NC * " & rcVar
  return (s, resultElems)

proc emitAdjoint*(target, src, srcElems: string, ctx: var OmpCodeCtx, d: int): MatResult =
  ## SIMD adjoint (conjugate transpose): all elements are simd_v.
  ## For complex: transpose matrix indices AND negate imaginary part.
  ##   elems = elementsPerSite = 2*NC*NC, tensor cols = elems/(2*NC).
  ## For real: just transpose matrix indices. elems = NC*NC, cols = elems/NC.
  var s = ""
  let p = ind(d)
  let colsVar = ctx.nextTmp() & "_ac"
  if ctx.isComplex:
    s &= p & "const int " & colsVar & " = " & srcElems & " / (2*NC);\n"
    s &= p & "for (int _i = 0; _i < NC; _i++)\n"
    s &= p & "  for (int _j = 0; _j < " & colsVar & "; _j++) {\n"
    # src[i,j] re at 2*(i*cols+j), im at 2*(i*cols+j)+1
    # target[j,i] re at 2*(j*NC+i), im at 2*(j*NC+i)+1
    s &= p & "    " & target & "[2*(_j*NC+_i)] = " & src & "[2*(_i*" & colsVar & "+_j)];\n"
    s &= p & "    " & target & "[2*(_j*NC+_i)+1] = simd_neg(" & src & "[2*(_i*" & colsVar & "+_j)+1]);\n"
    s &= p & "  }\n"
  else:
    s &= p & "const int " & colsVar & " = " & srcElems & " / NC;\n"
    s &= p & "for (int _i = 0; _i < NC; _i++)\n"
    s &= p & "  for (int _j = 0; _j < " & colsVar & "; _j++)\n"
    s &= p & "    " & target & "[_j*NC+_i] = " & src & "[_i*" & colsVar & "+_j];\n"
  return (s, srcElems)

proc emitMatExpr*(target: string, n: NimNode, ctx: var OmpCodeCtx, d: int): MatResult =
  ## Recursively transpile a matrix expression to SIMD C code.
  ## All temporaries are arrays of simd_v (one per AoSoA element).
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
              let idx = ompTranspileScalar(n[2], ctx)
              if ctx.isComplex:
                return (p & target & "[0] = " & name & "[2*(" & idx & ")]; " & target & "[1] = " & name & "[2*(" & idx & ")+1];\n", "2")
              else:
                return (p & target & "[0] = " & name & "[" & idx & "];\n", "1")
            elif n.len == 4:
              let row = ompTranspileScalar(n[2], ctx)
              let col = ompTranspileScalar(n[3], ctx)
              let flatIdx = "(" & row & ")*NC+(" & col & ")"
              if ctx.isComplex:
                return (p & target & "[0] = " & name & "[2*(" & flatIdx & ")]; " & target & "[1] = " & name & "[2*(" & flatIdx & ")+1];\n", "2")
              else:
                return (p & target & "[0] = " & name & "[" & flatIdx & "];\n", "1")
        let sr = ompResolveSiteRef(n[1], n[2], ctx)
        return emitLoadView(target, sr, ctx, d)
      if fn == "adjoint" and n.len >= 2:
        let tmp = ctx.nextTmp()
        let p = ind(d)
        var elemsGuess = "NC*NC"
        if ctx.isComplex: elemsGuess = "2*NC*NC"
        var s = p & "simd_v " & tmp & "[" & elemsGuess & "];\n"
        let inner = emitMatExpr(tmp, n[1], ctx, d)
        s &= inner.code
        let adj = emitAdjoint(target, tmp, inner.elems, ctx, d)
        s &= adj.code
        return (s, adj.elems)
      if fn == "trace" and n.len >= 2:
        # trace(matexpr): sum diagonal elements → scalar result
        let tmp = ctx.nextTmp()
        let p = ind(d)
        var elemsGuess = "NC*NC"
        if ctx.isComplex: elemsGuess = "2*NC*NC"
        var s = p & "simd_v " & tmp & "[" & elemsGuess & "];\n"
        let inner = emitMatExpr(tmp, n[1], ctx, d)
        s &= inner.code
        if ctx.isComplex:
          s &= p & target & "[0] = simd_setzero(); " & target & "[1] = simd_setzero();\n"
          s &= p & "for (int _i = 0; _i < NC; _i++) {\n"
          s &= p & "  " & target & "[0] = simd_add(" & target & "[0], " & tmp & "[2*(_i*NC+_i)]);\n"
          s &= p & "  " & target & "[1] = simd_add(" & target & "[1], " & tmp & "[2*(_i*NC+_i)+1]);\n"
          s &= p & "}\n"
          return (s, "2")
        else:
          s &= p & target & "[0] = simd_setzero();\n"
          s &= p & "for (int _i = 0; _i < NC; _i++)\n"
          s &= p & "  " & target & "[0] = simd_add(" & target & "[0], " & tmp & "[_i*NC+_i]);\n"
          return (s, "1")

      if fn == "identity" or fn == "siteIdentity":
        # identity()/siteIdentity() — generate NC×NC identity matrix
        let p = ind(d)
        var s = ""
        if ctx.isComplex:
          s &= p & "for (int _e = 0; _e < 2*NC*NC; _e++) " & target & "[_e] = simd_setzero();\n"
          s &= p & "for (int _i = 0; _i < NC; _i++) " & target & "[2*(_i*NC+_i)] = simd_set1(1.0);\n"
          return (s, "2*NC*NC")
        else:
          s &= p & "for (int _e = 0; _e < NC*NC; _e++) " & target & "[_e] = simd_setzero();\n"
          s &= p & "for (int _i = 0; _i < NC; _i++) " & target & "[_i*NC+_i] = simd_set1(1.0);\n"
          return (s, "NC*NC")

      # --- Custom site operation dispatch ---
      let opIdx = lookupCustomOp(fn)
      if opIdx >= 0:
        let op = customSiteOps[opIdx]
        let p = ind(d)
        var s = ""
        var args: seq[string]
        var argElems: seq[string]
        var elemsGuess = if ctx.isComplex: "2*NC*NC" else: "NC*NC"
        for i in 1..op.arity:
          let argTmp = ctx.nextTmp()
          args.add argTmp
          s &= p & "simd_v " & argTmp & "[" & elemsGuess & "];\n"
          let argRes = emitMatExpr(argTmp, n[i], ctx, d)
          s &= argRes.code
          argElems.add argRes.elems
        s &= instantiateTemplate(op.codeTemplate, target, args, argElems,
                                  "simd_v", op.resultElems, d)
        return (s, op.resultElems)

    let defaultElems = if ctx.isComplex: "2*NC*NC" else: "NC*NC"
    return (ind(d) & "// unhandled call: " & n[0].strVal & "\n", defaultElems)

  of nnkSym:
    let name = n.strVal
    let p = ind(d)
    # Check if this is a float scalar (not a matrix array)
    var isFloat = false
    try: isFloat = isFloatSym(n)
    except: discard
    if isFloat:
      # Scalar float: broadcast to single element
      if ctx.isComplex:
        return (p & target & "[0] = simd_set1(" & name & "); " & target & "[1] = simd_setzero();\n", "2")
      else:
        return (p & target & "[0] = simd_set1(" & name & ");\n", "1")
    let lb = ctx.info.getLetBinding(name)
    if lb.kind == lbkMatMul or lb.kind == lbkOther:
      let elemsName = name & "_elems"
      return (p & "for (int _e = 0; _e < " & elemsName & "; _e++) " & target & "[_e] = " & name & "[_e];\n",
              elemsName)
    else:
      let elemsName = name & "_elems"
      return (p & "for (int _e = 0; _e < " & elemsName & "; _e++) " & target & "[_e] = " & name & "[_e];\n",
              elemsName)

  of nnkIntLit..nnkInt64Lit:
    let p = ind(d)
    let val = $n.intVal
    if ctx.isComplex:
      # Complex scalar literal: re = val for diagonal, im = 0
      return (p & "for (int _e = 0; _e < 2*NC*NC; _e += 2) { " & target & "[_e] = simd_set1(" & val & ".0); " & target & "[_e+1] = simd_setzero(); }\n", "2*NC*NC")
    else:
      return (p & "for (int _e = 0; _e < NC*NC; _e++) " & target & "[_e] = simd_set1(" & val & ".0);\n", "NC*NC")

  of nnkFloatLit..nnkFloat64Lit:
    let p = ind(d)
    let val = $n.floatVal
    if ctx.isComplex:
      return (p & "for (int _e = 0; _e < 2*NC*NC; _e += 2) { " & target & "[_e] = simd_set1(" & val & "); " & target & "[_e+1] = simd_setzero(); }\n", "2*NC*NC")
    else:
      return (p & "for (int _e = 0; _e < NC*NC; _e++) " & target & "[_e] = simd_set1(" & val & ");\n", "NC*NC")

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
          var elemsGuess = "NC*NC"
          if ctx.isComplex: elemsGuess = "2*NC*NC"
          var s = p & "simd_v " & tmpL & "[" & elemsGuess & "];\n"
          s &= p & "simd_v " & tmpR & "[" & elemsGuess & "];\n"
          let lRes = emitMatExpr(tmpL, n[1], ctx, d)
          s &= lRes.code
          let rRes = emitMatExpr(tmpR, n[2], ctx, d)
          s &= rRes.code
          let mmRes = emitMatMul(target, tmpL, tmpR, lRes.elems, rRes.elems, ctx, d)
          s &= mmRes.code
          return (s, mmRes.elems)
        else:
          # Element-wise multiply (scalar * tensor or tensor * scalar)
          let tmpL = ctx.nextTmp()
          let tmpR = ctx.nextTmp()
          let p = ind(d)
          var elemsGuess = "NC*NC"
          if ctx.isComplex: elemsGuess = "2*NC*NC"
          var s = p & "simd_v " & tmpL & "[" & elemsGuess & "];\n"
          s &= p & "simd_v " & tmpR & "[" & elemsGuess & "];\n"
          let lRes = emitMatExpr(tmpL, n[1], ctx, d)
          s &= lRes.code
          let rRes = emitMatExpr(tmpR, n[2], ctx, d)
          s &= rRes.code
          # Determine output elems — use the max of both sides
          # A real scalar has elems="1", complex scalar has elems="2",
          # a full matrix has elems="NC*NC" or "2*NC*NC".
          let outElems = if lRes.elems == "1" and rRes.elems != "1": rRes.elems
                         elif rRes.elems == "1" and lRes.elems != "1": lRes.elems
                         else: lRes.elems
          # Check if one side is a real scalar (elems="1") — broadcast multiply
          let lIsRealScalar = lRes.elems == "1"
          let rIsRealScalar = rRes.elems == "1"
          if lIsRealScalar or rIsRealScalar:
            # Real scalar × anything: just multiply all elements by the scalar
            let scalarTmp = if lIsRealScalar: tmpL else: tmpR
            let otherTmp = if lIsRealScalar: tmpR else: tmpL
            let otherElems = if lIsRealScalar: rRes.elems else: lRes.elems
            s &= p & "for (int _e = 0; _e < " & otherElems & "; _e++)\n"
            s &= p & "  " & target & "[_e] = simd_mul(" & scalarTmp & "[0], " & otherTmp & "[_e]);\n"
          elif ctx.isComplex:
            # Complex element-wise multiply: (are,aim) * (bre,bim)
            s &= p & "for (int _e = 0; _e < " & outElems & " / 2; _e++) {\n"
            s &= p & "  simd_v _tre, _tim;\n"
            s &= p & "  simd_cmul(&_tre, &_tim, " & tmpL & "[2*_e], " & tmpL & "[2*_e+1], " & tmpR & "[2*_e], " & tmpR & "[2*_e+1]);\n"
            s &= p & "  " & target & "[2*_e] = _tre;\n"
            s &= p & "  " & target & "[2*_e+1] = _tim;\n"
            s &= p & "}\n"
          else:
            s &= p & "for (int _e = 0; _e < " & outElems & "; _e++)\n"
            s &= p & "  " & target & "[_e] = simd_mul(" & tmpL & "[_e], " & tmpR & "[_e]);\n"
          return (s, outElems)

      if op == "+" or op == "-":
        let tmpL = ctx.nextTmp()
        let tmpR = ctx.nextTmp()
        let p = ind(d)
        var elemsGuess = "NC*NC"
        if ctx.isComplex: elemsGuess = "2*NC*NC"
        var s = p & "simd_v " & tmpL & "[" & elemsGuess & "];\n"
        s &= p & "simd_v " & tmpR & "[" & elemsGuess & "];\n"
        let lRes = emitMatExpr(tmpL, n[1], ctx, d)
        s &= lRes.code
        let rRes = emitMatExpr(tmpR, n[2], ctx, d)
        s &= rRes.code
        let outElems = lRes.elems
        if ctx.isComplex:
          # Complex +/- : re and im parts are separate simd_v entries.
          # outElems = elementsPerSite = 2*tensor_elems (already includes re,im).
          let simdOp = if op == "+": "simd_add" else: "simd_sub"
          s &= p & "for (int _e = 0; _e < " & outElems & "; _e++)\n"
          s &= p & "  " & target & "[_e] = " & simdOp & "(" & tmpL & "[_e], " & tmpR & "[_e]);\n"
        else:
          let simdOp = if op == "+": "simd_add" else: "simd_sub"
          s &= p & "for (int _e = 0; _e < " & outElems & "; _e++)\n"
          s &= p & "  " & target & "[_e] = " & simdOp & "(" & tmpL & "[_e], " & tmpR & "[_e]);\n"
        return (s, outElems)

  of nnkHiddenCallConv:
    if n.len >= 2:
      return emitMatExpr(target, n[1], ctx, d)

  of nnkHiddenDeref, nnkHiddenAddr, nnkHiddenStdConv, nnkConv, nnkHiddenSubConv:
    if n.len > 0:
      return emitMatExpr(target, n[^1], ctx, d)

  of nnkStmtListExpr:
    # Template expansion: process statements, then emit result expression
    if n.len >= 2:
      var s = ""
      for ci in 0..<n.len - 1:
        s &= ompTranspileStmt(n[ci], ctx, ctx.info, d)
      let resultExpr = n[n.len - 1]
      let matRes = emitMatExpr(target, resultExpr, ctx, d)
      s &= matRes.code
      return (s, matRes.elems)
    elif n.len == 1:
      return emitMatExpr(target, n[0], ctx, d)

  else:
    discard

  let defaultElems2 = if ctx.isComplex: "2*NC*NC" else: "NC*NC"
  return (ind(d) & "// unhandled expr kind: " & $n.kind & "\n", defaultElems2)

#[ --- Scalar expression transpilation --- ]#

proc ompTranspileScalar*(n: NimNode, ctx: var OmpCodeCtx): string =
  case n.kind
  of nnkSym:
    return n.strVal
  of nnkIntLit..nnkInt64Lit:
    return $n.intVal
  of nnkFloatLit..nnkFloat64Lit:
    return $n.floatVal
  of nnkHiddenStdConv, nnkConv, nnkHiddenDeref, nnkHiddenAddr, nnkHiddenSubConv:
    if n.len > 0: return ompTranspileScalar(n[^1], ctx)
  of nnkStmtListExpr:
    for i in countdown(n.len - 1, 0):
      if n[i].kind != nnkEmpty:
        return ompTranspileScalar(n[i], ctx)
  of nnkPrefix:
    if n.len >= 2 and n[0].kind == nnkSym:
      let op = n[0].strVal
      if op == "not":
        return "!(" & ompTranspileScalar(n[1], ctx) & ")"
      else:
        return op & "(" & ompTranspileScalar(n[1], ctx) & ")"
  of nnkCall:
    if n[0].kind == nnkSym:
      let fn = n[0].strVal
      # Element read on local matrix temp: m[i] or m[i,j]
      if fn == "[]" and n.len >= 3:
        let target = n[1]
        if target.kind == nnkSym:
          var isMat = false
          try: isMat = isMatrixTypedNode(target)
          except: discard
          if isMat:
            let name = target.strVal
            if n.len == 3:
              let idx = ompTranspileScalar(n[2], ctx)
              return name & "[" & idx & "]"
            elif n.len == 4:
              let row = ompTranspileScalar(n[2], ctx)
              let col = ompTranspileScalar(n[3], ctx)
              if ctx.isComplex:
                return name & "[2*((" & row & ")*NC+(" & col & "))]"
              else:
                return name & "[(" & row & ")*NC+(" & col & ")]"
            elif n.len >= 5:
              let i = ompTranspileScalar(n[2], ctx)
              let j = ompTranspileScalar(n[3], ctx)
              let k = ompTranspileScalar(n[4], ctx)
              return name & "[(" & i & ")*NC*NC+(" & j & ")*NC+(" & k & ")]"
      # conj — complex conjugate; in scalar context returns real part unchanged
      if fn == "conj" and n.len == 2:
        let inner = ompTranspileScalar(n[1], ctx)
        return inner  # real part is unchanged by conj; imag negate handled at stmt level
      var args: seq[string]
      for i in 1..<n.len: args.add ompTranspileScalar(n[i], ctx)
      return n[0].strVal & "(" & args.join(", ") & ")"
  of nnkInfix:
    if n.len >= 3 and n[0].kind == nnkSym:
      return "(" & ompTranspileScalar(n[1], ctx) & " " & n[0].strVal & " " & ompTranspileScalar(n[2], ctx) & ")"
  of nnkDotExpr:
    if n.len >= 2 and n[0].kind == nnkSym and n[1].kind == nnkSym:
      return n[0].strVal & "_" & n[1].strVal
    elif n.len >= 2:
      return ompTranspileScalar(n[0], ctx) & "_" & n[1].strVal
  else: discard
  return "0"

#[ ============================================================================
   Recursive Statement Dispatch
   ============================================================================ ]#

proc ompTranspileStmt(stmt: NimNode, ctx: var OmpCodeCtx, info: KernelInfo, d: int): string =
  ## Transpile a single statement inside an ``each`` loop body to OpenMP SIMD C.
  ## `d` is the indentation depth.  Called recursively for nested blocks
  ## (if-bodies, for-bodies, etc.).
  let p = ind(d)
  var elemsGuess = "NC*NC"
  if ctx.isComplex: elemsGuess = "2*NC*NC"

  case stmt.kind
  of nnkLetSection, nnkVarSection:
    var s = ""
    for idefs in stmt:
      if idefs.kind == nnkIdentDefs and idefs.len >= 3:
        let vn = idefs[0].strVal
        let val = idefs[2]
        let lb = info.getLetBinding(vn)

        case lb.kind
        of lbkStencilFwd:
          s &= p & "// fwd neighbor: " & vn & "\n"
          s &= p & "int " & vn & "_indices[VW];\n"
          s &= p & "for (int _lane = 0; _lane < VW; _lane++) {\n"
          s &= ind(d+1) & "int _site = group * VW + _lane;\n"
          s &= ind(d+1) & vn & "_indices[_lane] = " & lb.stencilName & "_offsets[_site * " & lb.stencilName & "_npts + 2 * " & lb.dirExpr & "];\n"
          s &= p & "}\n\n"
        of lbkStencilBwd:
          s &= p & "// bwd neighbor: " & vn & "\n"
          s &= p & "int " & vn & "_indices[VW];\n"
          s &= p & "for (int _lane = 0; _lane < VW; _lane++) {\n"
          s &= ind(d+1) & "int _site = group * VW + _lane;\n"
          s &= ind(d+1) & vn & "_indices[_lane] = " & lb.stencilName & "_offsets[_site * " & lb.stencilName & "_npts + 2 * " & lb.dirExpr & " + 1];\n"
          s &= p & "}\n\n"
        of lbkStencilNeighbor:
          s &= p & "// stencil neighbor: " & vn & "\n"
          s &= p & "int " & vn & "_indices[VW];\n"
          s &= p & "for (int _lane = 0; _lane < VW; _lane++) {\n"
          s &= ind(d+1) & "int _site = group * VW + _lane;\n"
          s &= ind(d+1) & vn & "_indices[_lane] = " & lb.stencilName & "_offsets[_site * " & lb.stencilName & "_npts + " & lb.pointExpr & "];\n"
          s &= p & "}\n\n"
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
          s &= p & "int " & prefix & "_ptIdx = 2*" & nd & " + " & prefix & "_pair * 4 + " & prefix & "_si;\n"
          s &= p & "int " & vn & "_indices[VW];\n"
          s &= p & "for (int _lane = 0; _lane < VW; _lane++) {\n"
          s &= ind(d+1) & "int _site = group * VW + _lane;\n"
          s &= ind(d+1) & vn & "_indices[_lane] = " & lb.stencilName & "_offsets[_site * " & lb.stencilName & "_npts + " & prefix & "_ptIdx];\n"
          s &= p & "}\n\n"
        of lbkMatMul:
          s &= p & "// matrix temp: " & vn & "\n"
          s &= p & "simd_v " & vn & "[" & elemsGuess & "];\n"
          if val.kind == nnkStmtListExpr and val.len >= 2:
            # Template expansion: process all but last child as statements,
            # then copy the result (last child) into target
            for ci in 0..<val.len - 1:
              s &= ompTranspileStmt(val[ci], ctx, info, d)
            let resultExpr = val[val.len - 1]
            let matRes = emitMatExpr(vn, resultExpr, ctx, d)
            s &= matRes.code
            s &= p & "const int " & vn & "_elems = " & matRes.elems & ";\n\n"
          elif val.kind != nnkEmpty:
            let matRes = emitMatExpr(vn, val, ctx, d)
            s &= matRes.code
            s &= p & "const int " & vn & "_elems = " & matRes.elems & ";\n\n"
          else:
            # Uninitialized var — just declare array, will be filled by for-loop
            let defaultElems = if ctx.isComplex: "2*NC*NC" else: "NC*NC"
            s &= p & "const int " & vn & "_elems = " & defaultElems & ";\n\n"
        of lbkOther:
          let code = ompTranspileScalar(val, ctx)
          s &= p & ctx.scalarType & " " & vn & " = " & code & ";\n"
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
          let code = ompTranspileScalar(rhs, ctx)
          return p & vn & " = " & code & ";\n"
    return ""

  of nnkInfix:
    if stmt.len >= 3 and stmt[0].kind == nnkSym and stmt[0].strVal == "+=":
      let lhs = stmt[1]
      let rhs = stmt[2]
      if lhs.kind == nnkCall and lhs[0].kind == nnkSym and lhs[0].strVal == "[]":
        let sr = ompResolveSiteRef(lhs[1], lhs[2], ctx)
        let tmp = ctx.nextTmp()
        var s = p & "{ // +=\n"
        s &= ind(d+1) & "simd_v " & tmp & "[" & elemsGuess & "];\n"
        let matRes = emitMatExpr(tmp, rhs, ctx, d+1)
        s &= matRes.code
        let storeElems = sr.elemsVar
        if sr.isGauge:
          for di in 0..<sr.gaugeDim:
            let cond = if di == 0: "if" else: "else if"
            s &= ind(d+1) & cond & " (" & sr.dirExpr & " == " & $di & ") {\n"
            s &= ind(d+2) & "for (int _e = 0; _e < " & storeElems & "; _e++) {\n"
            s &= ind(d+3) & "simd_v _cur = simd_load(" & aosoaBase(sr.gaugeName & "_" & $di & "_data", sr.groupVar, sr.elemsVar, "_e") & ");\n"
            s &= ind(d+3) & "simd_store(" & aosoaBase(sr.gaugeName & "_" & $di & "_data", sr.groupVar, sr.elemsVar, "_e") & ", simd_add(_cur, " & tmp & "[_e]));\n"
            s &= ind(d+2) & "}\n"
            s &= ind(d+1) & "}\n"
        else:
          s &= ind(d+1) & "for (int _e = 0; _e < " & storeElems & "; _e++) {\n"
          s &= ind(d+2) & "simd_v _cur = simd_load(" & aosoaBase(sr.dataVar, sr.groupVar, sr.elemsVar, "_e") & ");\n"
          s &= ind(d+2) & "simd_store(" & aosoaBase(sr.dataVar, sr.groupVar, sr.elemsVar, "_e") & ", simd_add(_cur, " & tmp & "[_e]));\n"
          s &= ind(d+1) & "}\n"
        s &= p & "}\n"
        return s
    return ""

  of nnkCall:
    if isAddFlopCall(stmt):
      return ""  # Skip addFLOPImpl — handled on host side
    elif stmt.len >= 2 and stmt[0].kind == nnkSym and stmt[0].strVal == "[]=":
      if isElementLevelWrite(stmt):
        let innerCall = stmt[1]
        var viewName = "output"
        if innerCall.len >= 3 and innerCall[1].kind == nnkSym:
          viewName = innerCall[1].strVal

        let siteNode = innerCall[2]
        var elGroupVar = "group"
        if siteNode.kind == nnkSym:
          let sn = siteNode.strVal
          if sn != ctx.loopVarStr:
            let lb = ctx.info.getLetBinding(sn)
            if lb.kind in {lbkStencilFwd, lbkStencilBwd, lbkStencilNeighbor, lbkStencilCorner}:
              elGroupVar = sn & "_group"

        if stmt.len == 4:
          let idxCode = ompTranspileScalar(stmt[2], ctx)
          let valCode = ompTranspileScalar(stmt[3], ctx)
          return p & "simd_store(" & aosoaBase(viewName & "_data", elGroupVar, viewName & "_elems", idxCode) & ", simd_set1(" & valCode & "));\n"
        elif stmt.len >= 5:
          let rowCode = ompTranspileScalar(stmt[2], ctx)
          let colCode = ompTranspileScalar(stmt[3], ctx)
          let valCode = ompTranspileScalar(stmt[4], ctx)
          let flatIdx = "(" & rowCode & ")*NC+(" & colCode & ")"
          return p & "simd_store(" & aosoaBase(viewName & "_data", elGroupVar, viewName & "_elems", flatIdx) & ", simd_set1(" & valCode & "));\n"
      else:
        # Check for local var element write: localVar[i,j] = val
        let target = stmt[1]
        if target.kind == nnkSym:
          let vn = target.strVal
          var isProxy = false
          try: isProxy = isTensorSiteProxySym(target)
          except: discard
          if isProxy:
            # Determine the flat index for the LHS
            var lhsFlatIdx = "0"
            if stmt.len == 4:
              lhsFlatIdx = ompTranspileScalar(stmt[2], ctx)
            elif stmt.len >= 5:
              let rowCode = ompTranspileScalar(stmt[2], ctx)
              let colCode = ompTranspileScalar(stmt[3], ctx)
              lhsFlatIdx = "(" & rowCode & ")*NC+(" & colCode & ")"
            let valNode = stmt[stmt.len - 1]

            # Check if RHS is an element read (possibly wrapped in conj)
            var rhsIsElemRead = false
            var rhsIsConj = false
            var rhsName = ""
            var rhsFlatIdx = ""
            var innerVal = valNode
            # Unwrap conj() if present
            if valNode.kind == nnkCall and valNode.len == 2 and
               valNode[0].kind == nnkSym and valNode[0].strVal == "conj":
              innerVal = valNode[1]
              rhsIsConj = true
            if innerVal.kind == nnkCall and innerVal.len >= 3 and
               innerVal[0].kind == nnkSym and innerVal[0].strVal == "[]":
              let rhsTarget = innerVal[1]
              if rhsTarget.kind == nnkSym:
                var rhsIsMat = false
                try: rhsIsMat = isMatrixTypedNode(rhsTarget)
                except: discard
                if rhsIsMat:
                  rhsIsElemRead = true
                  rhsName = rhsTarget.strVal
                  if innerVal.len == 3:
                    rhsFlatIdx = ompTranspileScalar(innerVal[2], ctx)
                  elif innerVal.len == 4:
                    let rr = ompTranspileScalar(innerVal[2], ctx)
                    let rc = ompTranspileScalar(innerVal[3], ctx)
                    rhsFlatIdx = "(" & rr & ")*NC+(" & rc & ")"

            if rhsIsElemRead:
              if ctx.isComplex:
                var s = p & vn & "[2*(" & lhsFlatIdx & ")] = " & rhsName & "[2*(" & rhsFlatIdx & ")];\n"
                if rhsIsConj:
                  s &= p & vn & "[2*(" & lhsFlatIdx & ")+1] = simd_neg(" & rhsName & "[2*(" & rhsFlatIdx & ")+1]);\n"
                else:
                  s &= p & vn & "[2*(" & lhsFlatIdx & ")+1] = " & rhsName & "[2*(" & rhsFlatIdx & ")+1];\n"
                return s
              else:
                return p & vn & "[" & lhsFlatIdx & "] = " & rhsName & "[" & rhsFlatIdx & "];\n"
            else:
              let valCode = ompTranspileScalar(valNode, ctx)
              if ctx.isComplex:
                var s = p & vn & "[2*(" & lhsFlatIdx & ")] = simd_set1(" & valCode & ");\n"
                s &= p & vn & "[2*(" & lhsFlatIdx & ")+1] = simd_setzero();\n"
                return s
              else:
                return p & vn & "[" & lhsFlatIdx & "] = simd_set1(" & valCode & ");\n"

        # Tensor-level: view[n] = matrix_expr
        let viewNode = stmt[1]
        let siteNode = stmt[2]
        let rhsNode = stmt[3]
        let sr = ompResolveSiteRef(viewNode, siteNode, ctx)
        let tmp = ctx.nextTmp()
        var s = p & "{ // assign\n"
        s &= ind(d+1) & "simd_v " & tmp & "[" & elemsGuess & "];\n"
        let matRes = emitMatExpr(tmp, rhsNode, ctx, d+1)
        s &= matRes.code
        let storeElems = sr.elemsVar
        if sr.isGauge:
          for di in 0..<sr.gaugeDim:
            let cond = if di == 0: "if" else: "else if"
            s &= ind(d+1) & cond & " (" & sr.dirExpr & " == " & $di & ") {\n"
            s &= ind(d+2) & "for (int _e = 0; _e < " & storeElems & "; _e++)\n"
            s &= ind(d+3) & "simd_store(" & aosoaBase(sr.gaugeName & "_" & $di & "_data", sr.groupVar, sr.elemsVar, "_e") & ", " & tmp & "[_e]);\n"
            s &= ind(d+1) & "}\n"
        else:
          s &= ind(d+1) & "for (int _e = 0; _e < " & storeElems & "; _e++)\n"
          s &= ind(d+2) & "simd_store(" & aosoaBase(sr.dataVar, sr.groupVar, sr.elemsVar, "_e") & ", " & tmp & "[_e]);\n"
        s &= p & "}\n"
        return s
    return ""

  of nnkIfStmt:
    var s = ""
    for branch in stmt:
      if branch.kind == nnkElifBranch:
        let condCode = ompTranspileScalar(branch[0], ctx)
        s &= p & "if (" & condCode & ") {\n"
        var innerStmts: seq[NimNode]
        let bodyStmt = branch[1]
        if bodyStmt.kind == nnkStmtList:
          for child in bodyStmt: innerStmts.add child
        else:
          innerStmts.add bodyStmt
        for inner in innerStmts:
          s &= ompTranspileStmt(inner, ctx, info, d+1)
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
          s &= ompTranspileStmt(inner, ctx, info, d+1)
        s &= p & "}\n"
    return s

  of nnkForStmt:
    # for i in 0..<NC  →  for (int i = 0; i < NC; i++)
    if stmt.len >= 3:
      let loopVarNode = stmt[0]
      let rangeNode = stmt[1]
      let forBody = stmt[2]
      let iterVar = if loopVarNode.kind == nnkSym: loopVarNode.strVal else: "i"

      var loExpr = "0"
      var hiExpr = "NC"
      if rangeNode.kind == nnkCall and rangeNode.len >= 3:
        if rangeNode[0].kind == nnkSym:
          let fn = rangeNode[0].strVal
          loExpr = ompTranspileScalar(rangeNode[1], ctx)
          if fn == "..<":
            hiExpr = ompTranspileScalar(rangeNode[2], ctx)
          elif fn == "countup":
            hiExpr = "(" & ompTranspileScalar(rangeNode[2], ctx) & "+1)"
      elif rangeNode.kind == nnkInfix and rangeNode.len >= 3:
        if rangeNode[0].kind == nnkSym and rangeNode[0].strVal == "..<":
          loExpr = ompTranspileScalar(rangeNode[1], ctx)
          hiExpr = ompTranspileScalar(rangeNode[2], ctx)

      var s = p & "for (int " & iterVar & " = " & loExpr & "; " & iterVar & " < " & hiExpr & "; " & iterVar & "++) {\n"
      var innerStmts: seq[NimNode]
      if forBody.kind == nnkStmtList:
        for child in forBody: innerStmts.add child
      else:
        innerStmts.add forBody
      for inner in innerStmts:
        s &= ompTranspileStmt(inner, ctx, info, d+1)
      s &= p & "}\n"
      return s
    return ""

  of nnkDiscardStmt:
    return ""

  else:
    return p & "// skipped: " & $stmt.kind & "\n"

#[ ============================================================================
   C Function Source Assembly
   ============================================================================ ]#

var ompKernelCounter {.compileTime.} = 0

proc generateOmpFunction(body: NimNode, info: KernelInfo): tuple[funcSrc: string, funcName: string] =
  ## Generate a complete C function string with ``#pragma omp parallel for``.
  ## The function takes raw pointers to AoSoA buffers and scalar parameters.
  ##
  ## Loop structure (SIMD intrinsic):
  ##   #pragma omp parallel for
  ##   for (group = 0; group < numGroups; ++group) {
  ##     // Each tensor element is a simd_v vector (VW doubles)
  ##     // All arithmetic uses explicit SIMD intrinsics — no scalar fallback
  ##   }
  ##
  ## The outer loop distributes groups across threads. Within each group,
  ## VW sites are processed simultaneously using SIMD intrinsics.
  var ctx = newOmpCodeCtx(info)
  let vw = $VectorWidth

  ompKernelCounter += 1
  let funcName = "omp_each_" & $body.lineInfoObj.line & "_" & $ompKernelCounter

  var src = ""

  # Define VW and SIMD type before including simd_intrinsics.h
  src &= "#define VW " & vw & "\n"
  src &= "#define " & ctx.simdUseMacro() & "\n"
  src &= "#include \"" & simdHeaderPath & "\"\n\n"

  # Build parameter list — numGroups replaces numSites
  var params: seq[string]
  for v in info.views:
    params.add ctx.cPtrType() & v.name & "_data"
  for gv in info.gaugeViews:
    for d in 0..<gv.dim:
      params.add ctx.cPtrType() & gv.name & "_" & $d & "_data"
  for v in info.views:
    params.add "const int " & v.name & "_elems"
  for gv in info.gaugeViews:
    params.add "const int " & gv.name & "_elems"
  params.add "const int numGroups"
  params.add "const int NC"
  for s in info.stencils:
    params.add "const int* " & s.name & "_offsets"
    params.add "const int " & s.name & "_npts"
  let runtimeVars = findRuntimeIntVars(body, info)
  for rv in runtimeVars:
    params.add "const int " & rv.name
  let runtimeFloatVars = findRuntimeFloatVars(body, info)
  for rv in runtimeFloatVars:
    params.add "const double " & rv.name
  # Runtime dot-accessed float vars (e.g. c.cp -> c_cp)
  let runtimeDotFloatVars = findRuntimeDotFloatVars(body, info)
  for rv in runtimeDotFloatVars:
    params.add "const double " & rv.name
  # Runtime dot-accessed int vars (e.g. c.someInt -> c_someInt)
  let runtimeDotIntVars = findRuntimeDotIntVars(body, info)
  for rv in runtimeDotIntVars:
    params.add "const int " & rv.name

  src &= "static void " & funcName & "(\n"
  src &= "    " & params.join(",\n    ")
  src &= "\n) {\n"
  src &= "  #pragma omp parallel for schedule(static)\n"
  src &= "  for (int group = 0; group < numGroups; ++group) {\n"

  # Process each statement using recursive dispatch
  var stmts: seq[NimNode]
  if body.kind == nnkStmtList:
    for child in body: stmts.add child
  else:
    stmts.add body

  for stmt in stmts:
    src &= ompTranspileStmt(stmt, ctx, info, 2)

  src &= "  }\n"  # end for group
  src &= "}\n"    # end function
  return (src, funcName)

#[ ============================================================================
   Typed Each Implementation — C Codegen
   ============================================================================ ]#

macro eachImpl*(loopVar: untyped, lo: typed, hi: typed, body: typed): untyped =
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

  let (funcSrc, funcName) = generateOmpFunction(body, info)

  # Write the C function to a file at compile time, then -include it via passC
  # This avoids the problem where {.emit.} inside a proc body places C code
  # inside the generated C function body instead of at file scope.
  let kernelDir = querySetting(SingleValueSetting.nimcacheDir) & "/omp_kernels"
  discard staticExec("mkdir -p " & kernelDir)
  let kernelFile = kernelDir & "/" & funcName & ".c"
  writeFile(kernelFile, funcSrc)
  let passCPragma = newNimNode(nnkPragma).add(
    newNimNode(nnkExprColonExpr).add(ident"passC", newLit("-include " & kernelFile)))

  # Find a view sym for NC computation
  var shapeViewSym: NimNode = nil
  for v in info.views:
    shapeViewSym = v.nimSym; break
  if shapeViewSym == nil and info.gaugeViews.len > 0:
    shapeViewSym = info.gaugeViews[0].nimSym

  # Build an {.importc, cdecl, nodecl.} proc declaration and normal Nim call
  var argSetupStmts = newStmtList()

  let ncSym = genSym(nskLet, "nc")
  let nsSym = genSym(nskLet, "ns")

  if info.gaugeViews.len > 0:
    # NC always comes from the gauge view (matrix dimension)
    let gSym = info.gaugeViews[0].nimSym
    if info.views.len == 0:
      argSetupStmts.add quote do:
        let `ncSym` = `gSym`.field[0].shape[0]
        let `nsSym` = `gSym`.field[0].numSites()
    else:
      argSetupStmts.add quote do:
        let `ncSym` = `gSym`.field[0].shape[0]
        let `nsSym` = `shapeViewSym`.numSites()
  else:
    argSetupStmts.add quote do:
      let `ncSym` = if `shapeViewSym`.shape.len >= 1: `shapeViewSym`.shape[0] else: 1
      let `nsSym` = `shapeViewSym`.numSites()

  # Build parameter list for the importc proc and call arguments
  var params = newNimNode(nnkFormalParams)
  params.add newEmptyNode()  # void return

  var callArgs: seq[NimNode]
  var paramIdx = 0

  template addParam(argExpr: NimNode, argType: NimNode) =
    let pname = ident("p" & $paramIdx)
    params.add newIdentDefs(pname, argType)
    callArgs.add argExpr
    paramIdx += 1

  # View data pointers
  for v in info.views:
    let dataSym = v.nimSym
    let dataExpr = quote do: cast[pointer](`dataSym`.data.aosoaData)
    addParam(dataExpr, ident"pointer")

  # Gauge view data pointers (D per view)
  for gv in info.gaugeViews:
    for d in 0..<gv.dim:
      let gSym = gv.nimSym
      let dLit = newLit(d)
      let dataExpr = quote do: cast[pointer](`gSym`.field[`dLit`].data.aosoaData)
      addParam(dataExpr, ident"pointer")

  # Per-view elems — always use elementsPerSite (count of doubles), since
  # SIMD works with double* pointers. For complex fields, elementsPerSite
  # = 2*NC*NC (re and im are separate contiguous VW-wide AoSoA slots).
  for v in info.views:
    let vSym = v.nimSym
    let elemsExpr = quote do: `vSym`.data.elementsPerSite.cint
    addParam(elemsExpr, ident"cint")

  for gv in info.gaugeViews:
    let gSym = gv.nimSym
    let elemsExpr = quote do: `gSym`.field[0].data.elementsPerSite.cint
    addParam(elemsExpr, ident"cint")

  # numGroups = ceil(numSites / VectorWidth)
  let vwLit = newLit(VectorWidth)
  let ngExpr = quote do: ((`nsSym` + `vwLit` - 1) div `vwLit`).cint
  addParam(ngExpr, ident"cint")

  # NC
  let ncExpr = quote do: `ncSym`.cint
  addParam(ncExpr, ident"cint")

  # Stencil args: offsets pointer + nPoints
  for s in info.stencils:
    let sSym = s.nimSym
    let offsetExpr = quote do: cast[pointer](addr `sSym`.offsets[0])
    addParam(offsetExpr, ident"pointer")
    let npExpr = quote do: `sSym`.nPoints.cint
    addParam(npExpr, ident"cint")

  # Runtime int vars
  let runtimeVars = findRuntimeIntVars(body, info)
  for rv in runtimeVars:
    let rvSym = rv.sym
    let rvExpr = quote do: `rvSym`.cint
    addParam(rvExpr, ident"cint")

  # Runtime float vars
  let runtimeFloatVars = findRuntimeFloatVars(body, info)
  for rv in runtimeFloatVars:
    let rvSym = rv.sym
    let rvExpr = quote do: `rvSym`.cdouble
    addParam(rvExpr, ident"cdouble")

  # Runtime dot-accessed float vars (e.g. c.cp -> c_cp)
  let runtimeDotFloatVars = findRuntimeDotFloatVars(body, info)
  for rv in runtimeDotFloatVars:
    let dotNode = rv.dotNode
    # Rebuild dot expr from scratch: objSym.fieldIdent (avoid skField symbol)
    let freshDot = nnkDotExpr.newTree(dotNode[0], ident(dotNode[1].strVal))
    let convExpr = newCall(ident"cdouble", freshDot)
    addParam(convExpr, ident"cdouble")

  # Runtime dot-accessed int vars (e.g. c.someInt -> c_someInt)
  let runtimeDotIntVars = findRuntimeDotIntVars(body, info)
  for rv in runtimeDotIntVars:
    let dotNode = rv.dotNode
    let freshDot = nnkDotExpr.newTree(dotNode[0], ident(dotNode[1].strVal))
    let convExpr = newCall(ident"cint", freshDot)
    addParam(convExpr, ident"cint")

  # Build the importc proc declaration
  let wrapperName = genSym(nskProc, funcName)
  let funcNameLit = newLit(funcName)
  let procDecl = newNimNode(nnkProcDef).add(
    wrapperName,           # name
    newEmptyNode(),        # pattern
    newEmptyNode(),        # generic params
    params,                # formal params
    newNimNode(nnkPragma).add(  # pragmas
      newNimNode(nnkExprColonExpr).add(ident"importc", funcNameLit),
      ident"cdecl",
      ident"nodecl"
    ),
    newEmptyNode(),        # reserved
    newEmptyNode()         # body (empty for importc)
  )

  # Build the call
  var callNode = newNimNode(nnkCall).add(wrapperName)
  for arg in callArgs:
    callNode.add arg

  result = newStmtList(
    passCPragma,
    argSetupStmts,
    procDecl,
    callNode
  )

#[ ============================================================================
   Public Each Macro (ForLoopStmt)
   ============================================================================ ]#

macro each*(forLoop: ForLoopStmt): untyped =
  ## OpenMP parallel each loop for TensorFieldView.
  ##
  ## Transpiles the loop body to a complete C function with
  ## ``#pragma omp parallel for`` and emits it at file scope.
  ## Handles all expression patterns (copy, add, sub, matmul,
  ## scalar ops, stencil, gauge, adjoint, element writes, etc.).
  ##
  ## Echo/debugEcho falls back to a sequential CPU loop.
  ##
  ## Usage:
  ##   for n in each view.all:
  ##     viewC[n] = viewA[n] + viewB[n]

  let loopVar = forLoop[0]
  let loopRangeNode = forLoop[1][1]  # Skip 'each' wrapper
  let body = forLoop[2]

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
  elif loopRangeNode.kind == nnkDotExpr and loopRangeNode.len >= 2:
    # Handle view.all before template expansion
    if loopRangeNode[1].eqIdent("all"):
      isRange = true
      startExpr = newLit(0)
      endExpr = newCall(ident"numSites", loopRangeNode[0])

  if isRange:
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
  echo "OpenMP C codegen dispatch module loaded"
  echo "Max threads: ", getNumThreads()