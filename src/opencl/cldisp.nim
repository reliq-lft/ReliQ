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

import std/[macros]
import std/[tables]
import std/[genasts]
import std/[strutils]

import clbase

type VarInfo* = tuple[name: string, typ: NimNode]

proc nimTypeToOpenCL*(typ: NimNode): string =
  ## Converts a Nim type node to OpenCL C type string
  if typ.kind == nnkEmpty: return "double"  # fallback for untyped
  
  let typeStr = typ.repr
  
  # Handle GpuBuffer[T] -> T*
  if typeStr.startsWith("GpuBuffer["):
    let elementType = typeStr[10..^2]  # extract T from GpuBuffer[T]
    return nimTypeToOpenCL(newIdentNode(elementType)) & "*"
  
  # Handle pointer types: ptr T, ptr UncheckedArray[T]
  if typeStr.startsWith("ptr "):
    let inner = typeStr[4..^1]
    if inner.startsWith("UncheckedArray["):
      # ptr UncheckedArray[T] -> T*
      let elementType = inner[15..^2]  # extract T from UncheckedArray[T]
      return nimTypeToOpenCL(newIdentNode(elementType)) & "*"
    else: return nimTypeToOpenCL(newIdentNode(inner)) & "*"
  
  # Handle seq[T] as T* (pointer to data)
  if typeStr.startsWith("seq["):
    let elementType = typeStr[4..^2]
    return nimTypeToOpenCL(newIdentNode(elementType)) & "*"
  
  # Handle PMem as double* (type-erased, fallback)
  if typeStr == "Pmem": return "double*"
  
  # Basic type mapping
  case typeStr
  of "float", "float64": return "double"
  of "float32": return "float"
  of "int", "int64": return "long"
  of "int32": return "int"
  of "int16": return "short"
  of "int8": return "char"
  of "uint", "uint64": return "ulong"
  of "uint32": return "uint"
  of "uint16": return "ushort"
  of "uint8": return "uchar"
  of "bool": return "bool"
  else:
    # For complex types, try to extract from BracketExpr
    if typ.kind == nnkBracketExpr:
      let baseType = typ[0].strVal
      # Handle GpuBuffer[T]
      if baseType == "GpuBuffer": return nimTypeToOpenCL(typ[1]) & "*"
      if baseType in ["ptr", "UncheckedArray"]: return nimTypeToOpenCL(typ[1]) & "*"
      elif baseType == "seq": return nimTypeToOpenCL(typ[1]) & "*"
    return "double" # fallback

proc gatherVariablesTyped*(n: NimNode): seq[VarInfo] =
  ## Gathers variables from typed AST with full type information
  result = @[]
  
  proc gatherRecursive(node: NimNode, acc: var seq[VarInfo]) =
    # Handle typed bracket access: Call(Sym "[]", Sym "gpuA", Sym "i")
    if node.kind == nnkCall:
      if node.len >= 2 and node[0].kind == nnkSym:
        let opName = node[0].strVal
        if opName == "[]" or opName == "[]=":
          # node[1] is the array variable
          let base = node[1]
          if base.kind == nnkSym:
            let name = base.strVal
            let typ = base.getTypeInst()
            # Avoid duplicates
            var found = false
            for v in acc:
              if v.name == name:
                found = true
                break
            if not found:
              acc.add (name: name, typ: typ)
    
    # Also check untyped bracket expressions (fallback)
    if node.kind == nnkBracketExpr:
      let base = node[0]
      if base.kind == nnkSym:
        let name = base.strVal
        let typ = base.getTypeInst()
        var found = false
        for v in acc:
          if v.name == name:
            found = true
            break
        if not found: acc.add (name: name, typ: typ)
      elif base.kind == nnkIdent:
        let name = base.strVal
        var found = false
        for v in acc:
          if v.name == name:
            found = true
            break
        if not found: acc.add (name: name, typ: newEmptyNode())
    
    # Recurse into children
    for child in node.children: gatherRecursive(child, acc)
  
  gatherRecursive(n, result)

proc toC(n: NimNode): string =
  ## Convert AST to C-like string
  case n.kind
  of nnkIdent, nnkSym: return n.strVal
  of nnkIntLit..nnkInt64Lit: return $n.intVal
  of nnkFloatLit..nnkFloat64Lit: return $n.floatVal
  of nnkBracketExpr: return toC(n[0]) & "[" & toC(n[1]) & "]"
  of nnkInfix:
    let op = n[0].strVal
    return "(" & toC(n[1]) & " " & op & " " & toC(n[2]) & ")"
  of nnkAsgn:
    return toC(n[0]) & " = " & toC(n[1]) & ";"
  of nnkStmtList:
    var lines: seq[string]
    for child in n:
      let line = toC(child)
      if line.len > 0: lines.add line
    return lines.join("\n  ")
  of nnkPrefix: return n[0].strVal & toC(n[1])
  of nnkPar: return "(" & toC(n[0]) & ")"
  of nnkCall:
    # Handle typed bracket operators: Call(Sym "[]", arr, idx) -> arr[idx]
    if n.len >= 2 and n[0].kind == nnkSym:
      let opName = n[0].strVal
      if opName == "[]":
        # n[1] = array, n[2] = index
        return toC(n[1]) & "[" & toC(n[2]) & "]"
      elif opName == "[]=":
        # n[1] = array, n[2] = index, n[3] = value
        return toC(n[1]) & "[" & toC(n[2]) & "] = " & toC(n[3]) & ";"
    # Regular function call
    var args: seq[string]
    for i in 1..<n.len: args.add toC(n[i])
    return toC(n[0]) & "(" & args.join(", ") & ")"
  of nnkIfStmt, nnkIfExpr:
    var code = ""
    for i, branch in n:
      if branch.kind == nnkElifBranch:
        if i == 0:
          code &= "if (" & toC(branch[0]) & ") {\n    " & toC(branch[1]) & "\n  }"
        else:
          code &= " else if (" & toC(branch[0]) & ") {\n    " & toC(branch[1]) & "\n  }"
      elif branch.kind == nnkElse:
        code &= " else {\n    " & toC(branch[0]) & "\n  }"
    return code
  of nnkForStmt:
    # for i in lo..<hi: body  ->  for (int i = lo; i < hi; i++) { body }
    let loopVar = n[0].strVal
    let iterExpr = n[1]
    let body = n[2]
    # Handle range expressions: lo..<hi or lo..hi
    if iterExpr.kind == nnkInfix:
      let op = iterExpr[0].strVal
      let lo = toC(iterExpr[1])
      let hi = toC(iterExpr[2])
      if op == "..<":
        return "for (int " & loopVar & " = " & lo & "; " & loopVar & " < " & hi & "; " & loopVar & "++) {\n    " & toC(body) & "\n  }"
      elif op == "..":
        return "for (int " & loopVar & " = " & lo & "; " & loopVar & " <= " & hi & "; " & loopVar & "++) {\n    " & toC(body) & "\n  }"
    return "/* unsupported for loop */"
  of nnkWhileStmt: return "while (" & toC(n[0]) & ") {\n    " & toC(n[1]) & "\n  }"
  of nnkVarSection, nnkLetSection:
    var decls: seq[string]
    for def in n:
      if def.kind == nnkIdentDefs:
        let varName = def[0].strVal
        let varType = if def[1].kind != nnkEmpty: def[1] else: def[2].getType()
        let typeStr = nimTypeToOpenCL(varType)
        if def[2].kind != nnkEmpty:
          decls.add typeStr & " " & varName & " = " & toC(def[2]) & ";"
        else: decls.add typeStr & " " & varName & ";"
    return decls.join("\n  ")
  of nnkHiddenStdConv, nnkHiddenDeref: return toC(n[1])
  of nnkStmtListExpr:
    # Typed AST sometimes wraps expressions in StmtListExpr
    # The last child is the actual expression
    if n.len > 0: return toC(n[^1])
    return ""
  of nnkDiscardStmt:
    if n[0].kind != nnkEmpty: return toC(n[0]) & ";"
    return ""
  of nnkBreakStmt: return "break;"
  of nnkContinueStmt: return "continue;"
  else: return "/* unsupported: " & $n.kind & " */"

macro eachImpl*(loopVar: untyped, lo: typed, hi: typed, body: typed): untyped =
  ## Internal typed macro that receives the body with type information
  let idntName = loopVar.strVal
  
  # Gather variables with type info from typed body
  let variables = gatherVariablesTyped(body)
  
  # Generate unique kernel name
  let kernelName = "generatedKernel_" & $body.lineInfoObj.line
  
  # Build parameter list with detected types
  var params: seq[string]
  for (varName, typ) in variables:
    let clType = nimTypeToOpenCL(typ)
    # For pointers, the type already includes *, otherwise add __global *
    if clType.endsWith("*"): params.add "__global " & clType & " " & varName
    else: params.add "__global " & clType & "* " & varName
  
  # Add offset parameter for multi-device support
  params.add "int offset"
  let offsetArgIndex = newLit(variables.len)  # offset is the last argument
  
  let bodyC = toC(body)
  
  # Check if we need double precision
  var needsFp64 = false
  for (_, typ) in variables:
    let clType = nimTypeToOpenCL(typ)
    if "double" in clType:
      needsFp64 = true
      break
  
  var kernelSource = ""
  if needsFp64: kernelSource = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
  kernelSource &= "__kernel void " & kernelName & "(" & params.join(", ") & ") {\n"
  kernelSource &= "  int " & idntName & " = get_global_id(0) + offset;\n"
  kernelSource &= "  " & bodyC & "\n"
  kernelSource &= "}\n"
  
  let kernelSourceLit = newLit(kernelSource)
  let kernelNameLit = newLit(kernelName)
  
  # Build the result manually to include argument setting
  let clKernelSym = ident("clKernel")
  let clProgramSym = ident("clProgram")
  
  # Build argument setting: clKernel.args(gpuA, gpuB, gpuC, ...)
  # Note: offset is set per-device in the dispatch loop
  var argsCall = newNimNode(nnkCall)
  argsCall.add newDotExpr(clKernelSym, ident("args"))
  for (varName, _) in variables: argsCall.add ident(varName)
  
  # Generate the dispatch code with multi-device support
  result = genAst(
    kernelSourceLit, 
    kernelNameLit, 
    hi, 
    argsCall, 
    offsetArgIndex,
    clProgramSym, 
    clKernelSym,
    clContext = ident("clContext"),
    clDevices = ident("clDevices"),
    clQueues = ident("clQueues")
  ):
    block:
      # Compile the kernel
      let clProgramSym = createAndBuild(clContext, kernelSourceLit, clDevices)
      let clKernelSym = clProgramSym.createKernel(kernelNameLit)
      
      # Set kernel arguments (shared across devices)
      argsCall
      
      # Dispatch work across all devices
      let numDevices = clQueues.len
      let totalWork = hi
      let workPerDevice = (totalWork + numDevices - 1) div numDevices
      
      for deviceIdx in 0 ..< numDevices:
        let deviceOffset = deviceIdx * workPerDevice
        let deviceWork = min(workPerDevice, totalWork - deviceOffset)
        if deviceWork > 0:
          # Set the offset for this device
          var offset = deviceOffset.int32
          clKernelSym.setArg(offset, offsetArgIndex)
          clQueues[deviceIdx].run(clKernelSym, deviceWork)
      
      # Wait for all devices to complete
      for deviceIdx in 0 ..< numDevices: check clwrap.finish(clQueues[deviceIdx])

macro each*(x: ForLoopStmt): untyped =
  ## Transforms a for loop into an OpenCL kernel dispatch with automatic type detection.
  ## 
  ## Usage:
  ##   for i in each 0..<N:
  ##     gpuC[i] = gpuA[i] + gpuB[i]
  ##
  ## This generates an OpenCL kernel and dispatches it across available devices.
  ## Uses typed macro (eachImpl) internally for automatic type detection from GpuBuffer[T].
  
  let (idnt, call, body) = (x[0], x[1], x[2])
  let itr = call[1]
  let (lo, hi) = (itr[1], itr[^1])
  
  # We need to type the body in a context where the loop variable exists.
  # Create a block that defines the loop variable and then calls eachImpl.
  result = quote do:
    block:
      var `idnt` {.inject.}: int = 0
      eachImpl(`idnt`, `lo`, `hi`, `body`)