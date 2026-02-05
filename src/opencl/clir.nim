#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/opencl/clir.nim
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
import std/[strutils]
import std/[sequtils]
import std/[options]

## OpenCL intermediate representation
## 
## Provides facilities for OpenCL kernel generation via a GPU abstract syntax tree.
## This is based on the Constantine's implementation of a GPU AST for CUDA:
## - https://github.com/mratsim/constantine/tree/0900bc0b140804355f7d5cfd9601efbbebd30418/constantine/math_compiler/experimental
## - https://forum.nim-lang.org/t/12868

type # OpenCL node kinds
  ClNodeKind* = enum
    clnVoid,        # empty
    clnProc,        # procedure
    clnCall,        # procedure call
    clnIf,          # if statement
    clnFor,         # for loop
    clnWhile,       # while loop
    clnBinOp,       # binary operation
    clnVar,         # variable declaration
    clnAssign,      # assignment
    clnIdent,       # identifier
    clnLit,         # literal value
    clnArrayLit,    # array literal; e.g., [1, 2, 3]
    clnPrefix,      # prefix; e.g., `-`, `not`
    clnBlock,       # block
    clnReturn,      # return
    clnDot,         # object member access; e.g., x.attr
    clnIndex,       # array index; e.g., arr[i]
    clnTypeDef,     # struct
    clnObjConstr,   # struct constructor
    clnAddr,        # address of expression
    clnDeref,       # dereference expression
    clnCast,        # cast expressions
    clnComment,     # comment
    clnConstexpr,   # compile-time constant
    clnVecConstr    # vector type constructor

type # OpenCL type kinds and types
  ClTypeKind* = enum
    cltVoid,        # empty
    cltBool,        # boolean
    cltUint8,       # unsigned 8-bit integer
    cltUint16,      # unsigned 16-bit integer
    cltUint32,      # unsigned 32-bit integer
    cltUint64,      # unsigned 64-bit integer
    cltInt8,        # signed 8-bit integer
    cltInt16,       # signed 16-bit integer
    cltInt32,       # signed 32-bit integer
    cltInt64,       # signed 64-bit integer
    cltFloat32,     # 32-bit floating point
    cltFloat64,     # 64-bit floating point
    cltSize_t,      # size_t int
    cltArray,       # static array
    cltString,      # string
    cltObject,      # struct
    cltPtr,         # pointer
    cltVoidPtr      # void pointer
    # TODO: handle vector types

  ClType* = ref object
    case kind*: ClTypeKind
    of cltPtr:
      to*: ClType
    of cltObject:
      name*: string
      oFields*: seq[ClTypeField]
    of cltArray:
      aTyp*: ClType
      aLen*: int
    else: discard

  ClTypeField* = object
    name*: string
    typ*: ClType

type # OpenCL attributes
  ClAttribute* = enum
    attKernel = "__kernel",
    attInline = "inline"

  ClVarAttribute* = enum
    atvGlobal = "__global",
    atvLocal = "__local",
    atvConstant = "__constant",
    atvPrivate = "__private"

type # OpenCL abstract syntax tree
  ClAst* = ref object
    case kind*: ClNodeKind
    of clnVoid: discard
    of clnProc:
      pName*: string
      pRetType*: ClType
      pParams*: seq[tuple[name: string, typ: ClType, attrs: seq[ClVarAttribute]]]
      pBody*: ClAst
      pAttributes*: set[ClAttribute]
    of clnCall:
      cName*: string
      cArgs*: seq[ClAst]
    of clnIf:
      ifCond*: ClAst
      ifThen*: ClAst
      ifElse*: Option[ClAst]
    of clnFor:
      fVar*: string
      fStart*: ClAst
      fEnd*: ClAst
      fBody*: ClAst
    of clnWhile:
      wCond*: ClAst
      wBody*: ClAst
    of clnBinOp:
      bOp*: string
      bLeft*: ClAst
      bRight*: ClAst
    of clnPrefix:
      pOp*: string
      pVal*: ClAst
    of clnVar:
      vName*: string
      vType*: ClType
      vInit*: ClAst
      vRequiresMemcpy*: bool
      vAttributes*: seq[ClVarAttribute]
    of clnAssign:
      aLeft*: ClAst
      aRight*: ClAst
      aRequiresMemcpy*: bool
    of clnIdent:
      iName*: string
    of clnLit:
      lValue*: string
      lType*: ClType
    of clnArrayLit:
      aValues*: seq[string]
      aLitType*: ClType
    of clnConstexpr:
      cIdent*: ClAst
      cValue*: ClAst
      cType*: ClType
    of clnBlock:
      blockLabel*: string
      statements*: seq[ClAst]
    of clnReturn:
      rValue*: ClAst
    of clnDot:
      dParent*: ClAst
      dField*: ClAst
    of clnIndex:
      iArr*: ClAst
      iIndex*: ClAst
    of clnTypeDef:
      tName*: string
      tFields*: seq[ClTypeField]
    of clnObjConstr:
      ocName*: string
      ocFields*: seq[tuple[name: string, value: ClAst]]
    of clnCast:
      cTo*: ClType
      cExpr*: ClAst
    of clnAddr:
      aOf*: ClAst
    of clnDeref:
      dOf*: ClAst
    of clnComment:
      comment*: string
    of clnVecConstr:
      vcType*: ClType
      vcArgs*: seq[ClAst]

  ClContext* = object
    skipSemicolon*: bool

# Re-export DSL pragmas and stub functions from clbase
import clbase
export kernel, device, global, local, constant, oclName
export Dim, get_global_id, get_local_id, get_group_id
export get_global_size, get_local_size, get_num_groups
export get_work_dim, get_global_offset
export barrier, mem_fence, CLK_LOCAL_MEM_FENCE, CLK_GLOBAL_MEM_FENCE

#[ Nim -> ClIr type conversion ]#

proc newClType*(kind: ClTypeKind): ClType =
  ## Creates a new ClType of the specified kind.
  ## Raises an assertion error if the kind requires additional information.
  if kind in [cltObject, cltPtr, cltArray]:
    raiseAssert "Objects, pointers, and arrays must use specific constructors"
  return ClType(kind: kind)

proc newClPtrType*(to: ClType): ClType = 
  ## Creates a new ClType representing a pointer to the specified type
  return ClType(kind: cltPtr, to: to)

proc newClVoidPtrType*(): ClType = 
  ## Creates a new ClType representing a void pointer.
  return ClType(kind: cltVoidPtr)

proc newClObjectType*(name: string, fields: seq[ClTypeField]): ClType =
  ## Creates a new ClType representing an object (struct) with specified name and fields
  return ClType(kind: cltObject, name: name, oFields: fields)

proc toClType*(n: NimNode): ClType

proc newClArrayType*(aTyp: ClType, aLen: int): ClType =
  ## Creates a new ClType representing a static array of the specified type and length
  ClType(kind: cltArray, aTyp: aTyp, aLen: aLen)

proc toClTypeKind*(t: NimTypeKind): ClTypeKind =
  ## Converts a Nim type kind to the corresponding ClTypeKind
  case t
  of ntyBool: cltBool
  of ntyInt8: cltInt8
  of ntyInt16: cltInt16
  of ntyInt32: cltInt32
  of ntyInt64: cltInt64
  of ntyUint8: cltUint8
  of ntyUint16: cltUint16
  of ntyUint32: cltUint32
  of ntyUint64: cltUint64
  of ntyFloat32: cltFloat32
  of ntyFloat64: cltFloat64
  of ntyInt: cltInt32  # Default int maps to int32
  of ntyUInt: cltUint32  # Default uint maps to uint32
  of ntyFloat: cltFloat64  # Default float maps to float64
  else: raiseAssert "Unsupported Nim type: " & $t

proc parseTypeFields*(n: NimNode): seq[ClTypeField] =
  ## Converts a Nim object type node into a sequence of ClTypeField
  ## Needed to generate a c struct
  doAssert n.kind == nnkObjectTy
  doAssert n[2].kind == nnkRecList
  for ch in n[2]:
    doAssert ch.kind == nnkIdentDefs and ch.len == 3
    result.add ClTypeField(name: ch[0].strVal, typ: toClType(ch[1]))

proc getInnerPointerType*(n: NimNode): ClType =
  ## Extracts the type that a pointer is pointing to from a NimNode
  ## For `ptr UncheckedArray[T]`, returns ClType for T (not UncheckedArray[T])
  ## since in OpenCL C, `ptr UncheckedArray[T]` is just `T*`
  if n.typeKind in {ntyPointer, ntyUncheckedArray}:
    let typ = n.getTypeInst()
    doAssert typ.kind == nnkBracketExpr
    doAssert typ[0].strVal in ["ptr", "UncheckedArray"]
    return toClType(typ[1])
  elif n.typeKind == ntyPtr:
    let typ = n.getTypeInst()
    # Handle both nnkBracketExpr (ptr[T]) and nnkPtrTy (ptr T) representations
    if typ.kind == nnkBracketExpr:
      doAssert typ[0].strVal == "ptr"
      let inner = typ[1]
      # For ptr UncheckedArray[T], return T directly (avoids double pointer)
      if inner.kind == nnkBracketExpr and inner[0].strVal == "UncheckedArray":
        return toClType(inner[1])
      else:
        return toClType(inner)
    elif typ.kind == nnkPtrTy:
      let inner = typ[0]
      # For ptr UncheckedArray[T], return T directly (avoids double pointer)
      if inner.kind == nnkBracketExpr and inner[0].strVal == "UncheckedArray":
        return toClType(inner[1])
      else:
        return toClType(inner)
    else:
      raiseAssert "Unexpected ptr type representation: " & $typ.treeRepr
  elif n.kind == nnkPtrTy or n.kind == nnkVarTy: return toClType(n[0])
  else: raiseAssert "Unexpected node: " & $n.treerepr

proc determineArrayLength*(n: NimNode): int =
  ## Determines the length of a static array from a NimNode
  case n[1].kind
  of nnkSym: return n[1].getImpl.intVal.int
  of nnkIntLit: return n[1].intVal.int
  of nnkInfix: 
    doAssert n[1][1].intVal == 0
    return n[1][2].intVal.int + 1
  else: raiseAssert "Cannot determine array length: " & $n.treerepr

proc getTypeName*(n: NimNode): string =
  ## Extracts the type name from a NimNode
  case n.kind
  of nnkIdent, nnkSym: return n.strVal
  of nnkObjConstr: n.getTypeInst.strVal
  else: raiseAssert "Unexpected node in getTypeName: " & $n.treerepr

proc toClType*(n: NimNode): ClType =
  case n.kind
  of nnkIdentDefs:
    if n[n.len - 2].kind != nnkEmpty: return toClType(n[n.len - 2])
    else: return toClType(n[n.len - 1].getTypeInst())
  of nnkConstDef: 
    if n[1].kind != nnkEmpty: return toClType(n[1])
    else: return toClType(n[2])
  else:
    if n.kind == nnkEmpty: return newClType(cltVoid)

    # TODO: handle vector types

    case n.typeKind
    of ntyBool, ntyInt .. ntyUint64: return newClType(toClTypeKind(n.typeKind))
    of ntyPtr, ntyVar: return newClPtrType(getInnerPointerType(n))
    of ntyPointer: return newClVoidPtrType()
    of ntyUncheckedArray: return newClPtrType(getInnerPointerType(n))
    of ntyObject:
      # TODO: handle vector types

      let impl = n.getTypeImpl()
      let flds = parseTypeFields(impl)
      return newClObjectType(n.strVal(), flds)
    of ntyArray:
      if n.kind == nnkSym: return toClType(n.getTypeImpl())
      if n.len == 3: return newClArrayType(toClType(n[2]), determineArrayLength(n))
      else: return newClArrayType(toClType(n[0]), n.len)
    of ntyGenericInst:
      let impl = n.getTypeImpl()
      case impl.kind
      of nnkDistinctTy: return toClType(impl[0])
      else: raiseAssert "Unsupported generic: " & $n.treerepr
    else: raiseAssert "Unsupported type: " & $n.typeKind & " - " & $n.treerepr
  
#[ Nim AST -> Cl intermediate representation AST ]#

proc assignOp(op: string; isBoolean: bool): string =
  ## Maps Nim operation to OpenCL C assignment operation
  case op
  of "div": return "/"
  of "mod": return "%"
  of "shl": return "<<"
  of "shr": return ">>"
  of "and": (if isBoolean: "&&" else: "&")
  of "or": (if isBoolean: "||" else: "|")
  of "xor": "^"
  else: return op

proc assignPrefixOp(op: string): string =
  ## Maps Nim prefix operation to OpenCL C prefix operation
  case op
  of "not": return "!"
  else: return op

proc getFnName(n: NimNode): string =
  ## Extracts procedure name from NimNode
  if n.kind == nnkSym:
    let impl = n.getImpl()
    if impl.kind in [nnkProcDef, nnkFuncDef]:
      let pragma = impl.pragma
      if pragma.kind != nnkEmpty and pragma[0].kind == nnkExprColonExpr:
        if pragma[0][0].strVal == "oclName": return pragma[0][1].strVal
  let name = n.repr
  if name.endsWith("_impl"): return name[0..<name.len-5]
  return name
  # TODO: handle vector types

proc collectProcAttributes(n: NimNode): set[ClAttribute] =
  ## Collects OpenCL procedure attributes from NimNode pragmas
  doAssert n.kind == nnkPragma
  for pragma in n:
    doAssert pragma.kind in [nnkIdent, nnkSym]
    case pragma.strVal
    of "kernel": result.incl attKernel
    of "inline": result.incl attInline
    of "device": discard
  
proc collectVarAttributes(n: NimNode): seq[ClVarAttribute] =
  ## Collects OpenCL variable attributes from NimNode pragmas
  doAssert n.kind == nnkPragma
  for pragma in n:
    case pragma.strVal
    of "global": result.add atvGlobal
    of "local": result.add atvLocal
    of "constant": result.add atvConstant
    of "private": result.add atvPrivate

proc ensureBlock(ast: ClAst): ClAst =
  ## Ensures that the given ClAst is a block; if not, wraps it in a block
  if ast.kind == clnBlock: return ast
  else: return ClAst(kind: clnBlock, statements: @[ast])

proc requiresMemcpy(n: NimNode): bool =
  return n.typeKind == ntyArray and n.kind != nnkBracket

proc toClAst*(ctx: var ClContext; node: NimNode): ClAst =
  case node.kind
  of nnkEmpty: return ClAst(kind: clnVoid)
  of nnkStmtList:
    result = ClAst(kind: clnBlock)
    for el in node: result.statements.add ctx.toClAst(el)
  of nnkBlockStmt:
    let blockLabel = (
      if node[0].kind in {nnkSym, nnkIdent}: node[0].strVal
      elif node[0].kind == nnkEmpty: ""
      else: ""
    )
    result = ClAst(kind: clnBlock, blockLabel: blockLabel)
    for idx in 1..<node.len: result.statements.add ctx.toClAst(node[idx])
  of nnkStmtListExpr:
    result = ClAst(kind: clnBlock)
    for el in node:
      if el.kind != nnkEmpty: result.statements.add ctx.toClAst(el)
  of nnkDiscardStmt: return ctx.toClAst(node[0])
  of nnkProcDef, nnkFuncDef:
    result = ClAst(kind: clnProc)
    result.pName = node.name.strVal
    result.pRetType = toClType(node[3][0])

    # process parameters
    for idx in 1..<node[3].len:
      let param = node[3][idx]
      let numParams = param.len - 2
      let typIdx = param.len - 2
      let paramType = toClType(param[typIdx])
      for jdx in 0..<numParams:
        var attrs: seq[ClVarAttribute]

        # check for pointer parameters in kernels - default to __global
        if paramType.kind == cltPtr: attrs.add atvGlobal
        result.pParams.add (param[jdx].strVal, paramType, attrs)

    # process pragmas
    if node.pragma.kind != nnkEmpty:
      result.pAttributes = collectProcAttributes(node.pragma)

    result.pBody = ctx.toClAst(node.body).ensureBlock()
  of nnkLetSection, nnkVarSection:
    result = ClAst(kind: clnBlock)
    for decl in node:
      var varNode = ClAst(kind: clnVar)
      case decl[0].kind
      of nnkIdent, nnkSym:
        varNode.vName = decl[0].strVal
      of nnkPragmaExpr:
        varNode.vName = decl[0][0].strVal
        if decl[0][1].kind == nnkPragma:
          varNode.vAttributes = collectVarAttributes(decl[0][1])
      else: raiseAssert "Unexpected variable node: " & $decl.treeRepr

      varNode.vType = toClType(decl[1])
      if decl.len > 2 and decl[2].kind != nnkEmpty:
        varNode.vInit = ctx.toClAst(decl[2])
        varNode.vRequiresMemcpy = requiresMemcpy(decl[2])
      result.statements.add varNode
  of nnkAsgn:
    result = ClAst(kind: clnAssign)
    result.aLeft = ctx.toClAst(node[0])
    result.aRight = ctx.toClAst(node[1])
    result.aRequiresMemcpy = requiresMemcpy(node[1])
  of nnkIfStmt:
    result = ClAst(kind: clnIf)
    let branch = node[0]
    result.ifCond = ctx.toClAst(branch[0])
    result.ifThen = ensureBlock(ctx.toClAst(branch[1]))
    if node.len > 1 and node[^1].kind == nnkElse:
      result.ifElse = some(ensureBlock(ctx.toClAst(node[^1][0])))
  of nnkForStmt:
    result = ClAst(kind: clnFor)
    result.fVar = node[0].strVal
    result.fStart = ctx.toClAst(node[1][1])
    result.fEnd = ctx.toClAst(node[1][2])
    result.fBody = ensureBlock(ctx.toClAst(node[2]))
  of nnkWhileStmt:
    result = ClAst(kind: clnWhile)
    result.wCond = ctx.toClAst(node[0])
    result.wBody = ensureBlock(ctx.toClAst(node[1]))
  of nnkCall, nnkCommand:
    let name = getFnName(node[0])
    let args = node[1..^1].mapIt(ctx.toClAst(it))
    result = ClAst(kind: clnCall, cName: name, cArgs: args)
  of nnkInfix:
    result = ClAst(kind: clnBinOp)
    let isBoolean = node[1].typeKind == ntyBool
    result.bOp = assignOp(node[0].repr, isBoolean)
    result.bLeft = ctx.toClAst(node[1])
    result.bRight = ctx.toClAst(node[2])
  of nnkDotExpr:
    result = ClAst(kind: clnDot)
    result.dParent = ctx.toClAst(node[0])
    result.dField = ctx.toClAst(node[1])
  of nnkIdent, nnkSym, nnkOpenSymChoice: return ClAst(kind: clnIdent, iName: node.repr)
  of nnkUInt32Lit:
    return ClAst(kind: clnLit, lValue: $node.intVal, lType: newClType(cltUint32))
  of nnkIntLit, nnkInt32Lit:
    return ClAst(kind: clnLit, lValue: $node.intVal, lType: newClType(cltInt32))
  of nnkInt64Lit:
    return ClAst(kind: clnLit, lValue: $node.intVal, lType: newClType(cltInt64))
  of nnkFloat32Lit:
    return ClAst(kind: clnLit, lValue: $node.floatVal & "f", lType: newClType(cltFloat32))
  of nnkFloatLit, nnkFloat64Lit:
    return ClAst(kind: clnLit, lValue: $node.floatVal, lType: newClType(cltFloat64))
  of nnkStrLit, nnkRStrLit:
    return ClAst(
      kind: clnLit, 
      lValue: node.strVal.escape("", ""), 
      lType: newClType(cltString)
    )
  of nnkNilLit: return ClAst(kind: clnLit, lValue: "NULL", lType: newClVoidPtrType())
  of nnkPar:
    if node.len == 1: return ctx.toClAst(node[0])
    else: error "`nnkPar` with more than one argument not supported: " & $node.treerepr
  of nnkReturnStmt:
    if node[0].kind == nnkAsgn and node[0][0].strVal == "result":
      return ClAst(kind: clnReturn, rValue: ctx.toClAst(node[0][1]))
    else: return ClAst(kind: clnReturn, rValue: ctx.toClAst(node[0]))
  of nnkBracketExpr:
    result = ClAst(kind: clnIndex)
    result.iArr = ctx.toClAst(node[0])
    result.iIndex = ctx.toClAst(node[1])
  of nnkPrefix:
    result = ClAst(kind: clnPrefix, pVal: ctx.toClAst(node[1]))
    result.pOp = assignPrefixOp(node[0].strVal)
  of nnkTypeSection:
    result = ClAst(kind: clnBlock)
    for el in node:
      doAssert el.kind == nnkTypeDef
      result.statements.add ctx.toClAst(el)
  of nnkTypeDef:
    result = ClAst(kind: clnTypeDef, tName: node[0].strVal)
    result.tFields = parseTypeFields(node[2])
  of nnkObjConstr:
    let typName = getTypeName(node)
    result = ClAst(kind: clnObjConstr, ocName: typName)
    for i in 1 ..< node.len:
      doAssert node[i].kind == nnkExprColonExpr
      result.ocFields.add (name: node[i][0].strVal, value: ctx.toClAst(node[i][1]))
  of nnkBracket:
    let aLitTyp = toClType(node[0])
    var aValues: seq[string]
    for el in node:
      aValues.add $el.intVal
    result = ClAst(kind: clnArrayLit, aValues: aValues, aLitType: aLitTyp)
  of nnkCommentStmt:
    result = ClAst(kind: clnComment, comment: node.strVal)
  of nnkHiddenStdConv:
    doAssert node[0].kind == nnkEmpty
    return ctx.toClAst(node[1])
  of nnkCast, nnkConv:
    result = ClAst(kind: clnCast, cTo: toClType(node[0]), cExpr: ctx.toClAst(node[1]))
  of nnkAddr, nnkHiddenAddr:
    result = ClAst(kind: clnAddr, aOf: ctx.toClAst(node[0]))
  of nnkHiddenDeref:
    case node.typeKind
    of ntyUncheckedArray: return ctx.toClAst(node[0])
    else: result = ClAst(kind: clnDeref, dOf: ctx.toClAst(node[0]))
  of nnkDerefExpr:
    result = ClAst(kind: clnDeref, dOf: ctx.toClAst(node[0]))
  of nnkConstDef:
    result = ClAst(
      kind: clnConstexpr,
      cIdent: ctx.toClAst(node[0]),
      cValue: ctx.toClAst(node[2]),
      cType: toClType(node)
    )
  of nnkConstSection:
    result = ClAst(kind: clnBlock)
    for el in node:
      doAssert el.kind == nnkConstDef
      result.statements.add ctx.toClAst(el)
  of nnkTemplateDef:
    # Templates should be expanded by the time we see them (typed macro)
    result = ClAst(kind: clnVoid)
  else:
    raiseAssert "Unhandled node kind: " & $node.kind & "\n" & node.treerepr
