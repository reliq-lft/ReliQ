#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/record/recordvar.nim
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

## Gather `var` field definitions from the record body and
## generate the corresponding object fields.
##
## Adapted from jjv360/nim-classes (MIT License).

{.used.}

import std/[macros]
import std/[tables]

import recordinternal
import recordutils

#[ Types ]#

type Variable* = ref object of RootRef
  definition*: NimNode
  comment*: NimNode

proc clone*(this: Variable): Variable =
  let copy = Variable()
  copy.definition = copyNimTree(this.definition)
  copy.comment = copyNimTree(this.comment)
  return copy

type Variables* = ref object of RootRef
  definitions*: seq[Variable]
  whenBlocks*: seq[NimNode]

proc vars*(this: RecordDescription): Variables =
  if not this.metadata.contains("vars"):
    this.metadata["vars"] = Variables()
  return (Variables)this.metadata["vars"]

#[ Type auto-detection from literal values ]#

proc autodetectVarType(identDef: NimNode) =
  let nameDef = identDef.variableName
  var typeNode: NimNode = identDef[1]
  if typeNode.kind != nnkEmpty:
    return

  let valueNode = identDef[2]
  if valueNode.kind == nnkEmpty:
    error("The record field '" & $nameDef & "' must have a type or default value.", identDef)

  if valueNode.kind == nnkCharLit: typeNode = ident"char"
  elif valueNode.kind == nnkIntLit: typeNode = ident"int"
  elif valueNode.kind == nnkInt8Lit: typeNode = ident"int8"
  elif valueNode.kind == nnkInt16Lit: typeNode = ident"int16"
  elif valueNode.kind == nnkInt32Lit: typeNode = ident"int32"
  elif valueNode.kind == nnkInt64Lit: typeNode = ident"int64"
  elif valueNode.kind == nnkUIntLit: typeNode = ident"uint"
  elif valueNode.kind == nnkUInt8Lit: typeNode = ident"uint8"
  elif valueNode.kind == nnkUInt16Lit: typeNode = ident"uint16"
  elif valueNode.kind == nnkUInt32Lit: typeNode = ident"uint32"
  elif valueNode.kind == nnkUInt64Lit: typeNode = ident"uint64"
  elif valueNode.kind == nnkFloatLit: typeNode = ident"float"
  elif valueNode.kind == nnkFloat32Lit: typeNode = ident"float32"
  elif valueNode.kind == nnkFloat64Lit: typeNode = ident"float64"
  elif valueNode.kind == nnkFloat128Lit: typeNode = ident"float128"
  elif valueNode.kind == nnkStrLit: typeNode = ident"string"
  elif valueNode.kind == nnkSym and $valueNode == "false": typeNode = ident"bool"
  elif valueNode.kind == nnkSym and $valueNode == "true": typeNode = ident"bool"
  elif valueNode.kind == nnkIdent and $valueNode == "false": typeNode = ident"bool"
  elif valueNode.kind == nnkIdent and $valueNode == "true": typeNode = ident"bool"
  else:
    error("The record field '" & $nameDef & "' must have an explicit type.", identDef)

  identDef[1] = typeNode

#[ Gather ]#

proc gatherDefinitions(recDef: RecordDescription) =
  var previousComment: NimNode = nil
  traverseRecordStatementList recDef.inputBody, proc(idx: int, parent: NimNode, node: NimNode) =
    if node.kind == nnkLetSection or node.kind == nnkConstSection:
      error("Record fields must be defined with 'var'.", node)
    elif node.kind == nnkVarSection:
      for identDef in node:
        let copyIdent = identDef.copyNimTree()
        autodetectVarType(copyIdent)
        recDef.vars.definitions.add(Variable(definition: copyIdent, comment: previousComment))
        previousComment = nil
    elif node.kind == nnkWhenStmt:
      recDef.vars.whenBlocks.add(copyNimTree(node))
      previousComment = nil
    elif node.kind == nnkCommentStmt:
      previousComment = node
    else:
      previousComment = nil

#[ Transform when-block branches for object recList ]#

proc toRecWhen(node: NimNode): NimNode =
  ## Convert an nnkWhenStmt (from the macro body) into an nnkRecWhen
  ## suitable for an object type's nnkRecList. Recursively transforms
  ## branch bodies: strips nnkVarSection wrappers to bare nnkIdentDefs,
  ## and converts nested nnkWhenStmt nodes to nnkRecWhen.
  result = newNimNode(nnkRecWhen)
  for i in 0 ..< node.len:
    let branch = node[i]
    let newBranch = copyNimTree(branch)
    # Process the body (last child of the branch)
    let body = newBranch[newBranch.len - 1]

    var recList = newNimNode(nnkRecList)
    if body.kind == nnkStmtList:
      for stmt in body:
        if stmt.kind == nnkVarSection:
          for identDef in stmt:
            let fieldDef = copyNimTree(identDef)
            fieldDef[fieldDef.len - 1] = newEmptyNode()  # strip default value
            recList.add(fieldDef)
        elif stmt.kind == nnkWhenStmt:
          recList.add(toRecWhen(stmt))
        else:
          recList.add(copyNimTree(stmt))
    elif body.kind == nnkVarSection:
      # Single-line branch: `when cond: var x: T`
      for identDef in body:
        let fieldDef = copyNimTree(identDef)
        fieldDef[fieldDef.len - 1] = newEmptyNode()
        recList.add(fieldDef)

    newBranch[newBranch.len - 1] = recList
    result.add(newBranch)

#[ Code generation ]#

proc generateCode(recDef: RecordDescription) =
  # The output from `quote do: type Name* = object` is:
  #   nnkTypeSection
  #     nnkTypeDef
  #       [0] nnkPostfix(*, Name)
  #       [1] nnkEmpty              (generic params)
  #       [2] nnkObjectTy
  #             [0] nnkEmpty        (pragmas)
  #             [1] nnkEmpty        (of clause)
  #             [2] nnkEmpty        (recList – empty by default)
  # We replace [2] of the ObjectTy with a real nnkRecList.
  let recList = newNimNode(nnkRecList)
  recDef.output[0][2][2] = recList

  for variable in recDef.vars.definitions:
    # Add field without preset value (the object def doesn't allow defaults)
    let varCopy = copyNimTree(variable.definition)
    varCopy[2] = newEmptyNode()  # strip default value
    recList.add(varCopy)

  # Add conditional (when) field blocks
  for whenNode in recDef.vars.whenBlocks:
    recList.add(toRecWhen(whenNode))

#[ Debug ]#

proc debugEcho(recDef: RecordDescription) =
  for variable in recDef.vars.definitions:
    let typeDef = variable.definition[1]
    echo "- Field: " & $variable.definition.variableName & " : " & typeDef.repr

#[ Register ]#

static:
  recordCompilerHooks.add(proc(stage: RecordCompilerStage, recDef: RecordDescription) =
    if stage == RecordGatherDefinitions: gatherDefinitions(recDef)
    if stage == RecordGenerateCode: generateCode(recDef)
    if stage == RecordDebugEcho: debugEcho(recDef)
  )
