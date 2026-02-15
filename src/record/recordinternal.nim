#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/record/recordinternal.nim
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

## Core types and compilation pipeline for the `record` macro.
##
## Records are value-type objects inspired by Chapel records:
##   - Value semantics (plain `object`, not `ref object`)
##   - No inheritance
##   - Compiler-generated copy init, assignment, comparison, and hash
##   - Optional user-defined `init`, `deinit`, `init=`, `==`
##   - All methods resolved at compile time (procs, not methods)
##
## Architecture adapted from jjv360/nim-classes (MIT License).

import std/[macros]
import std/[tables]
import std/[strutils]

## Declare a record method as static (no `this` parameter).
template static* {.pragma.}

## Declare a record method as immutable (``this`` is not ``var``).
## Immutable methods receive ``this: RecordName`` instead of
## ``this: var RecordName``, allowing them to be called on both
## mutable and immutable values.
template immutable* {.pragma.}

type RecordDescription* = ref object of RootRef
  macroName*: string
  name*: NimNode
  genericParams*: NimNode  ## nnkGenericParams or nnkEmpty if non-generic
  inputBody*: NimNode
  comment*: NimNode
  metadata*: Table[string, RootRef]
  prefix*: NimNode
  output*: NimNode
  fwdDecl*: NimNode
  body*: NimNode
  suffix*: NimNode

type RecordCompilerStage* = enum
  RecordPreload
  RecordGatherDefinitions
  RecordAddExtraDefinitions
  RecordModifyDefinitions
  RecordGenerateCode
  RecordFinalize
  RecordDebugEcho

#[ Hook system ]#

type RecordCompilerHook* =
  proc(stage: RecordCompilerStage, recDef: RecordDescription)

## All registered hooks (populated at compile time).
var recordCompilerHooks* {.compileTime.}: seq[RecordCompilerHook] = @[]

## Cache of already-compiled record definitions.
var recordCompilerCache* {.compileTime.}: seq[RecordDescription] = @[]

#[ Helpers ]#

proc recordDefinitionFor*(name: NimNode): RecordDescription =
  ## Look up an already-compiled record by name, or nil.
  if name.kind != nnkIdent and name.kind != nnkSym:
    error("Expected an identifier.", name)
  for recDef in recordCompilerCache:
    if $recDef.name == $name:
      return recDef
  return nil

#[ Generic helpers ]#

proc isGeneric*(recDef: RecordDescription): bool =
  ## True when the record has generic type parameters.
  recDef.genericParams.kind != nnkEmpty

proc fullType*(recDef: RecordDescription): NimNode =
  ## Returns the fully-qualified type node.
  ##   Non-generic:  Name
  ##   Generic:      Name[T, U, ...]
  if not recDef.isGeneric:
    return recDef.name
  # Build nnkBracketExpr(Name, T, U, ...)
  let bracket = newNimNode(nnkBracketExpr)
  bracket.add(recDef.name)
  for identDef in recDef.genericParams:
    # Each identDef may list multiple names before the type+default.
    # We only take the name idents (all but the last 2 children).
    for i in 0 ..< identDef.len - 2:
      bracket.add(copyNimTree(identDef[i]))
  return bracket

proc addGenericParams*(procNode: NimNode, recDef: RecordDescription) =
  ## Inject the record's generic parameters into a proc/func node.
  ## procNode[2] is the GenericParams slot in a RoutineNode.
  if not recDef.isGeneric:
    return
  let gp = copyNimTree(recDef.genericParams)
  if procNode[2].kind == nnkEmpty:
    procNode[2] = gp
  else:
    # Merge: append each identDef
    for identDef in gp:
      procNode[2].add(identDef)

#[ Construction ]#

proc newRecordDescription*(macroName: string, head: NimNode,
                           body: NimNode): RecordDescription =
  ## Parse the macro head and body into a `RecordDescription`.
  let recDef = RecordDescription()
  recDef.macroName = macroName
  recDef.inputBody = body
  recDef.prefix = newNimNode(nnkStmtList)
  recDef.fwdDecl = newNimNode(nnkStmtList)
  recDef.body = newNimNode(nnkStmtList)
  recDef.suffix = newNimNode(nnkStmtList)
  recDef.genericParams = newEmptyNode()

  if head.kind == nnkIdent:
    # record MyRecord:
    recDef.name = head

  elif head.kind == nnkPostfix and $head[0] == "*":
    # record MyRecord*:  (exported, no generics)
    # The `*` export marker is accepted but redundant — records auto-export.
    recDef.name = head[1]

  elif head.kind == nnkBracketExpr:
    # record MyRecord[T]:
    # record MyRecord[T, U]:
    # record MyRecord[T: SomeInteger]:
    recDef.name = head[0]
    let gp = newNimNode(nnkGenericParams)
    for i in 1 ..< head.len:
      let param = head[i]
      if param.kind == nnkExprColonExpr:
        # T: Constraint
        gp.add(newIdentDefs(param[0], param[1], newEmptyNode()))
      else:
        # Plain T
        gp.add(newIdentDefs(param, newEmptyNode(), newEmptyNode()))
    recDef.genericParams = gp

  elif head.kind == nnkInfix and $head[0] == "*":
    # record MyRecord*[T]:       → nnkInfix("*", ident"MyRecord", nnkBracket)
    # The `*` export marker is accepted but redundant — records auto-export.
    let rhs = head[2]
    if rhs.kind == nnkEmpty or (rhs.kind == nnkIdent and $rhs == ""):
      # record MyRecord*:  (no generics, alternate parse)
      recDef.name = head[1]
    elif rhs.kind in {nnkBracket, nnkBracketExpr}:
      # record MyRecord*[T, U]:
      # nnkBracket when parsed as infix, nnkBracketExpr otherwise
      recDef.name = head[1]
      let gp = newNimNode(nnkGenericParams)
      let startIdx = if rhs.kind == nnkBracketExpr: 1 else: 0
      for i in startIdx ..< rhs.len:
        let param = rhs[i]
        if param.kind == nnkExprColonExpr:
          gp.add(newIdentDefs(param[0], param[1], newEmptyNode()))
        else:
          gp.add(newIdentDefs(param, newEmptyNode(), newEmptyNode()))
      recDef.genericParams = gp
    else:
      error("Invalid record syntax after '*'.", head)

  elif head.kind == nnkInfix:
    # Disallow inheritance – records are value types
    error("Records do not support inheritance. Use a plain name.", head)

  else:
    error("Invalid record syntax.", head)

  # Run hooks through all stages
  for hook in recordCompilerHooks: hook(RecordPreload, recDef)
  for hook in recordCompilerHooks: hook(RecordGatherDefinitions, recDef)
  for hook in recordCompilerHooks: hook(RecordAddExtraDefinitions, recDef)
  for hook in recordCompilerHooks: hook(RecordModifyDefinitions, recDef)

  # Create the object type definition:
  #   Non-generic:  type Name* = object
  #   Generic:      type Name*[T, U] = object
  let className = recDef.name
  recDef.output = quote do:
    type `className`* = object
  # Inject generic params into the TypeDef node (slot [1])
  if recDef.isGeneric:
    recDef.output[0][1] = copyNimTree(recDef.genericParams)

  # Run remaining stages
  for hook in recordCompilerHooks: hook(RecordGenerateCode, recDef)
  for hook in recordCompilerHooks: hook(RecordFinalize, recDef)

  return recDef

#[ Compile – assemble the final NimNode tree ]#

proc compile*(this: RecordDescription): NimNode =
  ## Assemble all generated code and return it as a single nnkStmtList.
  # Cache this record
  recordCompilerCache.add(this)

  var code = newStmtList()
  code.add(this.prefix)
  code.add(this.output)
  code.add(this.fwdDecl)
  code.add(this.body)
  code.add(this.suffix)

  # Debug output
  const debugrecords {.strdefine.} = ""
  let debugNames = debugrecords.split(",")
  if debugrecords == "true" or debugNames.contains($this.name):
    echo "\n=== Created " & this.macroName & " " & $this.name
    for hook in recordCompilerHooks:
      hook(RecordDebugEcho, this)
    echo ""
    echo code.repr
    echo ""

  return code
