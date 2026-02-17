#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/record/recordmethods.nim
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

## Gather `proc` (declared as `method` in the DSL) definitions
## from the record body and generate the corresponding Nim procs.
##
## Unlike nim-classes which uses dynamic dispatch (`method`), records use
## compile-time-resolved `proc` — matching Chapel's static resolution.
##
## Adapted from jjv360/nim-classes
## 
## MIT License
##  
## Copyright (c) 2021 jjv360
##  
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
##  
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
## WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
## CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

{.used.}

import std/[macros]
import std/[tables]

import recordinternal
import recordutils

#[ Types ]#

type RecordMethod* = ref object of RootRef
  definition*: NimNode
  comment*: NimNode
  body*: NimNode
  outputCode*: NimNode
  isStatic*: bool
  isConst*: bool
  insertUnmodified*: bool

proc clone*(this: RecordMethod): RecordMethod =
  let copy = RecordMethod()
  copy.definition = copyNimTree(this.definition)
  copy.body = copyNimTree(this.body)
  copy.comment = copyNimTree(this.comment)
  copy.insertUnmodified = this.insertUnmodified
  copy.isStatic = this.isStatic
  copy.isConst = this.isConst
  return copy

type RecordMethods* = ref object of RootRef
  definitions*: seq[RecordMethod]

proc methods*(this: RecordDescription): RecordMethods =
  if not this.metadata.contains("methods"):
    this.metadata["methods"] = RecordMethods()
  return (RecordMethods)this.metadata["methods"]

#[ Hash for method identity ]#

proc hash*(this: RecordMethod): string =
  var h = $this.definition.name & "("
  var first = true
  for i, param in this.definition.params:
    if i == 0: continue  # skip return type
    if not first: h &= ","
    first = false
    h &= param[1].repr
  h &= ")"
  if this.definition.params[0].kind != nnkEmpty:
    h &= ":" & this.definition.params[0].repr
  return h

#[ Gather ]#

proc gatherDefinitions(recDef: RecordDescription) =
  var previousComment: NimNode = nil
  traverseRecordStatementList recDef.inputBody, proc(idx: int, parent: NimNode, node: NimNode) =
    if node.kind == nnkMethodDef:
      var meth = RecordMethod()
      meth.definition = copyNimTree(node)
      meth.definition.body = newEmptyNode()
      meth.comment = previousComment
      previousComment = nil
      meth.body = copyNimTree(node.body)
      recDef.methods.definitions.add(meth)

      # Check static and immutable pragmas
      meth.isStatic = false
      meth.isConst = false
      for i, p in meth.definition.pragma:
        if p.kind == nnkIdent and $p == "static":
          meth.isStatic = true
        elif p.kind == nnkIdent and $p == "immutable":
          meth.isConst = true

      # Abstract methods (no body) → raise error if called
      if meth.body.kind == nnkEmpty:
        let mName = $meth.definition.name
        let text = $recDef.name & "." & mName & "() is not implemented."
        meth.body = newStmtList(
          quote do: raiseAssert(`text`)
        )

    elif node.kind in RoutineNodes:
      error("Only 'method' declarations are allowed inside a record body.", node)

    elif node.kind == nnkCommentStmt:
      previousComment = node

    else:
      previousComment = nil

#[ Code generation – produce `proc` (not `method`) ]#

const dslPragmas = ["static", "immutable"]  ## pragmas consumed by the record DSL

proc stripDslPragmas(procNode: NimNode) =
  ## Remove record-DSL pragmas (static, immutable, …) from the generated
  ## proc so the Nim compiler does not reject them as unknown.
  let pragmaNode = procNode.pragma
  if pragmaNode.kind == nnkEmpty: return
  var i = 0
  while i < pragmaNode.len:
    let p = pragmaNode[i]
    if p.kind == nnkIdent and $p in dslPragmas:
      pragmaNode.del(i)
    else:
      inc i
  # If all pragmas were removed, replace with nnkEmpty
  if pragmaNode.len == 0:
    procNode.pragma = newEmptyNode()

proc generateCode(recDef: RecordDescription) =
  for methodDef in recDef.methods.definitions:
    # Build the proc signature
    var procCopy = copyNimTree(methodDef.definition)

    # Convert nnkMethodDef → nnkProcDef
    let procNode = newNimNode(nnkProcDef)
    copyChildrenTo(procCopy, procNode)
    procCopy = procNode

    # Strip DSL-specific pragmas before emitting the proc
    stripDslPragmas(procCopy)

    if not methodDef.insertUnmodified and methodDef.isStatic:
      # Static: inject typedesc as first param.
      # Use fullType so generic params are inferrable at the call site.
      let classTypedesc = newNimNode(nnkBracketExpr)
      classTypedesc.add(ident"typedesc")
      classTypedesc.add(recDef.fullType)
      procCopy.params.insert(1, newIdentDefs(ident"_", classTypedesc))

    elif not methodDef.insertUnmodified and methodDef.isConst:
      # Immutable instance method: inject `this: RecordName[T,...]` (no var)
      # Allows calling on both mutable and immutable values.
      procCopy.params.insert(1, newIdentDefs(ident"this", recDef.fullType))

    elif not methodDef.insertUnmodified:
      # Instance method: inject `this: var RecordName[T,...]` as first param
      # Use `var` so methods can mutate fields (Chapel: ref this-intent)
      procCopy.params.insert(1, newIdentDefs(ident"this",
        newNimNode(nnkVarTy).add(recDef.fullType)))

    # Inject generic params into the proc signature
    if not methodDef.insertUnmodified:
      procCopy.addGenericParams(recDef)

    # Forward declaration – skip for generic records because Nim cannot
    # match a body-less forward decl to its implementation when generic
    # params are involved, resulting in duplicate overloads.
    # Also skip for insertUnmodified procs (convenience constructors).
    if not methodDef.insertUnmodified and not recDef.isGeneric:
      recDef.fwdDecl.add(procCopy)

    # Full body
    var procWithBody = copyNimTree(procCopy)
    procWithBody.body = copyNimTree(methodDef.body)
    recDef.body.add(procWithBody)
    methodDef.outputCode = procWithBody

#[ Debug ]#

proc debugEcho(recDef: RecordDescription) =
  for methodDef in recDef.methods.definitions:
    echo "- Proc: " & $methodDef.definition.name

#[ Register ]#

static:
  recordCompilerHooks.add(proc(stage: RecordCompilerStage, recDef: RecordDescription) =
    if stage == RecordGatherDefinitions: gatherDefinitions(recDef)
    if stage == RecordGenerateCode: generateCode(recDef)
    if stage == RecordDebugEcho: debugEcho(recDef)
  )
