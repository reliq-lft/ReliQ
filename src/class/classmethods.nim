#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/classes/classmethods.nim
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

## Gather ``method`` definitions from the class body and generate the
## corresponding Nim methods (dynamic dispatch) or procs (static dispatch).
##
## Chapel class method semantics:
##   - Class methods support dynamic dispatch (unlike record procs).
##   - The type of ``this`` is ``borrowed ClassName`` (non-nilable).
##   - Methods marked ``{.override.}`` override a parent class method.
##   - Static methods (``{.static.}``) and ``init``/``deinit`` use ``proc``.
##   - Other instance methods use Nim ``method`` for dynamic dispatch.
##
## Architecture mirrors ``recordmethods.nim`` but emits ``method`` nodes
## for eligible instance methods and ``proc`` for static/init/deinit.

{.used.}

import std/[macros]
import std/[tables]

import classinternal
import classutils

#[ Types ]#

type ClassMethod* = ref object of RootRef
  definition*: NimNode
  comment*: NimNode
  body*: NimNode
  outputCode*: NimNode
  isStatic*: bool
  isConst*: bool
  isOverride*: bool
  insertUnmodified*: bool

proc clone*(this: ClassMethod): ClassMethod =
  let copy = ClassMethod()
  copy.definition = copyNimTree(this.definition)
  copy.body = copyNimTree(this.body)
  copy.comment = copyNimTree(this.comment)
  copy.insertUnmodified = this.insertUnmodified
  copy.isStatic = this.isStatic
  copy.isConst = this.isConst
  copy.isOverride = this.isOverride
  return copy

type ClassMethods* = ref object of RootRef
  definitions*: seq[ClassMethod]

proc methods*(this: ClassDescription): ClassMethods =
  if not this.metadata.contains("methods"):
    this.metadata["methods"] = ClassMethods()
  return (ClassMethods)this.metadata["methods"]

#[ Hash for method identity ]#

proc hash*(this: ClassMethod): string =
  var h = $this.definition.name & "("
  var first = true
  for i, param in this.definition.params:
    if i == 0: continue
    if not first: h &= ","
    first = false
    h &= param[1].repr
  h &= ")"
  if this.definition.params[0].kind != nnkEmpty:
    h &= ":" & this.definition.params[0].repr
  return h

#[ Gather ]#

proc gatherDefinitions(classDef: ClassDescription) =
  var previousComment: NimNode = nil
  traverseClassStatementList classDef.inputBody, proc(idx: int, parent: NimNode, node: NimNode) =
    if node.kind == nnkMethodDef:
      var meth = ClassMethod()
      meth.definition = copyNimTree(node)
      meth.definition.body = newEmptyNode()
      meth.comment = previousComment
      previousComment = nil
      meth.body = copyNimTree(node.body)
      classDef.methods.definitions.add(meth)

      # Check pragmas
      meth.isStatic = false
      meth.isConst = false
      meth.isOverride = false
      for i, p in meth.definition.pragma:
        if p.kind == nnkIdent and $p == "static":
          meth.isStatic = true
        elif p.kind == nnkIdent and $p == "immutable":
          meth.isConst = true
        elif p.kind == nnkIdent and $p == "override":
          meth.isOverride = true

      # Abstract methods (no body) → raise error if called
      if meth.body.kind == nnkEmpty:
        let mName = $meth.definition.name
        let text = $classDef.name & "." & mName & "() is not implemented."
        meth.body = newStmtList(
          quote do: raiseAssert(`text`)
        )

    elif node.kind in RoutineNodes:
      error("Only 'method' declarations are allowed inside a class body.", node)

    elif node.kind == nnkCommentStmt:
      previousComment = node

    else:
      previousComment = nil

#[ Code generation ]#

## DSL pragmas to strip from generated procs/methods.
const dslPragmas = ["static", "immutable", "override"]

proc stripDslPragmas(procNode: NimNode) =
  let pragmaNode = procNode.pragma
  if pragmaNode.kind == nnkEmpty: return
  var i = 0
  while i < pragmaNode.len:
    let p = pragmaNode[i]
    if p.kind == nnkIdent and $p in dslPragmas:
      pragmaNode.del(i)
    else:
      inc i
  if pragmaNode.len == 0:
    procNode.pragma = newEmptyNode()

proc useMethodDispatch(methodDef: ClassMethod): bool =
  ## Determine if this method should use Nim ``method`` (dynamic dispatch).
  ## Chapel: init, deinit, static methods, type methods, param-returning methods
  ## are NOT candidates for dynamic dispatch.
  let name = $methodDef.definition.name
  if name == "init" or name == "deinit" or name == "postinit":
    return false
  if methodDef.isStatic:
    return false
  if methodDef.insertUnmodified:
    return false
  return true

proc generateCode(classDef: ClassDescription) =
  for methodDef in classDef.methods.definitions:
    var procCopy = copyNimTree(methodDef.definition)

    let useDynamic = methodDef.useMethodDispatch()

    # Convert nnkMethodDef → either nnkMethodDef (dynamic) or nnkProcDef (static)
    if useDynamic:
      # Keep as nnkMethodDef for Nim's dynamic dispatch
      let methodNode = newNimNode(nnkMethodDef)
      copyChildrenTo(procCopy, methodNode)
      procCopy = methodNode
      # Add {.base.} pragma if not overriding
      if not methodDef.isOverride:
        if procCopy.pragma.kind == nnkEmpty:
          procCopy.pragma = newNimNode(nnkPragma)
        procCopy.pragma.add(ident"base")
    else:
      let procNode = newNimNode(nnkProcDef)
      copyChildrenTo(procCopy, procNode)
      procCopy = procNode

    # Strip DSL-specific pragmas
    stripDslPragmas(procCopy)

    if not methodDef.insertUnmodified and methodDef.isStatic:
      # Static: inject typedesc as first param
      let classTypedesc = newNimNode(nnkBracketExpr)
      classTypedesc.add(ident"typedesc")
      classTypedesc.add(classDef.fullType)
      procCopy.params.insert(1, newIdentDefs(ident"_", classTypedesc))

    elif not methodDef.insertUnmodified and methodDef.isConst:
      # Immutable instance method: inject `this: ClassName` (no var)
      procCopy.params.insert(1, newIdentDefs(ident"this", classDef.fullType))

    elif not methodDef.insertUnmodified:
      # Instance method: inject `this: ClassName`
      # For ref objects, `this` is already a reference — no `var` needed
      # unless we want mutation of the reference itself (re-seating).
      # Chapel uses `borrowed` for `this`, which is just the reference.
      procCopy.params.insert(1, newIdentDefs(ident"this", classDef.fullType))

    # Inject generic params
    if not methodDef.insertUnmodified:
      procCopy.addGenericParams(classDef)

    # Forward declaration — skip for generic classes and methods using
    # dynamic dispatch (Nim handles method forward decls differently)
    if not methodDef.insertUnmodified and not classDef.isGeneric and not useDynamic:
      classDef.fwdDecl.add(procCopy)

    # Full body
    var procWithBody = copyNimTree(procCopy)
    procWithBody.body = copyNimTree(methodDef.body)
    classDef.body.add(procWithBody)
    methodDef.outputCode = procWithBody

#[ Debug ]#

proc debugEcho(classDef: ClassDescription) =
  for methodDef in classDef.methods.definitions:
    let dispatch = if methodDef.useMethodDispatch(): " (dynamic)" else: " (static)"
    echo "- Method: " & $methodDef.definition.name & dispatch

#[ Register ]#

static:
  classCompilerHooks.add(proc(stage: ClassCompilerStage, classDef: ClassDescription) =
    if stage == ClassGatherDefinitions: gatherDefinitions(classDef)
    if stage == ClassGenerateCode: generateCode(classDef)
    if stage == ClassDebugEcho: debugEcho(classDef)
  )
