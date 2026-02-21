#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/record/recordimpl.nim
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

## Rust-style ``impl`` blocks for adding methods to an already-defined record.
##
## Usage
## -----
## .. code-block:: nim
##   record Foo:
##     var x: int
##
##   impl Foo:
##     method doubled(): int =
##       return this.x * 2
##
## The ``impl`` macro looks up a previously compiled record by name,
## then parses the body for ``method`` declarations and generates
## the corresponding ``proc`` definitions with ``this`` injection,
## generic parameters, and export markers — identical to methods
## declared inside the original ``record`` body.
##
## ``impl`` blocks support the same method features as ``record`` bodies:
##   - Instance methods (receive ``this: var RecordName``)
##   - Immutable methods (``{.immutable.}``, receive ``this: RecordName``)
##   - Static methods (``{.static.}``, receive ``_: typedesc[RecordName]``)
##
## Multiple ``impl`` blocks can be used for the same record, and they
## can appear in different modules (as long as the record module is
## imported first).
##
## Generic records are fully supported:
##
## .. code-block:: nim
##   record Box[T]:
##     var value: T
##
##   impl Box:
##     method show(): string =
##       return "Box(" & $this.value & ")"

import std/[macros] 
import std/[strutils]
import recordinternal
import recordutils
import recordvar

#[ DSL pragmas to strip from generated procs ]#

const dslPragmas = ["static", "immutable"]

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

#[ Resolve the record name from the head node ]#

proc resolveRecordName(head: NimNode): NimNode =
  ## Extract the bare record name ident from the impl head.
  ##   impl Foo:             → ident"Foo"
  ##   impl Foo[T]:          → ident"Foo"   (generics come from cached def)
  ##   impl Foo*:            → ident"Foo"   (export marker ignored)
  if head.kind == nnkIdent:
    return head
  elif head.kind == nnkBracketExpr:
    return head[0]
  elif head.kind == nnkPostfix and $head[0] == "*":
    return head[1]
  elif head.kind == nnkInfix and $head[0] == "*":
    return head[1]
  else:
    error("Invalid impl syntax: expected a record name.", head)

#[ Generate convenience constructors for an init method defined in impl ]#

proc generateImplConstructors(
    code: NimNode, recDef: RecordDescription, initProcDef: NimNode) =
  ## Given a fully-built init proc (with this + generic params already
  ## injected), generate the three convenience constructors:
  ##   1. RecordName.init(args...)
  ##   2. newRecordName(args...)
  ##   3. RecordName.new(args...)

  let classType = recDef.fullType
  let underscore = ident"_"

  # Collect user params (skip slot 0=return type, slot 1=this)
  var userParams: seq[NimNode] = @[]
  for i in 2 ..< initProcDef.params.len:
    userParams.add(copyNimTree(initProcDef.params[i]))

  # Helper: build a constructor proc that creates a default instance,
  # calls init(args...) on it, and returns it.
  proc makeConstructor(procName: NimNode, isTypedesc: bool): NimNode =
    var def: NimNode
    if isTypedesc:
      def = quote do:
        proc `procName`*(`underscore`: typedesc[`classType`]): `classType`
      # typedesc param is at slot 1
    else:
      def = quote do:
        proc `procName`*(): `classType`

    def.addGenericParams(recDef)

    # Build body: var o: ClassType; o = o.init(args...); return o
    let initCall = newCall(newDotExpr(ident"o", ident"init"))
    for p in userParams:
      # Each param is an IdentDefs; extract the name(s)
      for j in 0 ..< p.len - 2:
        initCall.add(p[j])

    let body = newStmtList(
      newVarStmt(ident"o", newCall(newDotExpr(recDef.fullType, ident"init"))),
      newNimNode(nnkDiscardStmt).add(initCall),
      newNimNode(nnkReturnStmt).add(ident"o")
    )
    def.body = body

    # Add user params to the proc signature
    for p in userParams:
      def.params.add(copyNimTree(p))

    return def

  # 1. RecordName.init(args...)
  code.add(makeConstructor(ident"init", isTypedesc = true))

  # 2. newRecordName(args...)
  code.add(makeConstructor(ident("new" & $recDef.name), isTypedesc = false))

  # 3. RecordName.new(args...)
  code.add(makeConstructor(ident"new", isTypedesc = true))

#[ The impl macro ]#

macro recordImpl*(head: untyped, body: untyped): untyped =
  ## Add methods to an already-defined record outside its original body.
  let nameNode = resolveRecordName(head)
  let recDef = recordDefinitionFor(nameNode)
  if recDef == nil:
    error("No record named '" & $nameNode & "' found. " &
          "The record must be defined before the impl block.", head)

  var code = newStmtList()

  # Walk the body and process each method definition
  traverseRecordStatementList body, proc(idx: int, parent: NimNode, node: NimNode) =
    if node.kind == nnkCommentStmt:
      return  # skip comments

    if node.kind != nnkMethodDef:
      if node.kind in RoutineNodes:
        error("Only 'method' declarations are allowed inside an impl block.", node)
      else:
        error("Unexpected node in impl block: " & $node.kind, node)

    # Parse method properties
    var isStatic = false
    var isConst = false
    for p in node.pragma:
      if p.kind == nnkIdent and $p == "static":
        isStatic = true
      elif p.kind == nnkIdent and $p == "immutable":
        isConst = true

    # Handle abstract methods (no body)
    var methodBody = copyNimTree(node.body)
    if methodBody.kind == nnkEmpty:
      let mName = $node.name
      let text = $recDef.name & "." & mName & "() is not implemented."
      methodBody = newStmtList(
        quote do: raiseAssert(`text`)
      )

    # Build the proc definition
    var procDef = copyNimTree(node)
    procDef.body = newEmptyNode()  # clear for forward decl

    # Convert nnkMethodDef → nnkProcDef
    let procNode = newNimNode(nnkProcDef)
    copyChildrenTo(procDef, procNode)
    procDef = procNode

    # Strip DSL pragmas
    stripDslPragmas(procDef)

    # Export the method name
    if procDef[0].kind != nnkPostfix:
      let nameIdent = procDef[0]
      procDef[0] = newNimNode(nnkPostfix, nameIdent)
      procDef[0].add(ident"*")
      procDef[0].add(nameIdent)

    # Inject this/typedesc parameter
    if isStatic:
      let classTypedesc = newNimNode(nnkBracketExpr)
      classTypedesc.add(ident"typedesc")
      classTypedesc.add(recDef.fullType)
      procDef.params.insert(1, newIdentDefs(ident"_", classTypedesc))
    elif isConst:
      procDef.params.insert(1, newIdentDefs(ident"this", recDef.fullType))
    else:
      procDef.params.insert(1, newIdentDefs(ident"this",
        newNimNode(nnkVarTy).add(recDef.fullType)))

    # Inject generic params from the record
    procDef.addGenericParams(recDef)

    # Emit the full proc with body
    var procWithBody = copyNimTree(procDef)
    procWithBody.body = methodBody

    # Special handling for init methods: inject field defaults,
    # set return type, append `return this`, generate constructors
    let isInit = $node.name == "init" and not isStatic
    if isInit:
      # Set return type to the record's full type
      procWithBody.params[0] = recDef.fullType

      # Build field-default-injection code
      var initCode = newStmtList()
      for varDef in recDef.vars.definitions:
        let valueNode = varDef.definition[2]
        if valueNode.kind == nnkEmpty:
          continue
        let nameDef = varDef.definition.variableName
        initCode.add(quote do:
          this.`nameDef` = `valueNode`
        )

      # Inject field defaults at the top of the body
      if initCode.len > 0:
        procWithBody.body.insert(0, initCode)

      # Append `return this` at the end
      procWithBody.body.add(quote do:
        return this
      )

    code.add(procWithBody)

    # Generate convenience constructors for init
    if isInit:
      generateImplConstructors(code, recDef, procWithBody)

  # Debug output
  const debugrecords {.strdefine.} = ""
  let debugNames = debugrecords.split(",")
  if debugrecords == "true" or debugNames.contains($nameNode):
    echo "\n=== impl " & $nameNode
    echo code.repr
    echo ""

  return code
