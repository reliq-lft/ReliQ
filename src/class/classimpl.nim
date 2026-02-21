#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/classes/classimpl.nim
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

## Rust-style ``classImpl`` blocks for adding methods to an already-defined class.
##
## Usage
## -----
## .. code-block:: nim
##   class Foo:
##     var x: int
##
##   classImpl Foo:
##     method doubled(): int =
##       return this.x * 2
##
## The ``classImpl`` macro looks up a previously compiled class by name,
## then parses the body for ``method`` declarations and generates
## the corresponding ``proc``/``method`` definitions with ``this`` injection,
## generic parameters, and export markers — identical to methods
## declared inside the original ``class`` body.
##
## ``classImpl`` blocks support the same method features as ``class`` bodies:
##   - Instance methods (receive ``this: ClassName``, dynamic dispatch)
##   - Immutable methods (``{.immutable.}``, receive ``this: ClassName``)
##   - Static methods (``{.static.}``, receive ``_: typedesc[ClassName]``)
##   - Override methods (``{.override.}``, override a parent method)
##
## Multiple ``classImpl`` blocks can be used for the same class, and they
## can appear in different modules.
##
## Architecture adapted from jjv360/nim-classes 
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

import std/[macros]
import std/[strutils]
import classinternal
import classmethods
import classutils
import classvar

#[ DSL pragmas to strip from generated procs ]#

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

#[ Resolve the class name from the head node ]#

proc resolveClassName(head: NimNode): NimNode =
  if head.kind == nnkIdent:
    return head
  elif head.kind == nnkBracketExpr:
    return head[0]
  elif head.kind == nnkPostfix and $head[0] == "*":
    return head[1]
  elif head.kind == nnkInfix and $head[0] == "*":
    return head[1]
  else:
    error("Invalid classImpl syntax: expected a class name.", head)

#[ Generate convenience constructors for an init method defined in classImpl ]#

proc generateClassImplConstructors(
    code: NimNode, classDef: ClassDescription, initProcDef: NimNode) =
  let classType = classDef.fullType
  let underscore = ident"_"

  # Collect user params (skip slot 0=return type, slot 1=this)
  var userParams: seq[NimNode] = @[]
  for i in 2 ..< initProcDef.params.len:
    userParams.add(copyNimTree(initProcDef.params[i]))

  # Check if there's a postinit method
  var hasPostinit = false
  for m in classDef.methods.definitions:
    if $m.definition.name == "postinit":
      hasPostinit = true
      break

  proc makeConstructor(procName: NimNode, isTypedesc: bool): NimNode =
    var def: NimNode
    if isTypedesc:
      def = quote do:
        proc `procName`*(`underscore`: typedesc[`classType`]): `classType`
    else:
      def = quote do:
        proc `procName`*(): `classType`

    def.addGenericParams(classDef)

    let initCall = newCall(newDotExpr(ident"o", ident"init"))
    for p in userParams:
      for j in 0 ..< p.len - 2:
        initCall.add(p[j])

    var body = newStmtList(
      newVarStmt(ident"o", newCall(classType)),
      newAssignment(ident"o", initCall),
    )
    if hasPostinit:
      body.add(newCall(newDotExpr(ident"o", ident"postinit")))
    body.add(newNimNode(nnkReturnStmt).add(ident"o"))
    def.body = body

    for p in userParams:
      def.params.add(copyNimTree(p))

    return def

  code.add(makeConstructor(ident"init", isTypedesc = true))
  code.add(makeConstructor(ident("new" & $classDef.name), isTypedesc = false))
  code.add(makeConstructor(ident"new", isTypedesc = true))

#[ Determine if a method should use dynamic dispatch ]#

proc useMethodDispatch(name: string, isStatic, isOverride: bool, insertUnmodified: bool): bool =
  if name == "init" or name == "deinit" or name == "postinit":
    return false
  if isStatic:
    return false
  if insertUnmodified:
    return false
  return true

#[ The classImpl macro ]#

macro classImpl*(head: untyped, body: untyped): untyped =
  ## Add methods to an already-defined class outside its original body.
  let nameNode = resolveClassName(head)
  let classDef = classDefinitionFor(nameNode)
  if classDef == nil:
    error("No class named '" & $nameNode & "' found. " &
          "The class must be defined before the classImpl block.", head)

  var code = newStmtList()

  traverseClassStatementList body, proc(idx: int, parent: NimNode, node: NimNode) =
    if node.kind == nnkCommentStmt:
      return

    if node.kind != nnkMethodDef:
      if node.kind in RoutineNodes:
        error("Only 'method' declarations are allowed inside a classImpl block.", node)
      else:
        error("Unexpected node in classImpl block: " & $node.kind, node)

    # Parse method properties
    var isStatic = false
    var isConst = false
    var isOverride = false
    for p in node.pragma:
      if p.kind == nnkIdent and $p == "static":
        isStatic = true
      elif p.kind == nnkIdent and $p == "immutable":
        isConst = true
      elif p.kind == nnkIdent and $p == "override":
        isOverride = true

    var methodBody = copyNimTree(node.body)
    if methodBody.kind == nnkEmpty:
      let mName = $node.name
      let text = $classDef.name & "." & mName & "() is not classImpled."
      methodBody = newStmtList(
        quote do: raiseAssert(`text`)
      )

    var procDef = copyNimTree(node)
    procDef.body = newEmptyNode()

    let methodName = $node.name
    let useDynamic = useMethodDispatch(methodName, isStatic, isOverride, false)

    # Convert to appropriate node type
    if useDynamic:
      let methodNode = newNimNode(nnkMethodDef)
      copyChildrenTo(procDef, methodNode)
      procDef = methodNode
    else:
      let procNode = newNimNode(nnkProcDef)
      copyChildrenTo(procDef, procNode)
      procDef = procNode

    # Strip DSL pragmas
    stripDslPragmas(procDef)

    # For dynamic dispatch methods that are NOT overrides, add {.base.}
    if useDynamic and not isOverride:
      if procDef.pragma.kind == nnkEmpty:
        procDef.pragma = newNimNode(nnkPragma)
      procDef.pragma.add(ident"base")

    # Export
    if procDef[0].kind != nnkPostfix:
      let nameIdent = procDef[0]
      procDef[0] = newNimNode(nnkPostfix, nameIdent)
      procDef[0].add(ident"*")
      procDef[0].add(nameIdent)

    # Inject this/typedesc parameter
    if isStatic:
      let classTypedesc = newNimNode(nnkBracketExpr)
      classTypedesc.add(ident"typedesc")
      classTypedesc.add(classDef.fullType)
      procDef.params.insert(1, newIdentDefs(ident"_", classTypedesc))
    elif isConst:
      procDef.params.insert(1, newIdentDefs(ident"this", classDef.fullType))
    else:
      # For ref objects, `this` is already a reference
      procDef.params.insert(1, newIdentDefs(ident"this", classDef.fullType))

    procDef.addGenericParams(classDef)

    var procWithBody = copyNimTree(procDef)
    procWithBody.body = methodBody

    # Special handling for init methods
    let isInit = methodName == "init" and not isStatic
    if isInit:
      procWithBody.params[0] = classDef.fullType

      var initCode = newStmtList()
      for varDef in classDef.vars.definitions:
        let valueNode = varDef.definition[2]
        if valueNode.kind == nnkEmpty:
          continue
        let nameDef = varDef.definition.variableName
        initCode.add(quote do:
          this.`nameDef` = `valueNode`
        )

      if initCode.len > 0:
        procWithBody.body.insert(0, initCode)

      procWithBody.body.add(quote do:
        return this
      )

    code.add(procWithBody)

    if isInit:
      generateClassImplConstructors(code, classDef, procWithBody)

    # Generate lifecycle hooks: =copy, =sink, =dup, =wasMoved, =trace
    const lifecycleHooks = ["copy", "sink", "dup", "wasMoved", "trace"]
    if methodName in lifecycleHooks and not isStatic:
      let hookIdent = ident("`=" & methodName & "`")
      let thisIdent = ident"this"
      let call = newCall(newDotExpr(thisIdent, ident(methodName)))
      for i in 2 ..< procWithBody.params.len:
        let paramGroup = procWithBody.params[i]
        for j in 0 ..< paramGroup.len - 2:
          call.add(copyNimTree(paramGroup[j]))
      let hookProc =
        if methodName == "dup":
          newProc(
            name = newNimNode(nnkPostfix).add(ident"*", hookIdent),
            params = @[
              copyNimTree(classDef.fullType),
              newIdentDefs(thisIdent, copyNimTree(classDef.fullType))
            ],
            body = newStmtList(call)
          )
        else:
          newProc(
            name = newNimNode(nnkPostfix).add(ident"*", hookIdent),
            params = @[
              newEmptyNode(),
              newIdentDefs(thisIdent, newNimNode(nnkVarTy).add(classDef.fullType))
            ],
            body = newStmtList(call)
          )
      for i in 2 ..< procWithBody.params.len:
        hookProc.params.add(copyNimTree(procWithBody.params[i]))
      hookProc.addGenericParams(classDef)
      code.add(hookProc)

  # Debug output
  const debugclasses {.strdefine.} = ""
  let debugNames = debugclasses.split(",")
  if debugclasses == "true" or debugNames.contains($nameNode):
    echo "\n=== classImpl " & $nameNode
    echo code.repr
    echo ""

  return code
