#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/classes/classmeta.nim
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

## Add meta-programming helpers to classes.
##
##   - ``className()``: returns the class type name as a string (static method).
##   - ``$`` (stringify): produces a Chapel-style
##     ``ClassName(field1 = val1, field2 = val2)`` representation.
##   - ``isNil()``: checks if the reference is nil.
##   - ``isInstance(T)``: checks if the instance is of a particular type.

{.used.}

import std/[macros]

import classinternal
import classmethods
import classvar
import classutils

proc addExtras(classDef: ClassDescription) =
  # className() → static proc returning the class's name
  var m = ClassMethod()
  let funcName = ident"className"
  let output = $classDef.name
  m.definition = quote do:
    method `funcName`(): string
  m.body = quote do:
    return `output`
  m.isStatic = true
  classDef.methods.definitions.add(m)

proc generateToString(classDef: ClassDescription) =
  let classType = classDef.fullType
  let fields = classDef.vars.definitions
  let clsName = $classDef.name
  let dollarOp = ident"$"
  let ampEqOp = ident"&="
  let ampOp = ident"&"
  let thisIdent = ident"this"
  let sIdent = ident"s"

  var body = newStmtList()

  # Handle nil
  body.add(quote do:
    if `thisIdent`.isNil:
      return "nil"
  )

  body.add(
    newVarStmt(sIdent,
      newCall(ampOp, newLit(clsName), newLit("(")))
  )

  for i, varDef in fields:
    let name = varDef.definition.variableName
    let nameStr = $name
    if i > 0:
      body.add(
        newCall(ampEqOp, sIdent, newLit(", "))
      )
    body.add(
      newCall(ampEqOp, sIdent,
        newCall(ampOp,
          newLit(nameStr & " = "),
          newCall(dollarOp, newDotExpr(thisIdent, name))))
    )

  body.add(newCall(ampEqOp, sIdent, newLit(")")))
  body.add(newNimNode(nnkReturnStmt).add(sIdent))

  let dollarProc = quote do:
    proc `dollarOp`*(`thisIdent`: `classType`): string =
      `body`
  dollarProc.addGenericParams(classDef)
  classDef.body.add(dollarProc)

proc generateNilCheck(classDef: ClassDescription) =
  ## Generate an ``isNilClass`` proc for the class.
  ## (Nim already has ``isNil`` for ref types; this is a named alias.)
  let classType = classDef.fullType
  let thisIdent = ident"this"
  let isNilName = ident"isNilClass"
  let isNilProc = quote do:
    proc `isNilName`*(`thisIdent`: `classType`): bool =
      return `thisIdent`.isNil
  isNilProc.addGenericParams(classDef)
  classDef.body.add(isNilProc)

static:
  classCompilerHooks.add(proc(stage: ClassCompilerStage, classDef: ClassDescription) =
    if stage == ClassGatherDefinitions: addExtras(classDef)
    if stage == ClassGenerateCode:
      generateToString(classDef)
      generateNilCheck(classDef)
  )
