#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/record/recordmeta.nim
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

## Add meta-programming helpers to records.
##
##   - `recordName()`: returns the record type name as a string.
##   - `$` (stringify): produces a Chapel-style `(field1 = val1, field2 = val2)` repr.
##
## Adapted from jjv360/nim-classes (MIT License).

{.used.}

import std/[macros]

import recordinternal
import recordmethods
import recordvar
import recordutils

proc addExtras(recDef: RecordDescription) =
  # recordName() → static proc returning the record's name
  var m = RecordMethod()
  let funcName = ident"recordName"
  let output = $recDef.name
  m.definition = quote do:
    method `funcName`(): string
  m.body = quote do:
    return `output`
  m.isStatic = true
  recDef.methods.definitions.add(m)

proc generateToString(recDef: RecordDescription) =
  let className = recDef.fullType
  let fields = recDef.vars.definitions
  let recName = $recDef.name
  let dollarOp = ident"$"
  let ampEqOp = ident"&="
  let ampOp = ident"&"
  let thisIdent = ident"this"
  let sIdent = ident"s"

  # Build a `$` proc that produces Chapel-style output:
  #   RecordName(field1 = val1, field2 = val2)
  var body = newStmtList()
  body.add(
    newVarStmt(sIdent,
      newCall(ampOp, newLit(recName), newLit("(")))
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
    proc `dollarOp`*(`thisIdent`: `className`): string =
      `body`
  dollarProc.addGenericParams(recDef)
  recDef.body.add(dollarProc)

static:
  recordCompilerHooks.add(proc(stage: RecordCompilerStage, recDef: RecordDescription) =
    if stage == RecordGatherDefinitions: addExtras(recDef)
    if stage == RecordGenerateCode: generateToString(recDef)
  )
