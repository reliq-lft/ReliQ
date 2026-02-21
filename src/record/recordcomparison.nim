#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/record/recordcomparison.nim
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

## Generate Chapel-style default comparison operators for records.
##
## Chapel semantics:
##   - `==` / `!=` : field-by-field equality.
##   - `<`, `<=`, `>`, `>=` : lexicographic ordering over fields.
##   - The user may override these externally.

{.used.}

import std/[macros]

import recordinternal
import recordvar
import recordmethods
import recordutils

#[ Generate comparison operators ]#

proc generateComparisons(recDef: RecordDescription) =
  let className = recDef.fullType
  let fields = recDef.vars.definitions

  # Collect the names of comparison operators already defined by the user
  # as methods in this record body.  If the user provides e.g. `method ==`,
  # we skip auto-generating that operator so the user version wins.
  var userDefined: set[char] = {}  # use chars as simple flags: '=' == !=, '<' <, 'l' <=
  for m in recDef.methods.definitions:
    let n = $m.definition.name
    if n == "==": userDefined.incl('=')
    elif n == "!=": userDefined.incl('!')
    elif n == "<": userDefined.incl('<')
    elif n == "<=": userDefined.incl('l')

  let eqOp = ident"=="
  let neqOp = ident"!="
  let ltOp = ident"<"
  let leOp = ident"<="
  let andOp = ident"and"
  let notOp = ident"not"
  let orOp = ident"or"
  let lhsIdent = ident"lhs"
  let rhsIdent = ident"rhs"

  # `==`
  if '=' notin userDefined:
    var eqBody: NimNode
    if fields.len == 0:
      eqBody = quote do: return true
    else:
      var cond: NimNode = nil
      for varDef in fields:
        let name = varDef.definition.variableName
        let fieldCmp = newCall(
          eqOp,
          newDotExpr(lhsIdent, name),
          newDotExpr(rhsIdent, name)
        )
        if cond == nil: cond = fieldCmp
        else: cond = newCall(andOp, cond, fieldCmp)
      eqBody = newStmtList(newNimNode(nnkReturnStmt).add(cond))

    let eqProc = quote do:
      proc `eqOp`*(`lhsIdent`, `rhsIdent`: `className`): bool =
        `eqBody`
    eqProc.addGenericParams(recDef)
    recDef.body.add(eqProc)

  # `!=`
  if '!' notin userDefined:
    let neqBody = newStmtList(
      newNimNode(nnkReturnStmt).add(newCall(notOp, newCall(eqOp, lhsIdent, rhsIdent)))
    )
    let neqProc = quote do:
      proc `neqOp`*(`lhsIdent`, `rhsIdent`: `className`): bool = `neqBody`
    neqProc.addGenericParams(recDef)
    recDef.body.add(neqProc)

  # `<` (lexicographic)
  if fields.len > 0 and '<' notin userDefined:
    var ltBody = newStmtList()
    for varDef in fields:
      let name = varDef.definition.variableName
      let lf = newDotExpr(lhsIdent, name)
      let rf = newDotExpr(rhsIdent, name)
      ltBody.add(newIfStmt(
        (newCall(ltOp, lf, rf), newStmtList(newNimNode(nnkReturnStmt).add(newLit(true))))
      ))
      ltBody.add(newIfStmt(
        (newCall(neqOp, copyNimTree(lf), copyNimTree(rf)),
         newStmtList(newNimNode(nnkReturnStmt).add(newLit(false))))
      ))
    ltBody.add(newNimNode(nnkReturnStmt).add(newLit(false)))

    let ltProc = quote do:
      proc `ltOp`*(`lhsIdent`, `rhsIdent`: `className`): bool =
        `ltBody`
    ltProc.addGenericParams(recDef)
    recDef.body.add(ltProc)

  # `<=`
  if fields.len > 0 and 'l' notin userDefined:
    let leBody = newStmtList(
      newNimNode(nnkReturnStmt).add(
        newCall(orOp,
          newCall(ltOp, lhsIdent, rhsIdent),
          newCall(eqOp, lhsIdent, rhsIdent)
    ) ) )
    let leProc = quote do:
      proc `leOp`*(`lhsIdent`, `rhsIdent`: `className`): bool =
        `leBody`
    leProc.addGenericParams(recDef)
    recDef.body.add(leProc)

#[ Hash (Chapel: default hash for records) ]#

proc generateHash(recDef: RecordDescription) =
  let className = recDef.fullType
  let fields = recDef.vars.definitions
  let thisIdent = ident"this"
  let hIdent = ident"h"

  let hashIdent = ident"hash"
  let combineOp = ident"!&"

  var hashBody = newStmtList()
  hashBody.add(newVarStmt(hIdent, newCall(ident"Hash", newLit(0))))
  for varDef in fields:
    let name = varDef.definition.variableName
    let fieldAccess = newDotExpr(thisIdent, name)
    # h = h !& hash(this.field)
    let hashCall = newCall(hashIdent, fieldAccess)
    let combineExpr = newCall(combineOp, hIdent, hashCall)
    hashBody.add(newAssignment(hIdent, combineExpr))
  hashBody.add(newNimNode(nnkReturnStmt).add(newCall(ident"!$", hIdent)))

  let hashProc = quote do:
    proc `hashIdent`*(`thisIdent`: `className`): Hash =
      `hashBody`
  hashProc.addGenericParams(recDef)
  recDef.body.add(hashProc)

#[ Register ]#

static:
  recordCompilerHooks.add(
    proc(stage: RecordCompilerStage, recDef: RecordDescription) =
      if stage == RecordGenerateCode:
        generateComparisons(recDef)
        generateHash(recDef)
  )
