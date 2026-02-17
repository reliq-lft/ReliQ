#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/classes/classcomparison.nim
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

## Generate Chapel-style default comparison operators for classes.
##
## Chapel class semantics:
##   - ``==`` / ``!=``: by default compare **identity** (same ref).
##   - Field-by-field equality can be opted into by the user.
##   - ``<``, ``<=``, ``>``, ``>=``: lexicographic ordering over fields,
##     same as records.
##
## We provide both identity comparison and field-wise equality/ordering
## so users have full Chapel-like flexibility:
##   - ``==`` / ``!=``: field-by-field equality (matching Chapel's
##     compiler-generated default comparison for classes with fields).
##   - ``<``, ``<=``: lexicographic ordering over fields.
##   - ``===`` / ``!==``: identity (reference) comparison.
##   - ``hash``: field-based hash (for use in Tables/Sets).
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

{.used.}

import std/[macros]

import classinternal
import classvar
import classutils

proc generateComparisons(classDef: ClassDescription) =
  let className = classDef.fullType
  let fields = classDef.vars.definitions
  let eqOp = ident"=="
  let neqOp = ident"!="
  let ltOp = ident"<"
  let leOp = ident"<="
  let andOp = ident"and"
  let notOp = ident"not"
  let orOp = ident"or"
  let lhsIdent = ident"lhs"
  let rhsIdent = ident"rhs"

  # `==` — field-by-field equality (handles nil)
  var eqBody: NimNode
  if fields.len == 0:
    # No fields: just check identity or both nil
    eqBody = quote do:
      if lhs.isNil and rhs.isNil: return true
      if lhs.isNil or rhs.isNil: return false
      return true
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

    eqBody = quote do:
      if lhs.isNil and rhs.isNil: return true
      if lhs.isNil or rhs.isNil: return false
    eqBody.add(newNimNode(nnkReturnStmt).add(cond))

  let eqProc = quote do:
    proc `eqOp`*(`lhsIdent`, `rhsIdent`: `className`): bool =
      `eqBody`
  eqProc.addGenericParams(classDef)

  # `!=`
  let neqBody = newStmtList(
    newNimNode(nnkReturnStmt).add(newCall(notOp, newCall(eqOp, lhsIdent, rhsIdent)))
  )
  let neqProc = quote do:
    proc `neqOp`*(`lhsIdent`, `rhsIdent`: `className`): bool = `neqBody`
  neqProc.addGenericParams(classDef)

  if fields.len == 0 or classDef.isGeneric:
    # No fields or generic class — emit unconditionally
    classDef.body.add(eqProc)
    classDef.body.add(neqProc)
  else:
    # Guard with when compiles(default(FieldType) == default(FieldType)) for each field
    var eqCompiles: NimNode = nil
    for varDef in fields:
      let fieldType = varDef.definition[1]
      let check = newCall(ident"compiles", newCall(eqOp,
        newCall(ident"default", copyNimTree(fieldType)),
        newCall(ident"default", copyNimTree(fieldType))
      ))
      if eqCompiles == nil: eqCompiles = check
      else: eqCompiles = newCall(andOp, eqCompiles, check)
    let eqWhen = newNimNode(nnkWhenStmt).add(
      newNimNode(nnkElifBranch).add(eqCompiles, newStmtList(eqProc, neqProc))
    )
    classDef.body.add(eqWhen)

  # `<` (lexicographic) — only generated when all fields support `<`
  if fields.len > 0:
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
    ltProc.addGenericParams(classDef)

    # `<=`
    let leBody = newStmtList(
      newNimNode(nnkReturnStmt).add(
        newCall(orOp,
          newCall(ltOp, lhsIdent, rhsIdent),
          newCall(eqOp, lhsIdent, rhsIdent)
    ) ) )
    let leProc = quote do:
      proc `leOp`*(`lhsIdent`, `rhsIdent`: `className`): bool =
        `leBody`
    leProc.addGenericParams(classDef)

    if classDef.isGeneric:
      # Generic classes: emit unconditionally (resolved at instantiation)
      classDef.body.add(ltProc)
      classDef.body.add(leProc)
    else:
      # Non-generic: guard with when compiles(default(FieldType) < default(FieldType))
      var allCompile: NimNode = nil
      for varDef in fields:
        let fieldType = varDef.definition[1]
        let check = newCall(ident"compiles", newCall(ltOp,
          newCall(ident"default", copyNimTree(fieldType)),
          newCall(ident"default", copyNimTree(fieldType))
        ))
        if allCompile == nil: allCompile = check
        else: allCompile = newCall(andOp, allCompile, check)

      let whenBlock = newNimNode(nnkWhenStmt).add(
        newNimNode(nnkElifBranch).add(allCompile, newStmtList(ltProc, leProc))
      )
      classDef.body.add(whenBlock)

#[ Identity comparison: === and !== ]#

proc generateIdentityComparison(classDef: ClassDescription) =
  let className = classDef.fullType
  let lhsIdent = ident"lhs"
  let rhsIdent = ident"rhs"

  # `===` — reference identity
  let identEqBody = quote do:
    if `lhsIdent`.isNil and `rhsIdent`.isNil: return true
    if `lhsIdent`.isNil or `rhsIdent`.isNil: return false
    return cast[pointer](`lhsIdent`) == cast[pointer](`rhsIdent`)

  let identEqName = ident"==="
  let identEqProc = quote do:
    proc `identEqName`*(`lhsIdent`, `rhsIdent`: `className`): bool =
      `identEqBody`
  identEqProc.addGenericParams(classDef)
  classDef.body.add(identEqProc)

  # `!==` — reference inequality
  let identNeqName = ident"!=="
  let identNeqProc = quote do:
    proc `identNeqName`*(`lhsIdent`, `rhsIdent`: `className`): bool =
      return not (`lhsIdent` === `rhsIdent`)
  identNeqProc.addGenericParams(classDef)
  classDef.body.add(identNeqProc)

#[ Hash (field-based) ]#

proc generateHash(classDef: ClassDescription) =
  let className = classDef.fullType
  let fields = classDef.vars.definitions
  let thisIdent = ident"this"
  let hIdent = ident"h"
  let hashIdent = ident"hash"
  let combineOp = ident"!&"

  var hashBody = newStmtList()
  hashBody.add(newVarStmt(hIdent, newCall(ident"Hash", newLit(0))))
  for varDef in fields:
    let name = varDef.definition.variableName
    let fieldAccess = newDotExpr(thisIdent, name)
    let hashCall = newCall(hashIdent, fieldAccess)
    let combineExpr = newCall(combineOp, hIdent, hashCall)
    hashBody.add(newAssignment(hIdent, combineExpr))
  hashBody.add(newNimNode(nnkReturnStmt).add(newCall(ident"!$", hIdent)))

  let hashProc = quote do:
    proc `hashIdent`*(`thisIdent`: `className`): Hash =
      `hashBody`
  hashProc.addGenericParams(classDef)

  if fields.len == 0 or classDef.isGeneric:
    classDef.body.add(hashProc)
  else:
    var hashCompiles: NimNode = nil
    for varDef in fields:
      let fieldType = varDef.definition[1]
      let check = newCall(ident"compiles", newCall(hashIdent,
        newCall(ident"default", copyNimTree(fieldType))
      ))
      if hashCompiles == nil: hashCompiles = check
      else: hashCompiles = newCall(ident"and", hashCompiles, check)
    let hashWhen = newNimNode(nnkWhenStmt).add(
      newNimNode(nnkElifBranch).add(hashCompiles, newStmtList(hashProc))
    )
    classDef.body.add(hashWhen)

#[ Register ]#

static:
  classCompilerHooks.add(
    proc(stage: ClassCompilerStage, classDef: ClassDescription) =
      if stage == ClassGenerateCode:
        generateComparisons(classDef)
        generateIdentityComparison(classDef)
        generateHash(classDef)
  )
