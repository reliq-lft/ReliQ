#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/classes/classcopy.nim
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

## Generate Chapel-style copy operations for classes.
##
## Chapel class semantics differ from records:
##   - Assignment is **by reference** (both variables point to the same instance).
##   - Explicit ``clone()`` creates a **new** instance with copied fields.
##   - ``borrow()`` returns the reference itself (a no-op for ref objects).
##
## We generate:
##   - ``clone(other: T): T``  — allocates a new instance, copies all fields.
##   - ``copyFrom(lhs: T, rhs: T)`` — copies all fields from rhs into lhs
##     (same instance, fields overwritten).
##   - ``borrow(this: T): T`` — returns the same reference (identity).
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

import std/macros
import ./classinternal
import ./classvar
import ./classutils

proc generateCopyOps(classDef: ClassDescription) =
  let className = classDef.fullType

  # --- clone: allocate a new ref object and copy all fields ---
  let resIdent = ident"res"
  let otherIdent = ident"other"
  var cloneBody = newStmtList()
  cloneBody.add(newVarStmt(resIdent, newCall(className)))
  for varDef in classDef.vars.definitions:
    let name = varDef.definition.variableName
    cloneBody.add(newAssignment(
      newDotExpr(resIdent, name),
      newDotExpr(otherIdent, name)))
  cloneBody.add(newNimNode(nnkReturnStmt).add(resIdent))

  let cloneProc = quote do:
    proc clone*(`otherIdent`: `className`): `className` =
      `cloneBody`
  cloneProc.addGenericParams(classDef)
  classDef.body.add(cloneProc)

  # --- copyFrom: copy all fields from rhs to lhs (same instance) ---
  let lhsIdent = ident"lhs"
  let rhsIdent = ident"rhs"
  var assignBody = newStmtList()
  for varDef in classDef.vars.definitions:
    let name = varDef.definition.variableName
    assignBody.add(newAssignment(
      newDotExpr(lhsIdent, name),
      newDotExpr(rhsIdent, name)))

  let copyFromProc = quote do:
    proc copyFrom*(`lhsIdent`: `className`, `rhsIdent`: `className`) =
      `assignBody`
  copyFromProc.addGenericParams(classDef)
  classDef.body.add(copyFromProc)

  # --- borrow: return the same reference (Chapel borrowed semantics) ---
  let thisIdent = ident"this"
  let borrowProc = quote do:
    proc borrow*(`thisIdent`: `className`): `className` =
      return `thisIdent`
  borrowProc.addGenericParams(classDef)
  classDef.body.add(borrowProc)

static:
  classCompilerHooks.add(proc(stage: ClassCompilerStage, classDef: ClassDescription) =
    if stage == ClassGenerateCode: generateCopyOps(classDef)
  )
