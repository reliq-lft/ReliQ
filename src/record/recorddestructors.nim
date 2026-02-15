#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/record/recorddestructors.nim
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

## Generate `=destroy` hook when a `deinit()` method is defined.
##
## Chapel semantics:
##   - `deinit()` is called automatically when a record goes out of scope.
##   - If no `deinit` is defined, nothing special happens.
##
## Adapted from jjv360/nim-classes (MIT License).

{.used.}

import std/[macros]

import recordinternal
import recordmethods

proc generateDestructor(recDef: RecordDescription) =
  # Check if this record has a deinit()
  var hasDeinit = false
  for m in recDef.methods.definitions:
    if $m.definition.name == "deinit":
      hasDeinit = true
      break

  if not hasDeinit:
    return

  # Generate `=destroy` for the record's object type.
  # For a plain `object` (not ref), `=destroy` receives `var T`.
  #
  # We build the AST manually (instead of `quote do`) to keep the
  # `deinit` call as an unbound ident.  `quote do` captures whatever
  # `deinit` proc is already in scope, which breaks when multiple
  # records (e.g. Tracked and TrackedBox[T]) each define deinit —
  # the second record's =destroy would call the first record's deinit.
  let className = recDef.fullType

  # Build: this.deinit()
  let thisIdent = ident"this"
  let deinitCall = newCall(newDotExpr(thisIdent, ident"deinit"))

  # Build: {.cast(raises: []).}: this.deinit()
  # AST: PragmaBlock(Pragma(Cast(Empty, ExprColonExpr("raises", Bracket))), StmtList(...))
  let castNode = newNimNode(nnkCast).add(
    newEmptyNode(),
    newNimNode(nnkExprColonExpr).add(
      ident"raises",
      newNimNode(nnkBracket)
    )
  )
  let castPragma = newNimNode(nnkPragma).add(castNode)
  let castBlock = newNimNode(nnkPragmaBlock).add(castPragma, newStmtList(deinitCall))

  # Build: proc `=destroy`*(this: var ClassName) {.raises: [].} = ...
  let raisesAnnotation = newNimNode(nnkExprColonExpr).add(
    ident"raises",
    newNimNode(nnkBracket)
  )
  let destroyProc = newProc(
    name = newNimNode(nnkPostfix).add(ident"*", ident"`=destroy`"),
    params = [
      newEmptyNode(),  # no return type
      newIdentDefs(thisIdent, newNimNode(nnkVarTy).add(className))
    ],
    body = newStmtList(castBlock),
    pragmas = newNimNode(nnkPragma).add(raisesAnnotation)
  )
  destroyProc.addGenericParams(recDef)
  recDef.body.insert(0, destroyProc)

static:
  recordCompilerHooks.add(proc(stage: RecordCompilerStage, recDef: RecordDescription) =
    if stage == RecordGenerateCode: generateDestructor(recDef)
  )
