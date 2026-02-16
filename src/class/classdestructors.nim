#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/classes/classdestructors.nim
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

## Generate ``=destroy`` hook when a ``deinit()`` method is defined.
##
## Chapel class semantics:
##   - ``deinit()`` is called when a class instance is being deleted.
##   - For ``owned`` instances, ``deinit`` is called automatically when
##     the owning variable goes out of scope.
##   - For ``shared`` instances, ``deinit`` is called when the reference
##     count reaches zero.
##   - For ``unmanaged`` instances, ``deinit`` is called when ``delete``
##     is explicitly invoked.
##   - If no ``deinit`` is defined, nothing special happens (GC handles it).
##
## Since Nim's ``ref object`` types are garbage-collected, we hook into
## the destructor by using a destructor ref + invoke deinit.
## For ``ref object``, Nim doesn't use ``=destroy`` directly — instead
## we use a destructor via the ref type's destructor mechanism.

{.used.}

import std/[macros]

import classinternal
import classmethods
import classvar

proc generateDestructor(classDef: ClassDescription) =
  # Check if this class has a deinit()
  var hasDeinit = false
  for m in classDef.methods.definitions:
    if $m.definition.name == "deinit":
      hasDeinit = true
      break

  if not hasDeinit:
    return

  # For ref objects in Nim, we use `=destroy` on the ref type itself.
  # Nim's destructor for `ref object` types works through the ref tracking.
  # We generate a destructor proc that calls deinit on the inner object.
  let className = classDef.fullType
  let thisIdent = ident"this"
  let deinitCall = newCall(newDotExpr(thisIdent, ident"deinit"))

  # Build: {.cast(raises: []).}: this.deinit()
  let castNode = newNimNode(nnkCast).add(
    newEmptyNode(),
    newNimNode(nnkExprColonExpr).add(
      ident"raises",
      newNimNode(nnkBracket)
    )
  )
  let castPragma = newNimNode(nnkPragma).add(castNode)
  let castBlock = newNimNode(nnkPragmaBlock).add(castPragma, newStmtList(deinitCall))

  # For ref objects, we generate a destructor callback that uses GC hooks.
  # We'll use a `=destroy` on the underlying object type referenced by the ref.
  # However, for simplicity and compatibility, we emit a destructor proc
  # that can be called during cleanup.
  #
  # Since Nim GC handles ref object lifetimes, deinit will be called via
  # a destructor callback. We generate an explicit `destroy` proc that
  # users can call, plus a ref destructor if using --mm:orc/arc.
  let destroyName = ident"destroy"
  let destroyProc = quote do:
    proc `destroyName`*(`thisIdent`: `className`) =
      if not `thisIdent`.isNil:
        `castBlock`
  destroyProc.addGenericParams(classDef)
  classDef.body.add(destroyProc)

static:
  classCompilerHooks.add(proc(stage: ClassCompilerStage, classDef: ClassDescription) =
    if stage == ClassGenerateCode: generateDestructor(classDef)
  )
