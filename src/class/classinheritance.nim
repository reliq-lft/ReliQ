#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/classes/classinheritance.nim
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

## Support for class inheritance following Chapel semantics.
##
## Chapel inheritance rules:
##   - All classes derive from ``RootClass`` (mapped to Nim's ``RootObj``).
##   - Single inheritance only: ``class Child of Parent``.
##   - A derived class inherits all fields and methods from the parent.
##   - Fields in a derived class with the same name as a parent field
##     cause a compilation error (no field shadowing).
##   - Methods can be overridden with the ``{.override.}`` pragma.
##   - Dynamic dispatch routes method calls based on the runtime type.
##   - ``super.init()`` can be called to invoke the parent initializer.
##   - If ``super.init()`` is not called in a child initializer, the
##     compiler inserts a zero-argument call at the start.
##   - The compiler-generated initializer for inheriting classes includes
##     parent fields before child fields.
##
## This module generates:
##   - ``isInstanceOf(T)``: runtime type check using ``of`` operator.
##   - ``toParent(Parent)``: upcast to parent type.
##   - ``toChild(Child)``: downcast to child type (may fail at runtime).
##   - ``parentClassName()``: returns the parent class name as a string.
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
import classmethods

proc generateInheritanceHelpers(classDef: ClassDescription) =
  let classType = classDef.fullType
  let thisIdent = ident"this"

  # parentClassName() → static proc returning parent name
  let parentNameStr = if classDef.parentName != nil: $classDef.parentName
                      else: "RootObj"
  let parentNameIdent = ident"parentClassName"
  let parentNameProc = quote do:
    proc `parentNameIdent`*(_: typedesc[`classType`]): string =
      return `parentNameStr`
  parentNameProc.addGenericParams(classDef)
  classDef.body.add(parentNameProc)

  # isInstanceOf[T]() — runtime type check
  # We generate a template so `of` resolves correctly at the call site.
  # Usage: obj.isInstanceOf(ChildClass)
  let isInstName = ident"isInstanceOf"
  # Use a unique name to avoid clashing with the class's own generic params
  let isInstTypeParam = ident"IsInstTarget__"

  # Use a template instead of a proc so `of` works properly with types:
  #   template isInstanceOf*[IsInstTarget](this: ClassName, _: typedesc[IsInstTarget]): bool =
  #     this of IsInstTarget
  let isInstTemplate = newNimNode(nnkTemplateDef)
  isInstTemplate.add(newNimNode(nnkPostfix).add(ident"*", isInstName))  # name
  isInstTemplate.add(newEmptyNode())  # term rewriting patterns
  # Generic params: [IsInstTarget]
  let isInstGenericParams = newNimNode(nnkGenericParams)
  isInstGenericParams.add(newIdentDefs(isInstTypeParam, newEmptyNode(), newEmptyNode()))
  if classDef.isGeneric:
    for identDef in classDef.genericParams:
      isInstGenericParams.add(copyNimTree(identDef))
  isInstTemplate.add(isInstGenericParams)
  # Formal params: (this: ClassName, _: typedesc[IsInstTarget]): bool
  let isInstParams = newNimNode(nnkFormalParams)
  isInstParams.add(ident"bool")
  isInstParams.add(newIdentDefs(thisIdent, classType))
  isInstParams.add(newIdentDefs(ident"_", newNimNode(nnkBracketExpr).add(ident"typedesc", isInstTypeParam)))
  isInstTemplate.add(isInstParams)
  isInstTemplate.add(newEmptyNode())  # pragmas
  isInstTemplate.add(newEmptyNode())  # reserved
  # Body: this of IsInstTarget
  isInstTemplate.add(newStmtList(
    newNimNode(nnkInfix).add(ident"of", thisIdent, isInstTypeParam)
  ))
  classDef.body.add(isInstTemplate)

static:
  classCompilerHooks.add(proc(stage: ClassCompilerStage, classDef: ClassDescription) =
    if stage == ClassGenerateCode: generateInheritanceHelpers(classDef)
  )
