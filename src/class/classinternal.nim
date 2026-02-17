#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/classes/classinternal.nim
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

## Core types and compilation pipeline for the ``class`` macro.
##
## Chapel-style classes (https://chapel-lang.org/docs/language/spec/classes.html):
##   - Reference semantics (``ref object``)
##   - Single inheritance (all classes derive from ``RootClass``)
##   - Memory management strategies: ``owned``, ``shared``, ``borrowed``, ``unmanaged``
##   - Nilable class types with ``?`` suffix
##   - Dynamic dispatch for overridable methods
##   - ``override`` keyword for overriding base class methods
##   - ``deinit`` for destructor/cleanup
##   - ``super`` for invoking parent class initializer/methods
##
## Architecture mirrors ``recordinternal.nim`` — a hook-based compile-time
## plugin system where each ``class`` invocation passes through a series
## of compilation stages.
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
import std/[tables]
import std/[strutils]

## Declare a class method as static (no ``this`` parameter).
template static* {.pragma.}

## Declare a class method as immutable (``this`` is not ``var``).
template immutable* {.pragma.}

## Mark a method as overriding a base class method.
template override* {.pragma.}

type ClassDescription* = ref object of RootRef
  macroName*: string
  name*: NimNode
  parentName*: NimNode        ## ident of the parent class (nil → RootClass)
  genericParams*: NimNode     ## nnkGenericParams or nnkEmpty if non-generic
  inputBody*: NimNode
  comment*: NimNode
  metadata*: Table[string, RootRef]
  prefix*: NimNode
  output*: NimNode
  fwdDecl*: NimNode
  body*: NimNode
  suffix*: NimNode

type ClassCompilerStage* = enum
  ClassPreload
  ClassGatherDefinitions
  ClassAddExtraDefinitions
  ClassModifyDefinitions
  ClassGenerateCode
  ClassFinalize
  ClassDebugEcho

#[ Hook system ]#

type ClassCompilerHook* =
  proc(stage: ClassCompilerStage, classDef: ClassDescription)

## All registered hooks (populated at compile time).
var classCompilerHooks* {.compileTime.}: seq[ClassCompilerHook] = @[]

## Cache of already-compiled class definitions.
var classCompilerCache* {.compileTime.}: seq[ClassDescription] = @[]

#[ Helpers ]#

proc classDefinitionFor*(name: NimNode): ClassDescription =
  ## Look up an already-compiled class by name, or nil.
  if name.kind != nnkIdent and name.kind != nnkSym:
    error("Expected an identifier.", name)
  for classDef in classCompilerCache:
    if $classDef.name == $name:
      return classDef
  return nil

proc hasParent*(classDef: ClassDescription): bool =
  ## True when this class has an explicit parent (not RootClass).
  classDef.parentName != nil and $classDef.parentName != "RootObj"

proc effectiveParent*(classDef: ClassDescription): NimNode =
  ## Returns the parent class name or ``RootObj`` if none specified.
  if classDef.parentName != nil:
    return classDef.parentName
  return ident"RootObj"

#[ Generic helpers ]#

proc isGeneric*(classDef: ClassDescription): bool =
  ## True when the class has generic type parameters.
  classDef.genericParams.kind != nnkEmpty

proc fullType*(classDef: ClassDescription): NimNode =
  ## Returns the fully-qualified type node.
  ##   Non-generic:  Name
  ##   Generic:      Name[T, U, ...]
  if not classDef.isGeneric:
    return classDef.name
  # Build nnkBracketExpr(Name, T, U, ...)
  let bracket = newNimNode(nnkBracketExpr)
  bracket.add(classDef.name)
  for identDef in classDef.genericParams:
    for i in 0 ..< identDef.len - 2:
      bracket.add(copyNimTree(identDef[i]))
  return bracket

proc addGenericParams*(procNode: NimNode, classDef: ClassDescription) =
  ## Inject the class's generic parameters into a proc/method node.
  if not classDef.isGeneric:
    return
  let gp = copyNimTree(classDef.genericParams)
  if procNode[2].kind == nnkEmpty:
    procNode[2] = gp
  else:
    for identDef in gp:
      procNode[2].add(identDef)

#[ Construction ]#

proc parseClassHead(head: NimNode): tuple[name: NimNode, parent: NimNode, generics: NimNode] =
  ## Parse the class head syntax and extract name, parent, and generic params.
  ##
  ## Supported forms:
  ##   class MyClass:                        → name=MyClass, parent=nil
  ##   class MyClass of Parent:              → name=MyClass, parent=Parent
  ##   class MyClass[T]:                     → name=MyClass[T], parent=nil
  ##   class MyClass[T] of Parent:           → name=MyClass[T], parent=Parent
  var name: NimNode = nil
  var parent: NimNode = nil
  var generics = newEmptyNode()

  proc extractGenerics(node: NimNode): tuple[nm: NimNode, gp: NimNode] =
    ## Given ident or BracketExpr, extract name and generic params.
    if node.kind == nnkIdent:
      return (node, newEmptyNode())
    elif node.kind == nnkBracketExpr:
      let gp = newNimNode(nnkGenericParams)
      for i in 1 ..< node.len:
        let param = node[i]
        if param.kind == nnkExprColonExpr:
          gp.add(newIdentDefs(param[0], param[1], newEmptyNode()))
        else:
          gp.add(newIdentDefs(param, newEmptyNode(), newEmptyNode()))
      return (node[0], gp)
    elif node.kind == nnkPostfix and $node[0] == "*":
      return extractGenerics(node[1])
    else:
      error("Invalid class name syntax.", node)

  if head.kind == nnkIdent:
    # class MyClass:
    name = head

  elif head.kind == nnkPostfix and $head[0] == "*":
    # class MyClass*:
    name = head[1]

  elif head.kind == nnkBracketExpr:
    # class MyClass[T]:
    let (nm, gp) = extractGenerics(head)
    name = nm
    generics = gp

  elif head.kind == nnkInfix and $head[0] == "of":
    # class MyClass of Parent:
    # class MyClass[T] of Parent:
    let lhs = head[1]
    let rhs = head[2]
    let (nm, gp) = extractGenerics(lhs)
    name = nm
    generics = gp
    if rhs.kind == nnkIdent:
      parent = rhs
    elif rhs.kind == nnkBracketExpr:
      parent = rhs[0]
    else:
      error("Invalid parent class syntax.", rhs)

  elif head.kind == nnkInfix and $head[0] == "*":
    # class MyClass*[T]:     → nnkInfix("*", ident"MyClass", bracket/bracketExpr)
    # class MyClass* of ...: alternate parse paths
    let rhs = head[2]
    if rhs.kind == nnkEmpty or (rhs.kind == nnkIdent and $rhs == ""):
      name = head[1]
    elif rhs.kind in {nnkBracket, nnkBracketExpr}:
      name = head[1]
      let gp = newNimNode(nnkGenericParams)
      let startIdx = if rhs.kind == nnkBracketExpr: 1 else: 0
      for i in startIdx ..< rhs.len:
        let param = rhs[i]
        if param.kind == nnkExprColonExpr:
          gp.add(newIdentDefs(param[0], param[1], newEmptyNode()))
        else:
          gp.add(newIdentDefs(param, newEmptyNode(), newEmptyNode()))
      generics = gp
    else:
      error("Invalid class syntax after '*'.", head)

  else:
    error("Invalid class syntax.", head)

  return (name, parent, generics)


proc newClassDescription*(macroName: string, head: NimNode,
                          body: NimNode): ClassDescription =
  ## Parse the macro head and body into a `ClassDescription`.
  let classDef = ClassDescription()
  classDef.macroName = macroName
  classDef.inputBody = body
  classDef.prefix = newNimNode(nnkStmtList)
  classDef.fwdDecl = newNimNode(nnkStmtList)
  classDef.body = newNimNode(nnkStmtList)
  classDef.suffix = newNimNode(nnkStmtList)
  classDef.genericParams = newEmptyNode()

  let (name, parent, generics) = parseClassHead(head)
  classDef.name = name
  classDef.parentName = parent
  classDef.genericParams = generics

  # Run hooks through all stages up to code generation
  for hook in classCompilerHooks: hook(ClassPreload, classDef)
  for hook in classCompilerHooks: hook(ClassGatherDefinitions, classDef)
  for hook in classCompilerHooks: hook(ClassAddExtraDefinitions, classDef)
  for hook in classCompilerHooks: hook(ClassModifyDefinitions, classDef)

  # Create the ref object type definition with inheritance:
  #   type Name* = ref object of Parent
  let className = classDef.name
  let parentName = classDef.effectiveParent()

  classDef.output = quote do:
    type `className`* = ref object of `parentName`

  # Inject generic params into the TypeDef node (slot [1])
  if classDef.isGeneric:
    classDef.output[0][1] = copyNimTree(classDef.genericParams)

  # Run remaining stages
  for hook in classCompilerHooks: hook(ClassGenerateCode, classDef)
  for hook in classCompilerHooks: hook(ClassFinalize, classDef)

  return classDef

#[ Compile – assemble the final NimNode tree ]#

proc compile*(this: ClassDescription): NimNode =
  ## Assemble all generated code and return it as a single nnkStmtList.
  # Cache this class
  classCompilerCache.add(this)

  var code = newStmtList()
  code.add(this.prefix)
  code.add(this.output)
  code.add(this.fwdDecl)
  code.add(this.body)
  code.add(this.suffix)

  # Debug output
  const debugclasses {.strdefine.} = ""
  let debugNames = debugclasses.split(",")
  if debugclasses == "true" or debugNames.contains($this.name):
    echo "\n=== Created " & this.macroName & " " & $this.name
    for hook in classCompilerHooks:
      hook(ClassDebugEcho, this)
    echo ""
    echo code.repr
    echo ""

  return code
