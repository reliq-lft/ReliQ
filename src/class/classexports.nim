#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/classes/classexports.nim
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

## Auto-export all class fields and methods.
##
## Mirrors ``recordexports.nim`` — ensures all fields and methods
## declared in a class body are publicly visible.

{.used.}

import std/[macros]

import classinternal
import classmethods
import classvar

proc exportAll(classDef: ClassDescription) =
  # Export all fields
  for varDef in classDef.vars.definitions:
    if varDef.definition[0].kind == nnkPostfix:
      continue
    let nameNode = varDef.definition[0]
    varDef.definition[0] = newNimNode(nnkPostfix, nameNode)
    varDef.definition[0].add(ident"*")
    varDef.definition[0].add(nameNode)

  # Export all methods
  for methodDef in classDef.methods.definitions:
    if methodDef.definition[0].kind == nnkPostfix:
      continue
    let nameNode = methodDef.definition[0]
    methodDef.definition[0] = newNimNode(nnkPostfix, nameNode)
    methodDef.definition[0].add(ident"*")
    methodDef.definition[0].add(nameNode)

static:
  classCompilerHooks.add(proc(stage: ClassCompilerStage, classDef: ClassDescription) =
    if stage == ClassModifyDefinitions: exportAll(classDef)
  )
