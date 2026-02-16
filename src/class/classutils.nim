#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/classes/classutils.nim
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

## Extra NimNode processing utilities for the class macro system.
## Mirrors ``recordutils.nim`` for classes.
##
## Provides helpers for traversing DSL statement lists, extracting variable
## names, iterating over formal parameters, and unbinding symbols.

import std/[macros]

proc traverseClassStatementList*(item: NimNode,
    callback: proc(index: int, parent: NimNode, node: NimNode)) =
  ## Recursively traverse an nnkStmtList, calling `callback` for every
  ## non-nnkStmtList leaf node.
  if item.kind != nnkStmtList:
    raiseAssert("Only nnkStmtList nodes can be passed to traverseClassStatementList()")

  for idx, childNode in item:
    if childNode.kind == nnkStmtList:
      traverseClassStatementList(childNode, callback)
    else:
      callback(idx, item, childNode)

proc variableName*(this: NimNode): NimNode =
  ## Extract the variable name (nnkIdent or nnkSym) from an nnkIdentDefs node.
  if this.kind == nnkIdent or this.kind == nnkSym:
    return this

  if this.kind != nnkIdentDefs:
    error("Expected an identifier but got " & $this.kind & " instead.", this)

  var item = this[0]

  # Strip postfix (export marker)
  if item.kind == nnkPostfix:
    item = item[1]

  # Strip pragma expression
  if item.kind == nnkPragmaExpr:
    item = item[0]

  if item.kind == nnkIdent or item.kind == nnkSym:
    return item
  else:
    error("Unexpected identifier in " & $this.kind & " node.", this)

proc traverseParams*(item: NimNode,
    callback: proc(idx: int, nameNode: NimNode, typeNode: NimNode,
                   identDef: NimNode, identDefIdx: int)) =
  ## Iterate over a routine's formal parameters, calling `callback` for each.
  var paramsNode: NimNode
  if item.kind == nnkFormalParams:
    paramsNode = item
  elif item.kind in RoutineNodes:
    paramsNode = item.params
  else:
    error("Expected a routine node but got a " & $item.kind & " instead.", item)

  var paramIdx = 0
  for i, identDef in paramsNode:
    # First entry is the return type
    if i == 0:
      continue

    let typeNode = identDef[identDef.len() - 2]
    for x, paramIdent in identDef:
      if x >= identDef.len() - 2:
        continue
      callback(paramIdx, paramIdent, typeNode, identDef, x)
      paramIdx += 1

proc unbindAllSym*(node: NimNode) =
  ## Convert all nnkSym nodes back into unbound nnkIdent nodes.
  for i, child in node:
    if child.kind == nnkSym:
      node[i] = ident($child)
    unbindAllSym(child)
