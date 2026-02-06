#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/openmp/ompdisp.nim
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

## Each Macro for TensorFieldView
##
## This module provides the `each` iterator for parallelized loops
## on TensorFieldView objects. Uses OpenMP CPU thread parallelization.
##
## Usage:
##   for n in each 0..<view.numSites():
##     viewC[n] = viewA[n] + viewB[n]
##
## The `each` macro is exported by parallel.nim for general use.
## For LocalTensorField operations, see `all` in omplocal.nim.

import std/macros

import ompbase
export ompbase

{.passC: "-fopenmp".}
{.passL: "-fopenmp".}

{.emit: """
#include <omp.h>
""".}

import ./ompwrap

#[ ============================================================================
   Echo Statement Detection
   ============================================================================ ]#

proc hasEchoStatement(n: NimNode): bool =
  ## Check if body contains echo/debugEcho (requires serial execution)
  case n.kind
  of nnkCall:
    if n[0].kind == nnkSym:
      let name = n[0].strVal
      if name in ["echo", "debugEcho"]:
        return true
    for child in n:
      if hasEchoStatement(child):
        return true
  of nnkCommand:
    if n[0].kind == nnkIdent and n[0].strVal == "echo":
      return true
    if n[0].kind == nnkSym and n[0].strVal in ["echo", "debugEcho"]:
      return true
    for child in n:
      if hasEchoStatement(child):
        return true
  else:
    for child in n:
      if hasEchoStatement(child):
        return true
  return false

#[ ============================================================================
   Main Each Macro
   ============================================================================ ]#

macro each*(forLoop: ForLoopStmt): untyped =
  ## OpenMP parallel each loop for TensorFieldView (CPU threads)
  ##
  ## Parallelizes the loop using OpenMP with static scheduling.
  ## Each iteration is distributed across available CPU threads.
  ##
  ## Usage:
  ##   for n in each 0..<view.numSites():
  ##     viewC[n] = viewA[n] + viewB[n]
  
  let loopVar = forLoop[0]
  let loopRangeNode = forLoop[1][1]  # Skip 'each' wrapper
  let body = forLoop[2]
  
  # Check for echo - needs serial execution
  let needsSerial = hasEchoStatement(body)
  
  if loopRangeNode.kind == nnkInfix and loopRangeNode[0].strVal == "..<":
    let startExpr = loopRangeNode[1]
    let endExpr = loopRangeNode[2]
    
    if needsSerial:
      result = quote do:
        block:
          for `loopVar` in `startExpr`..<`endExpr`:
            `body`
    else:
      # Use ompParallelFor for CPU thread parallelization
      result = quote do:
        block:
          proc loopBody(idx: int64, ctx: pointer) {.cdecl.} =
            let `loopVar` = int(idx)
            `body`
          ompParallelFor(int64(`startExpr`), int64(`endExpr`), loopBody, nil)
  else:
    result = quote do:
      block:
        for `loopVar` in `loopRangeNode`:
          `body`

when isMainModule:
  import ../tensor/sitetensor
  
  initOpenMP()
  echo "OpenMP dispatch module loaded"
  echo "Max threads: ", getNumThreads()
