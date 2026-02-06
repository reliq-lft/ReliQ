#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/openmp/omplocal.nim
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

## All Macro for LocalTensorField CPU-only loops
##
## This module provides the `all` iterator for parallelized loops
## on LocalTensorField objects. LocalTensorField operations run on
## host memory and use CPU thread parallelization.
##
## Usage:
##   for n in all 0..<local.numSites():
##     localC[n] = localA.getSite(n) + localB.getSite(n)
##
## The `all` macro is exported by parallel.nim for general use.
## For the corresponding GPU iterator, see `each` in ompdisp.nim.

import std/macros
import ./ompbase

#[ ============================================================================
   Helper: Check for echo statements (needs serial execution)
   ============================================================================ ]#

proc hasEchoStatement*(node: NimNode): bool =
  ## Check if the AST contains echo statements
  if node.kind == nnkCall:
    if node[0].kind == nnkIdent and node[0].strVal == "echo":
      return true
    if node[0].kind == nnkSym and node[0].strVal == "echo":
      return true
  
  for child in node:
    if hasEchoStatement(child):
      return true
  return false

#[ ============================================================================
   All Macro for LocalTensorField CPU-only loops
   ============================================================================ ]#

{.passC: "-fopenmp".}
{.passL: "-fopenmp".}

# Emit the OpenMP header at file level
{.emit: """
#include <omp.h>
""".}

# Import the C wrapper for parallel loops
import ./ompwrap

macro all*(forLoop: ForLoopStmt): untyped =
  ## OpenMP parallel all loop for LocalTensorField (CPU threads)
  ##
  ## Uses ompParallelFor template for actual parallelization.
  ##
  ## Usage:
  ##   for n in all 0..<local.numSites():
  ##     localC[n] = localA.getSite(n) + localB.getSite(n)
  
  let loopVar = forLoop[0]
  let loopRangeNode = forLoop[1][1]  # Skip 'all' wrapper
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
