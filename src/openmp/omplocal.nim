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
## This module provides the `all` iterator for CPU-only parallelized loops
## on LocalTensorField objects. Unlike TensorFieldView which can use GPU
## acceleration, LocalTensorField operations are always on host memory.
##
## Usage:
##   for n in all 0..<local.numSites():
##     localC[n] = localA.getSite(n) + localB.getSite(n)
##
## The `all` macro is exported by parallel.nim for general use.

import std/macros

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

macro all*(forLoop: ForLoopStmt): untyped =
  ## OpenMP parallel all loop for LocalTensorField
  ##
  ## This macro provides the `all` iterator for CPU-only parallelized loops
  ## on LocalTensorField objects. Unlike TensorFieldView which can use GPU
  ## acceleration, LocalTensorField operations are always on host memory.
  ##
  ## Usage:
  ##   for n in all 0..<local.numSites():
  ##     localC[n] = localA.getSite(n) + localB.getSite(n)
  ##
  ## Note: Uses getSite() to access sites - returns LocalSiteProxy
  
  # Extract loop components
  let loopVar = forLoop[0]
  let loopRangeNode = forLoop[1][1]  # Skip 'all' wrapper
  let body = forLoop[2]
  
  # Check if we need serial execution (echo statements)
  let needsSerial = hasEchoStatement(body)
  
  if needsSerial:
    # Serial fallback for debugging with echo
    result = quote do:
      block:
        let rangeVal = `loopRangeNode`
        for `loopVar` in rangeVal:
          `body`
    return result
  
  # Execute the body directly - operators handle memory access
  # TODO: Add OpenMP parallelization with proper pragma placement
  if loopRangeNode.kind == nnkInfix and loopRangeNode[0].strVal == "..<":
    let startExpr = loopRangeNode[1]
    let endExpr = loopRangeNode[2]
    result = quote do:
      block:
        let startVal = `startExpr`
        let endVal = `endExpr`
        for `loopVar` in startVal..<endVal:
          `body`
  else:
    result = quote do:
      block:
        let rangeVal = `loopRangeNode`
        for `loopVar` in rangeVal:
          `body`
