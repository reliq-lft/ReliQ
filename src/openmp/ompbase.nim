#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/openmp/ompbase.nim
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

## OpenMP Backend Base Utilities
##
## Provides OpenMP initialization, helper functions, and QEX-style parallel
## primitives for parallel CPU execution.
##
## The core threading model follows QEX (Quantum Expressions):
## - ``ompBlock("parallel")`` emits ``#pragma omp parallel`` + Nim ``block:``
##   which compiles to valid OpenMP ``#pragma omp parallel { ... }`` in C
## - Manual work division via ``threadDivideLow``/``threadDivideHigh``
##   instead of ``#pragma omp parallel for`` (avoids goto issues)
## - ``threadSum`` for reductions via padded global arrays + barriers
## - ``ompBarrier`` for thread synchronization

import std/macros

{.passC: "-fopenmp".}
{.passL: "-fopenmp".}

# OpenMP runtime function bindings
proc omp_get_max_threads*(): cint {.importc, header: "<omp.h>".}
proc omp_get_thread_num*(): cint {.importc, header: "<omp.h>".}
proc omp_get_num_threads*(): cint {.importc, header: "<omp.h>".}

# Global initialization state
var ompInitialized {.global.}: bool = false

proc initOpenMP*() =
  ## Initialize OpenMP runtime
  ## Respects OMP_NUM_THREADS environment variable if set
  if not ompInitialized:
    let numThreads = omp_get_max_threads()
    echo "OpenMP: Initialized with ", numThreads, " threads"
    ompInitialized = true

proc getNumThreads*(): int {.inline.} =
  ## Get current number of OpenMP threads
  omp_get_max_threads().int

proc getThreadId*(): int {.inline.} =
  ## Get current thread ID (0 to numThreads-1)
  omp_get_thread_num().int

#[ ============================================================================
   QEX-Style OpenMP Primitives
   ============================================================================ ]#

template ompPragma*(p: string) =
  ## Emit a raw ``#pragma omp <p>`` directive.
  ## Use for barriers, flushes, etc.
  ## The leading ``\n`` ensures the ``#pragma`` starts at the beginning of
  ## a C source line (required by the C preprocessor).
  {.emit: ["\n#pragma omp ", p].}

template ompBlock*(p: string; body: untyped) =
  ## Emit ``#pragma omp <p>`` followed by a Nim ``block:``.
  ## Since Nim's ``block:`` compiles to ``{ ... }`` in C, the C compiler
  ## sees ``#pragma omp parallel { ... }`` which is a valid structured block.
  ##
  ## This is the core of the QEX approach: the pragma applies to the
  ## immediately-following ``{`` from the Nim block, and all code inside
  ## (including any Nim-generated gotos) is valid because they stay
  ## within the structured block.
  {.emit: ["\n#pragma omp ", p].}
  block:
    body

template ompBarrier*() =
  ## ``#pragma omp barrier`` — synchronize all threads.
  ompPragma("barrier")

template ompCritical*(body: untyped) =
  ## ``#pragma omp critical { ... }``
  ompBlock("critical"):
    body

template ompMaster*(body: untyped) =
  ## ``#pragma omp master { ... }``
  ompBlock("master"):
    body

template ompSingle*(body: untyped) =
  ## ``#pragma omp single { ... }``
  ompBlock("single"):
    body

template ompParallel*(body: untyped) =
  ## Execute a block in parallel context.
  ## Initializes OpenMP and executes the body.
  ## NOTE: This does NOT emit ``#pragma omp parallel`` — it is the
  ## top-level initialization wrapper. The actual OpenMP parallel regions
  ## are created inside the ``each``/``reduce`` macros using ``ompBlock("parallel")``.
  initOpenMP()
  body

#[ ============================================================================
   Manual Work Division (replaces #pragma omp parallel for)
   ============================================================================ ]#

template threadDivideLow*(lo, hi: int; threadId, nThreads: int): int =
  ## Compute this thread's sub-range start for dividing [lo, hi) across threads.
  ## Thread ``threadId`` handles [threadDivideLow, threadDivideHigh).
  lo + (threadId * (hi - lo)) div nThreads

template threadDivideHigh*(lo, hi: int; threadId, nThreads: int): int =
  ## Compute this thread's sub-range end for dividing [lo, hi) across threads.
  lo + ((threadId + 1) * (hi - lo)) div nThreads

#[ ============================================================================
   threadSum — QEX-Style Parallel Reduction
   ============================================================================ ]#

# Maximum number of threads supported for the padded reduction arrays.
# Padding by 64 doubles (512 bytes) per slot avoids false sharing between
# cache lines. Uses {.global.} arrays (C static) to avoid heap allocation.
const ThreadSumMaxThreads* = 512
const ThreadSumPadding* = 64  # 64 doubles = 512 bytes ≥ cache line

macro threadSum*(args: varargs[untyped]): untyped =
  ## Sum variables across all OpenMP threads using padded global arrays.
  ##
  ## Must be called inside ``ompParallel`` / ``ompBlock("parallel")``.
  ## Each thread's value is written to a padded slot, a barrier synchronizes,
  ## then each thread reads the total (so all threads see the same result).
  ##
  ## Usage (inside ompParallel):
  ##   var mySum = 0.0
  ##   # ... each thread computes partial mySum ...
  ##   threadSum(mySum)
  ##   # now mySum == total across all threads
  ##
  ## Multiple variables can be summed at once:
  ##   threadSum(sumA, sumB)
  
  result = newStmtList()
  var sumStmts = newStmtList()
  
  let arrSize = newLit(ThreadSumMaxThreads * ThreadSumPadding)
  let padLit = newLit(ThreadSumPadding)
  let getThreadNumProc = bindSym("omp_get_thread_num")
  let getNumThreadsProc = bindSym("omp_get_num_threads")
  
  for i in 0..<args.len:
    let ai = args[i]
    let gi = genSym(nskVar, "tsArr" & $i)
    let tidSym = genSym(nskLet, "tsTid")
    let ntSym = genSym(nskLet, "tsNt")
    
    # Declare padded global array and store this thread's value
    result.add quote do:
      var `gi` {.global.}: array[`arrSize`, float64]
      let `tidSym` = `getThreadNumProc`().int
      let `ntSym` = `getNumThreadsProc`().int
      `gi`[`padLit` * `tidSym`] = float64(`ai`)
    
    # Build the summation loop (runs after barrier)
    let loopIdx = genSym(nskForVar, "tsI")
    sumStmts.add quote do:
      `ai` = type(`ai`)(`gi`[0])
      for `loopIdx` in 1..<`ntSym`:
        `ai` = type(`ai`)(float64(`ai`) + `gi`[`padLit` * `loopIdx`])
  
  # Barrier, sum, barrier
  result.add quote do:
    ompBarrier()
    `sumStmts`
    ompBarrier()
  
  result = newNimNode(nnkBlockStmt).add(newEmptyNode(), result)

when isMainModule:
  initOpenMP()
  echo "Max threads: ", getNumThreads()
