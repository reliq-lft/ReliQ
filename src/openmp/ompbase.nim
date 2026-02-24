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
## =============================
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
##
## Adapted from Quantum EXpressions:
## 
## MIT License
##  
## Copyright (c) 2017 James C. Osborn
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
import std/[strutils]

import utils/[private]

{.passC: "-fopenmp".}
{.passL: "-fopenmp".}

type
  ThreadShare* = object
    p*: pointer
    counter*: int
    extra*: int
  
  ThreadLocal* = object
    threadNum*: int
    numThreads*: int
    share*: ptr UncheckedArray[ThreadShare]
  
var myThread* {.threadvar.}: int
var numThreads* {.threadvar.}: int
var threadLocals* {.threadvar.}: ThreadLocal
var inited = false
var ts: pointer = nil
var nts = 0

var ompInitialized* {.global.}: bool = false

alwaysInlinePragma()

template ensureMP*: untyped =
  ## Initialize OpenMP runtime
  ## Respects OMP_NUM_THREADS environment variable if set
  if not ompInitialized:
    let maxThreads = omp_get_max_threads()
    ompInitialized = true

proc omp_get_max_threads(): cint {.importc, header: "<omp.h>".}
proc omp_get_thread_num(): cint {.importc, header: "<omp.h>".}
proc omp_get_num_threads(): cint {.importc, header: "<omp.h>".}

proc allocThreadShare {.alwaysinline.} =
  if numThreads > nts and myThread == 0:
    if ts == nil: ts = allocShared(numThreads*sizeof(ThreadShare))
    else: ts = reallocShared(ts, numThreads*sizeof(ThreadShare))
    nts = numThreads

proc initThreadLocals =
  bind ts
  threadLocals.threadNum = myThread
  threadLocals.numThreads = numThreads
  threadLocals.share = cast[ptr UncheckedArray[ThreadShare]](ts)
  threadLocals.share[myThread].p = nil
  threadLocals.share[myThread].counter = 0

proc initThreads =
  inited = true
  myThread = 0
  numThreads = 1
  allocThreadShare()
  initThreadLocals()

proc checkThreadInit =
  if not inited: initThreads()

macro emitStackTraceX(x: typed): untyped =
  template est(x) = 
    {.emit: "// instantiationInfo: " & x.}
  let ii = x.repr.replace("\n","")
  return getAst(est(ii))

template emitStackTrace: untyped =
  emitStackTraceX(instantiationInfo(-1))
  emitStackTraceX(instantiationInfo(-2))
  emitStackTraceX(instantiationInfo(-3))

template ompPragma(p: string) =
  ## Emit a raw ``#pragma omp <p>`` directive.
  ## Use for barriers, flushes, etc.
  ## The leading ``\n`` ensures the ``#pragma`` starts at the beginning of
  ## a C source line (required by the C preprocessor).
  {.emit: ["_Pragma(\"omp ", p, "\")"].}

template ompBlock(p: string; body: untyped) =
  ## Emit ``#pragma omp <p>`` followed by a Nim ``block:``.
  ## Since Nim's ``block:`` compiles to ``{ ... }`` in C, the C compiler
  ## sees ``#pragma omp parallel { ... }`` which is a valid structured block.
  ##
  ## This is the core of the QEX approach: the pragma applies to the
  ## immediately-following ``{`` from the Nim block, and all code inside
  ## (including any Nim-generated gotos) is valid because they stay
  ## within the structured block.
  ompPragma(p)
  block: body

template barrier* =
  ## ``#pragma omp barrier`` — synchronize all threads.
  ompPragma("barrier")

template flush* =
  ## ``#pragma omp flush`` — ensure memory visibility across threads.
  ompPragma("flush")

template main*(body: untyped) =
  ## Main thread
  ompBlock("master"): body

template threads*(body: untyped) =
  checkThreadInit()
  doAssert numThreads == 1

  proc execThreads {.gensym.} =
    ompBlock("parallel"):
      when(declared(setupForeignThreadGc)):
        if omp_get_thread_num() != 0: setupForeignThreadGc()      
      emitStackTrace()

      myThread = omp_get_thread_num()
      numThreads = omp_get_max_threads()

      allocThreadShare()
      barrier()
      initThreadLocals()
      barrier()
      body
      barrier()
  
  execThreads()
  myThread = 0
  numThreads = 1
  initThreadLocals()


when isMainModule:
  import std/[unittest]

  suite "OpenMP Base Utilities":
    test "Thread initialization sets correct state":
      inited = false
      myThread = -1
      numThreads = -1
      nts = 0
      ts = nil
      initThreads()
      check:
        inited == true
        myThread == 0
        numThreads == 1
        nts == 1
        ts != nil

    test "checkThreadInit calls initThreads if needed":
      inited = false
      myThread = -1
      numThreads = -1
      nts = 0
      ts = nil
      checkThreadInit()
      check:
        inited == true
        myThread == 0
        numThreads == 1

    test "ThreadLocal fields are set correctly":
      inited = false
      initThreads()
      check:
        threadLocals.threadNum == myThread
        threadLocals.numThreads == numThreads
        threadLocals.share != nil

    test "allocThreadShare allocates and reallocates":
      inited = false
      myThread = 0
      numThreads = 2
      nts = 0
      ts = nil
      allocThreadShare()
      let firstPtr = ts
      check:
        nts == 2
        ts != nil
      numThreads = 4
      allocThreadShare()
      check:
        nts == 4
        ts != nil
        ts != firstPtr

    test "initThreadLocals initializes share array":
      inited = false
      myThread = 0
      numThreads = 2
      nts = 0
      ts = nil
      allocThreadShare()
      initThreadLocals()
      check:
        threadLocals.share != nil
        threadLocals.share[myThread].p == nil
        threadLocals.share[myThread].counter == 0

    test "barrier and ompPragma macros compile":
      # This test just ensures the macros can be called without error
      threads:
        barrier()
        flush()
      check true

    test "threads template runs and sets thread state":
      var ran = false
      threads:
        ran = true
        check:
          myThread == omp_get_thread_num()
          numThreads == omp_get_max_threads()
      check ran
