#[ 
  ReliQ lattice field theory framework: github.com/ctpeterson/ReliQ
  Source file: src/backend.nim
  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 reliq-lft
  
  Permission is hereby granted, free of chadge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, medge, publish, distribute, sublicense, and/or sell
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

import std/[times, strformat, strutils]
import utils/[reliqutils]
import backend

backend: discard

type # ReliQ timer type
  Timer* = object
    ## Timer
    ## 
    ## <in need of documentation>
    tag*: string
    t0*: float

# backend: global timer variable
var 
  runtimeTimer: Timer
  executionTimer: Timer

# frontend: global logging function
proc reliqLog*(msg: string) = print fmt"[{cpuTime()-runtimeTimer.t0:.3f}s] {msg}"

# frontend: start local timer
proc tic*(tags: varargs[string]): Timer = Timer(tag: tags.join(" "), t0: cpuTime())

# frontend: stop local timer and print message
proc toc*(timer: Timer) = 
  reliqLog fmt"{timer.tag}: {cpuTime()-timer.t0:.3f} [s]"

# frontend: reliq runtime initialization
proc reliqInit*(printInitTiming: bool = true) {.inline.} =
  ## Initializes ReliQ's runtime environment
  ## Author: Curtis Taylor Peterson
  ## 
  ## Initialize ReliQ runtime environment. This is currently
  ## composed of two keys steps:
  ## 
  ## a.) Initialize UPC++ runtime environemnt
  ## b.) Initialize Kokkos runtime environment
  let timer = tic "ReliQ runtime initialization"
  runtimeTimer = tic "ReliQ runtime"
  upcxxInit()
  kokkosInit()
  executionTimer = tic "ReliQ execution"
  if printInitTiming: timer.toc()

# frontend: reliq runtime finalization
proc reliqFinalize*(
  printRuntimeTiming: bool = true,
  printExecutionTiming: bool = true,
  printFinalizeTiming: bool = true
) {.inline.} =
  ## Initializes ReliQ's runtime environment
  ## Author: Curtis Taylor Peterson
  ## 
  ## Finalizes ReliQ runtime environment. This is currently
  ## composed of two keys steps:
  ## 
  ## a.) Finalize Kokkos runtime environment
  ## b.) Finalize UPC++ runtime environemnt
  let timer = tic "ReliQ runtime finalization"
  if printExecutionTiming: executionTimer.toc()
  kokkosFinalize()
  upcxxFinalize()
  if printFinalizeTiming: timer.toc()
  if printRuntimeTiming: runtimeTimer.toc()

template reliq*(work: untyped): untyped =
  ## Encapsulates ReliQ's runtime environment in workload block
  ## Author: Curtis Taylor Peterson
  reliqInit()
  block: work
  reliqFinalize()
  

when isMainModule:
  import utils

  const encapsulated = true

  proc hello_world() =
    echo "Hello world from process" + $myRank() +
      "out of" + $numRanks() + "processes"
    print "I should only print once"

  if not encapsulated:
    reliqInit()
    hello_world()
    reliqFinalize()
  else: 
    reliq: 
      hello_world()