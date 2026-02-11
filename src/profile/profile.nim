#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/profile/profile.nim
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

import std/[times, strformat]
import globalarrays/[gawrap, gabase]

const ProfileMode* {.intdefine.} = 0

type
  Profiler* = object
    epochTime*: float
    localStartTime*: float
    localFlops*: int
    totalFlops*: int
    totalEventName*: string
    localEventName*: string

template initProfiler*(name: string): untyped =
  assert gaIsLive, "GlobalArrays must be initialized before creating a PerformanceProfiler"
  when ProfileMode == 1:
    var profiler {.inject.}: Profiler
    if GA_Nodeid() == 0:
        profiler = Profiler(
          totalEventName: name,
          epochTime: times.cpuTime(), 
          totalFlops: 0,
          localFlops: 0
        )
        echo name & ": "

template tic*(eventName: string): untyped =
  when ProfileMode == 1:
    if GA_Nodeid() == 0:
      profiler.localEventName = eventName
      profiler.localStartTime = times.cpuTime()
      profiler.localFlops = 0

template toc*(): untyped =
  when ProfileMode == 1:
    if GA_Nodeid() == 0:
      let dt {.inject.} = times.cpuTime() - profiler.localStartTime
      profiler.totalFlops += profiler.localFlops
      if profiler.localFlops == 0:
        echo fmt"  {profiler.localEventName}: {dt:.6f}s"
      else:
        let gflops {.inject.} = float64(profiler.localFlops) / 1e9
        echo fmt"  {profiler.localEventName}: {dt:.6f}s GFLOP: {gflops:.4f} GFLOP/s: {gflops/dt:.4f}"

proc matMulFLOP*(nc: int): int =
  nc*nc*(8*nc - 2)

proc matAddSubFLOP*(nc: int): int =
  2*nc*nc

proc traceFLOP*(nc: int): int =
  2*(nc - 1)

proc adjointFLOP*(nc: int): int =
  nc*nc

proc addFLOPImpl*(profiler: var Profiler, flops: int) {.inline.} =
  ## Host-side FLOP accumulator implementation.
  profiler.localFlops += flops

template addFLOP*(flops: int): untyped =
  ## FLOP accumulator. Inside `each` loops the transpiler intercepts the
  ## addFLOPImpl call: it is skipped in kernel codegen and emitted as
  ## host-side accumulation multiplied by the number of sites.
  ## Outside `each` loops it runs directly on the host.
  when ProfileMode == 1:
    addFLOPImpl(profiler, flops)

template finalizeProfiler*(): untyped =
  when ProfileMode == 1:
    if GA_Nodeid() == 0:
      let dt {.inject.} = times.cpuTime() - profiler.epochTime
      let gflops {.inject.} = float64(profiler.totalFlops) / 1e9
      echo fmt"  Total: {dt:.6f}s GLOP: {gflops:.4f} GFLOP/s: {gflops/dt:.4f}"