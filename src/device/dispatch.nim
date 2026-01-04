#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/device/dispatch.nim
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

import std/[macros, os, strutils, cpuinfo]

import malebolgia

import platforms

nvidia: import cuda/[cudawrap]
amd: import hip/[hipwrap]
cpu: import simd/[simdtypes]

var numThreads*: int = 1
let envThreads = getEnv("OMP_NUM_THREADS")
if envThreads.len > 0:
  try: numThreads = parseInt(envThreads)
  except ValueError: numThreads = countProcessors()

macro each*(x: ForLoopStmt): untyped =
  ## Threaded + vectorized for loop consturct
  ## 
  ## Turns a `for` loop of the form:
  ## ```
  ## for i in every 0..10: <body>
  ## ```
  ## into a threaded + vectorized loop on CPU/GPU. Behaves like `Grid`'s
  ## `accelerator_for` construct. 
  let (idnt, call, body) = (x[0], x[1], x[2])
  let (itr, rng) = (call[1], call[1][0])
  let (lo, hi) = (itr[1], itr[^1])
  
  if $rng != "..<":
    error("Only half-open ranges with '..<' are supported in 'all' loops")
  
  result = quote do:
    nvidia: discard
    amd: discard
    cpu:
      let totalWork = `hi` - `lo`
      let baseChunkSize = (totalWork + numThreads - 1) div numThreads
      let chunkSize = ((baseChunkSize + vectorWidth - 1) div vectorWidth) * vectorWidth

      proc workerAll(threadId: int) =
        let startIdx = `lo` + threadId * chunkSize
        let endIdx = min(`lo` + (threadId + 1) * chunkSize, `hi`)
        
        var `idnt` = startIdx
        while `idnt` < endIdx:
          `body`
          `idnt` += vectorWidth
      
      var m = createMaster()
      m.awaitAll:
        for threadId in 0..<numThreads:
          m.spawn workerAll(threadId)

macro all*(x: ForLoopStmt): untyped =
  ## Threaded for loop construct
  ## 
  ## Turns a `for` loop of the form:
  ## ```
  ## for i in every 0..10: <body>
  ## ```
  ## into a simple loop that iterates over the specified range.
  let (idnt, call, body) = (x[0], x[1], x[2])
  let (itr, rng) = (call[1], call[1][0])
  let (lo, hi) = (itr[1], itr[^1])
  
  if $rng != "..<":
    error("Only half-open ranges with '..<' are supported in 'every' loops")
  
  result = quote do:
    let totalWork = `hi` - `lo`
    let baseChunkSize = (totalWork + numThreads - 1) div numThreads

    proc workerEvery(threadId: int) =
      let startIdx = `lo` + threadId * baseChunkSize
      let endIdx = min(`lo` + (threadId + 1) * baseChunkSize, `hi`)
      
      for `idnt` in startIdx..<endIdx:
        `body`
      
    var m = createMaster()
    m.awaitAll:
      for threadId in 0..<numThreads:
        m.spawn workerEvery(threadId)

when isMainModule:
  for n in each 0..<80:
    echo n
  
  for n in all 0..<80:
    echo n
