#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/globalarrays.nim
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

import globalarrays/[gabase]
import globalarrays/[gampi]
import globalarrays/[gatypes]
import globalarrays/[gawrap]

import utils/[commandline]

#[ exports ]#

export gatypes # GlobalArray type/functionality

#[ full GlobalArrays initialization ]#

template initGlobalArrays*: untyped = 
  let argc = cargc()
  let argv = cargv(argc)
  
  initMPI(addr argc, addr argv)
  initGA()

template finalizeGlobalArrays*: untyped = 
  finalizeGA()
  finalizeMPI()

#[ fence ]#

template localFence*(body: untyped): untyped =
  GA_Init_fence()
  body
  GA_Fence()

#[ global operations ]#

proc gaGlobalSumInt32*(localSum: ptr int32): int32 =
  var value = localSum[]
  GA_Igop(addr value, 1, "+")
  GA_Sync()
  return int32(value)

proc gaGlobalSumInt64*(localSum: ptr int64): int64 =
  var value = localSum[]
  GA_Lgop(addr value, 1, "+")
  GA_Sync()
  return int64(value)

proc gaGlobalSumFloat32*(localSum: ptr float32): float32 =
  # Convert to double, sum, then convert back
  var value = localSum[]
  GA_Fgop(addr value, 1, "+")
  GA_Sync()
  return float32(value)

proc gaGlobalSumFloat64*(localSum: ptr float64): float64 =
  var value = localSum[]
  GA_Dgop(addr value, 1, "+")
  GA_Sync()
  return value

#[ misc ]#

template myRank*: int = GA_Nodeid()

template numRanks*: int = GA_Nnodes()

template globalBarrier*: untyped = GA_Sync()
