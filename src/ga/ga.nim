#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/globalarrays/globalarrays.nim
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

import std/[cmdline]
import std/[os]
import std/[macros]

import gampi

include gatypes

const
  GADATATYPE = 1001
  GASTACKSIZE = clong(10_000_000)
  GAHEAPSIZE = clong(10_000_000)

var myRank* = 0
var numRanks* = 0

#[ get c arguments for GA initialization ]#

proc cargc: cint = cint(paramCount() + 1)

proc cargv(argc: cint): cstringArray =
  var argv = newSeq[string](argc)
  argv[0] = getAppFilename()
  for idx in 1..<argv.len: 
    argv[idx] = paramStr(idx)
  return allocCStringArray(argv)

#[ single-process print statement wrapper ]#

macro reliqPrint(args: varargs[untyped]): untyped =
  var statements: seq[NimNode]
  for iarg, varg in args:
    if iarg < args.len - 1:
      statements.add newCall("write", ident"stdout", newCall(ident"$", varg))
      statements.add newCall("write", ident"stdout", newLit(" "))
    else: statements.add newCall("writeLine", ident"stdout", newCall(ident"$", varg))
  return newStmtList(statements)

template print*(args: varargs[untyped]): untyped =
  if myRank == 0: reliqPrint(args)

#[ GA initialization/finalization wrappers ]#

template gaParallel*(body: untyped): untyped =
  var argc = cargc()
  var argv = cargv(argc)

  initMPI(addr argc, addr argv)
  initGA()
  discard initMA(GADATATYPE, GASTACKSIZE, GAHEAPSIZE)
  
  myRank = GA_Nodeid()
  numRanks = GA_Nnodes()

  gaActive = true
  block: body
  gaActive = false
  
  finalizeGA()
  finalizeMPI()

# unit testing
  
when isMainModule:
  gaParallel:
    suite "GlobalArrays tests":
      test "GA initialization and finalization" :
        check myRank >= 0
        check gaActive

      for width in 0..2:
        test "GA creation":
          let physGrid = [8, 8, 8, 16]
          let mpiGrid = [-1, -1, -1, -1] # auto-decompose across all ranks
          let ghostGrid = [width, width, width, width]

          var gaSingleInt = newGlobalArray(physGrid, mpiGrid, ghostGrid): int32
          var gaDoubleInt = newGlobalArray(physGrid, mpiGrid, ghostGrid): int64
          var gaSingleFloat = newGlobalArray(physGrid, mpiGrid, ghostGrid): float32
          var gaDoubleFloat = newGlobalArray(physGrid, mpiGrid, ghostGrid): float64

          # check initialization

          check gaSingleInt.isInitialized()
          check gaDoubleInt.isInitialized()
          check gaSingleFloat.isInitialized()
          check gaDoubleFloat.isInitialized()
          
          # check ghost exchange

          if width != 0:
            gaSingleInt.exchange(update_corners = true)
            gaDoubleInt.exchange(update_corners = true)
            gaSingleFloat.exchange(update_corners = true)
            gaDoubleFloat.exchange(update_corners = true)

            gaSingleInt.exchange(update_corners = false)
            gaDoubleInt.exchange(update_corners = false)
            gaSingleFloat.exchange(update_corners = false)
            gaDoubleFloat.exchange(update_corners = false)

            check true

          # check local access

          discard gaSingleInt.accessLocal()
          discard gaDoubleInt.accessLocal()
          discard gaSingleFloat.accessLocal()
          discard gaDoubleFloat.accessLocal()

          gaSingleInt.releaseLocal()
          gaDoubleInt.releaseLocal()
          gaSingleFloat.releaseLocal()
          gaDoubleFloat.releaseLocal()

          check true

          # check padded access

          discard gaSingleInt.accessPadded()
          discard gaDoubleInt.accessPadded()
          discard gaSingleFloat.accessPadded()
          discard gaDoubleFloat.accessPadded()

          gaSingleInt.releaseLocal()
          gaDoubleInt.releaseLocal()
          gaSingleFloat.releaseLocal()
          gaDoubleFloat.releaseLocal()

          check true