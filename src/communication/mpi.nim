#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/communication/mpi.nim
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

import utils/[commandline]

template MPI(body: untyped): untyped =
  {.pragma: mpi, header: "mpi.h".}

MPI: discard

type MPI_Comm* {.importc: "MPI_Comm", mpi.} = pointer

var MPI_COMM_WORLD* {.importc: "MPI_COMM_WORLD", mpi.}: MPI_Comm

proc initMPI*(argc: ptr cint, argv: ptr cstringArray): cint
  {.importc: "MPI_Init", mpi, discardable.}

proc finalizeMPI*(): cint {.importc: "MPI_Finalize", mpi, discardable.}

proc MPI_Comm_rank*(comm: MPI_Comm, rank: ptr cint): cint
  {.importc: "MPI_Comm_rank", mpi.}

proc MPI_Comm_size*(comm: MPI_Comm, size: ptr cint): cint
  {.importc: "MPI_Comm_size", mpi.}
