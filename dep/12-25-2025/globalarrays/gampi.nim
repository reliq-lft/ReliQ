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

when isMainModule:
  import utils/[commandline]

template MPI(body: untyped): untyped =
  {.pragma: mpi, header: "mpi.h".}

MPI: discard

#[ OpenMPI initialization ]#

proc initMPI*(argc: ptr cint, argv: ptr cstringArray): cint
  {.importc: "MPI_Init", mpi, discardable.}

proc finalizeMPI*(): cint {.importc: "MPI_Finalize", mpi, discardable.}

#[ MPI communication ]#

# Import raw MPI_Allreduce function
proc MPI_Allreduce*(
  sendbuf: pointer, 
  recvbuf: pointer, 
  count: cint, 
  datatype: cint, 
  op: cint, 
  comm: cint
): cint {.importc: "MPI_Allreduce", mpi, discardable.}

# MPI constants
var MPI_COMM_WORLD* {.importc: "MPI_COMM_WORLD", mpi.}: cint
var MPI_SUM* {.importc: "MPI_SUM", mpi.}: cint
var MPI_DOUBLE* {.importc: "MPI_DOUBLE", mpi.}: cint
var MPI_FLOAT* {.importc: "MPI_FLOAT", mpi.}: cint
var MPI_INT* {.importc: "MPI_INT", mpi.}: cint
var MPI_LONG_LONG* {.importc: "MPI_LONG_LONG", mpi.}: cint

# MPI_Allreduce wrappers for different types
proc allReduceFloat64*(sendbuf: ptr float64, recvbuf: ptr float64, count: cint): cint =
  if sendbuf.isNil or recvbuf.isNil:
    raise newException(ValueError, "Null pointer passed to allReduceFloat64")
  return MPI_Allreduce(cast[pointer](sendbuf), cast[pointer](recvbuf), count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD)

proc allReduceFloat32*(sendbuf: ptr float32, recvbuf: ptr float32, count: cint): cint =
  if sendbuf.isNil or recvbuf.isNil:
    raise newException(ValueError, "Null pointer passed to allReduceFloat32")
  return MPI_Allreduce(cast[pointer](sendbuf), cast[pointer](recvbuf), count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD)

proc allReduceInt32*(sendbuf: ptr int32, recvbuf: ptr int32, count: cint): cint =
  if sendbuf.isNil or recvbuf.isNil:
    raise newException(ValueError, "Null pointer passed to allReduceInt32")
  return MPI_Allreduce(cast[pointer](sendbuf), cast[pointer](recvbuf), count, MPI_INT, MPI_SUM, MPI_COMM_WORLD)

proc allReduceInt64*(sendbuf: ptr int64, recvbuf: ptr int64, count: cint): cint =
  if sendbuf.isNil or recvbuf.isNil:
    raise newException(ValueError, "Null pointer passed to allReduceInt64")
  return MPI_Allreduce(cast[pointer](sendbuf), cast[pointer](recvbuf), count, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD)

#[ unit tests ]#

when isMainModule:
  block:
    var argc = cargc()
    var argv = cargv(argc)

    let errMsg1 = "MPI initialization failed with error code: "
    let errMsg2 = "MPI finalization failed with error code: "
    
    # Explicit MPI initialization and finalization
    let mpiInitResult = initMPI(addr argc, addr argv)
    assert mpiInitResult == 0, errMsg1 & $mpiInitResult
    
    # Finalize MPI
    let mpiFinalizeResult = finalizeMPI()
    assert mpiFinalizeResult == 0, errMsg2 & $mpiFinalizeResult

    echo "MPI initialization and finalization tests completed successfully"