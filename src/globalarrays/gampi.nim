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