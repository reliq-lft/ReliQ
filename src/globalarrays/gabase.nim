#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/globalarrays/gabase.nim
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
  import gampi
  import utils/[commandline]

template GlobalArrays*(body: untyped): untyped =
  {.pragma: ga, header: "ga.h".}
  {.pragma: ma, header: "madecls.h".}
  body

GlobalArrays: discard

#[ Global Arrays initialization ]#

proc initGA*() {.importc: "GA_Initialize", ga.}

proc initGA*(argc: ptr cint, argv: ptr cstringArray) 
  {.importc: "GA_Initialize_args", ga.}

proc finalizeGA*() {.importc: "GA_Terminate", ga.}

var gaIsLive* = false  ## Tracks whether GA library is initialized

#[ unit tests ]#

when isMainModule:
  block:
    var argc = cargc()
    var argv = cargv(argc)
    
    # Explicit MPI and GA initialization sequence
    # This allows proper shutdown without mpirun warnings
    initMPI(addr argc, addr argv)
    echo "MPI initialized"
    
    initGA()
    echo "Global Arrays initialized"
    
    echo "before GlobalArrays finalization"
    finalizeGA()
    echo "Global Arrays finalized"
    
    finalizeMPI()
    echo "MPI finalized"

    echo "gabase tests completed successfully"