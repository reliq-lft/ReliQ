#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/reliq.nim
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
import communication/[mpi]
import globalarrays/[gabase]
import kokkos/[kokkosbase]
import utils/[nimutils]

export nimutils

template main*(body: untyped): untyped =
  ## Main execution
  ## 
  ## Initializes MPI, Kokkos, and Global Arrays, then executes the provided body of code,
  ## and finally finalizes Global Arrays, Kokkos, and MPI.
  ## 
  ## Parameters:
  ## - `body`: The main body of code to execute within the initialized environment.
  ## 
  ## Example:
  ## ```nim
  ## main:
  ##   echo "Hello, World!"
  ## ```
  block:
    let argc {.inject.} = cargc()
    let argv {.inject.} = cargv(argc)
    
    initMPI(addr argc, addr argv)
    initKokkos(argc, argv)
    initGlobalArrays()

    deallocCStringArray(argv)
    body
    
    finalizeGlobalArrays()
    finalizeKokkos()
    finalizeMPI()

template test*(body: untyped): untyped =
  ## Test execution
  ## 
  ## Executes the provided body of code as a test case within the main execution environment.
  ## 
  ## Parameters:
  ## - `body`: The test code to execute.
  ## 
  ## Example:
  ## ```nim
  ## test:
  ##   echo "Running test case"
  ## ```
  when isMainModule:
    main: body
  
proc myRank*: int = int(GA_Nodeid())
  ## Returns the rank of the current process in the MPI communicator
  ## 
  ## Returns:
  ## Rank of the current process
  ## 
  ## Example:
  ## ```nim
  ## let rank = myRank()
  ## echo "My rank is ", rank
  ## ```

proc numRanks*: int = int(GA_Nnodes())
  ## Returns the total number of processes in the MPI communicator
  ## 
  ## Returns:
  ## Total number of processes
  ## 
  ## Example:
  ## ```nim
  ## let totalRanks = numRanks()
  ## echo "Total number of ranks is ", totalRanks
  ## ```

test:
  echo "hello from rank ", myRank(), "/", numRanks()