import utils/[commandline]
import communication/[mpi]
import globalarrays/[gabase]
import kokkos/[kokkosbase]

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

proc numRanks*: int = int(GA_Nnodes())

test:
  echo "hello from rank ", myRank(), "/", numRanks()