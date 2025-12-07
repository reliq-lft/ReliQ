import utils/[commandline]
import communication/[mpi]
import globalarrays/[gabase]
import kokkos/[kokkosbase]

template main*(body: untyped): untyped =
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
  when isMainModule:
    main: body

test:
  echo "Testing ReliQ initialization"