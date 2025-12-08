template GlobalArrays*(body: untyped): untyped =
  {.pragma: ga, header: "ga.h".}
  {.pragma: ma, header: "madecls.h".}
  body

GlobalArrays: discard

#[ Global Arrays initialization ]#

proc initGlobalArrays*() {.importc: "GA_Initialize", ga.}

proc initGlobalArraysArgs*(argc: ptr cint, argv: ptr cstringArray) 
  {.importc: "GA_Initialize_args", ga.}

proc finalizeGlobalArrays*() {.importc: "GA_Terminate", ga.}

#[ process information ]#

proc GA_Nodeid*(): cint {.importc: "GA_Nodeid", ga.}

proc GA_Nnodes*(): cint {.importc: "GA_Nnodes", ga.}

when isMainModule:
  import utils/[commandline]
  block:
    var argc = cargc()
    var argv = cargv(argc)
    
    # Explicit MPI and GA initialization sequence
    # This allows proper shutdown without mpirun warnings
    initMPI(addr argc, addr argv)
    echo "MPI initialized"
    
    initGlobalArrays()
    echo "Global Arrays initialized"
    
    echo "before GlobalArrays finalization"
    finalizeGlobalArrays()
    echo "Global Arrays finalized"
    
    finalizeMPI()
    echo "MPI finalized"
