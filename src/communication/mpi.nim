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
