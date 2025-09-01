import backend

proc reliqInit* {.inline.} =
  ## Initializes ReliQ's runtime environment
  ## Author: Curtis Taylor Peterson
  ## 
  ## Initialize ReliQ runtime environment. This is currently
  ## composed of two keys steps:
  ## 
  ## a.) Initialize UPC++ runtime environemnt
  ## b.) Initialize Kokkos runtime environment
  upcxxInit()
  kokkosInit()

proc reliqFinalize* {.inline.} =
  ## Initializes ReliQ's runtime environment
  ## Author: Curtis Taylor Peterson
  ## 
  ## Finalizes ReliQ runtime environment. This is currently
  ## composed of two keys steps:
  ## 
  ## a.) Finalize Kokkos runtime environment
  ## b.) Finalize UPC++ runtime environemnt
  kokkosFinalize()
  upcxxFinalize()

template reliq*(work: untyped): untyped =
  ## Encapsulates ReliQ's runtime environment in workload block
  ## Author: Curtis Taylor Peterson
  reliqInit()
  work
  reliqFinalize()

when isMainModule:
  import utils

  const encapsulated = true

  proc hello_world() =
    echo "Hello world from process" + $myRank() +
      "out of" + $numRanks() + "processes"
    print "I should only print once"

  if not encapsulated:
    reliqInit()
    hello_world()
    reliqFinalize()
  else: 
    reliq: 
      hello_world()