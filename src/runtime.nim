import upcxx/[upcxxdefs]
import kokkos/[kokkosdefs]

proc reliqInit* =
  ## Initializes ReliQ's runtime environment
  ## Author: Curtis Taylor Peterson
  ## 
  ## Initialize ReliQ runtime environment. This is currently
  ## composed of two keys steps:
  ## 
  ## a.) Initialize UPC++ runtime environemnt
  ## b.) Initialize Kokkos runtime environment
  upcxx_init()
  kokkos_init()

proc reliqFinalize* =
  ## Initializes ReliQ's runtime environment
  ## Author: Curtis Taylor Peterson
  ## 
  ## Finalizes ReliQ runtime environment. This is currently
  ## composed of two keys steps:
  ## 
  ## a.) Finalize Kokkos runtime environment
  ## b.) Finalize UPC++ runtime environemnt
  kokkos_finalize()
  upcxx_finalize()

when isMainModule:
  import utils

  reliqInit()

  # UPC++ hello world
  echo "Hello world from process" + $upcxx_rank_me() +
    "out of" + $upcxx_rank_n() + "processes"
  print "I should only print once"

  # Kokkos

  reliqFinalize()