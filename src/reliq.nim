import upcxx/[upcxxdefs]
import kokkos/[kokkosdefs]
import lattice/[bravais]

export bravais

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
  # kokkos init

proc reliqFinalize* =
  ## Initializes ReliQ's runtime environment
  ## Author: Curtis Taylor Peterson
  ## 
  ## Finalizes ReliQ runtime environment. This is currently
  ## composed of two keys steps:
  ## 
  ## a.) Finalize UPC++ runtime environemnt
  ## b.) Finalize Kokkos runtime environment
  upcxx_finalize()
  # kokkos finalize