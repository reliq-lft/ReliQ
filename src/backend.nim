import upcxx/[upcxxbase, globalptr]
import kokkos/[kokkosbase]
import kokkos/[views, simd]
import kokkos/[dispatch]

export upcxxbase
export globalptr

export kokkosbase
export views
export simd
export dispatch

# template returning UPC++ and Kokkos pragmas including core headers
template backend*(pragmas: untyped): untyped =
  kokkos: discard
  upcxx: discard
  pragmas

proc numLanes*: int =
  # fetch number of SIMD (SIMT) lanes; needs to work for GPU threads too
  # TO-DO: implement properly; placeholder for now
  let t = newSIMD(float64)
  return t.width