import upcxx/[upcxxbase, globalptr]
import kokkos/[kokkosbase, views]
import kokkos/[simd]

export upcxxbase
export globalptr

export kokkosbase
export views
export simd

proc numLanes*: int =
  # fetch number of SIMD (SIMT) lanes; needs to work for GPU threads too
  # TO-DO: implement properly; placeholder for now
  let t = newSIMD(float64)
  return t.width