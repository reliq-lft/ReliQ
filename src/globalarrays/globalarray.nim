import reliq
import gabase
import gatypes

GlobalArrays: discard

const GLOBALNAME = cstring("ReliQ_GlobalArray")

type 
  GlobalArray*[D: static[int], T] = object
    handle: int
    lattice: array[D, int] 
    mpiGrid: array[D, int]
    ghostGrid: array[D, int]

#[ GA wrappers ]#

proc GA_Create_handle: cint {.importc: "GA_Create_handle", ga.}

proc GA_Allocate(g_a: cint): cint {.importc: "GA_Allocate", ga.}

proc GA_Set_name(g_a: cint, name: ptr cchar) {.importc: "GA_Set_array_name", ga.}

proc GA_Set_data(
  g_a: cint,  
  array_ndim: cint, 
  array_dims: ptr cint, 
  data_type: cint
) {.importc: "GA_Set_data", ga.}

proc GA_Set_chunk(g_a: cint, chunk_dims: ptr cint) {.importc: "GA_Set_chunk", ga.}

proc GA_Set_ghosts(g_a: cint, widths: ptr cint) {.importc: "GA_Set_ghosts", ga.}

#[
proc GA_Get(
  g_a: cint, 
  lo: ptr cint, 
  hi: ptr cint, 
  buf: ptr void, 
  ld_buf: cint
) {.importc: "GA_Get", ga.}
]#

#[ GA constructor(s) ]#

proc newGlobalArray*[D: static[int]](
  latticeGrid: array[D, int],
  mpiGrid: array[D, int],
  ghostGrid: array[D, int],
  T: typedesc
): GlobalArray[D, T] =
  var handle = GA_Create_handle()
  var dims: array[D, cint]
  var chunks: array[D, cint]
  var widths: array[D, cint]

  for i in 0..<D:
    dims[i] = cint(latticeGrid[i])
    chunks[i] = cint(latticeGrid[i] div mpiGrid[i])
    widths[i] = cint(ghostGrid[i])
  
  handle.GA_Set_name(cast[ptr cchar](GLOBALNAME))
  handle.GA_Set_data(cint(D), addr dims[0], toGAType(T))
  handle.GA_Set_chunk(addr chunks[0])
  handle.GA_Set_ghosts(addr widths[0])

  let status = handle.GA_Allocate()
  if status == 0:
    let errMsg = "Error in GA " & $handle & ", status code: " & $status
    raise newException(ValueError, errMsg)

  return GlobalArray[D, T](
    handle: handle,
    lattice: latticeGrid,
    mpiGrid: mpiGrid,
    ghostGrid: ghostGrid
  )

#[ GA accessors ]#

#[
proc `[]`*[D: static[int], T](ga: GlobalArray[D], idx: array[D, int]): T =
  ## Access element at global index `idx` of global array `ga`
  var value: T
  var lo: array[D, cint]
  var hi: array[D, cint]

  for i in 0..<D:
    lo[i] = cint(idx[i])
    hi[i] = cint(idx[i])

  GA_Get(ga.handle, addr lo[0], addr hi[0], addr value, cint(1))
  return value
]#

test:
  let lattice = [8, 8, 8, 16]
  let mpigrid = [1, 1, 1, 2]
  let ghostgrid = [1, 1, 1, 1]
  var testGA1 = newGlobalArray(lattice, mpigrid, ghostgrid): float
  var testGA2 = newGlobalArray(lattice, mpigrid, ghostgrid): float

  echo "Created GA 1 with handle: ", testGA1.handle
  echo "Created GA 2 with handle: ", testGA2.handle

  #echo testGA1[[0,0,0,0]]