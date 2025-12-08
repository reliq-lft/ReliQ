import reliq
import gabase
import gatypes

GlobalArrays: discard

const GLOBALNAME = cstring("ReliQ_GlobalArray")

type GlobalArray*[D: static[int], T] = object
  ## Represents a Global Array
  ##
  ## This object encapsulates a Global Array (GA) handle along with
  ## metadata about its lattice dimensions, MPI grid configuration,
  ## and ghost cell widths.
  ## 
  ## Fields:
  ## - `handle`: The underlying GA handle.
  ## - `lattice`: An array specifying the size of each dimension of the global array.
  ## - `mpiGrid`: An array specifying the distribution of the global array across MPI ranks.
  ## - `ghostGrid`: An array specifying the width of ghost cells for each dimension.
  ## 
  ## Note: The type parameter `T` indicates the data type of the elements in the global array.
  handle: cint
  lattice: array[D, int] 
  mpiGrid: array[D, int]
  ghostGrid: array[D, int]

type LocalData*[D: static[int], T] = object
  ## Represents a local portion of a GlobalArray
  ##
  ## This object holds a pointer to the local data and metadata about
  ## its bounds and layout. Must be released when done.
  ga_handle: cint
  data*: ptr UncheckedArray[T]
  lo*: array[D, int]
  hi*: array[D, int]
  ld*: array[D-1, int]  # Leading dimensions for multidimensional access

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

proc GA_Destroy(g_a: cint) {.importc: "GA_Destroy", ga.}

proc GA_Copy(g_a: cint, g_b: cint) {.importc: "GA_Copy", ga.}

proc NGA_Distribution(g_a: cint, pid: cint, lo: ptr cint, hi: ptr cint) 
  {.importc: "NGA_Distribution", ga.}

proc NGA_Access(g_a: cint, lo: ptr cint, hi: ptr cint, p: ptr pointer, ld: ptr cint) 
  {.importc: "NGA_Access", ga.}

proc NGA_Release(g_a: cint, lo: ptr cint, hi: ptr cint) 
  {.importc: "NGA_Release", ga.}

#[ GA constructor, destructor, copy assignment ]#

proc newGlobalArray*[D: static[int]](
  lattice: array[D, int],
  mpiGrid: array[D, int],
  ghostGrid: array[D, int],
  T: typedesc
): GlobalArray[D, T] =
  ## Constructor for GlobalArray
  ## 
  ## Creates a new GlobalArray with the specified lattice dimensions,
  ## MPI grid configuration, ghost cell widths, and data type.
  ## 
  ## Parameters:
  ## - `lattice`: Array specifying the size of each dimension of the global array.
  ## - `mpiGrid`: Array specifying distribution of the global array across MPI ranks.
  ## - `ghostGrid`: An array specifying the width of ghost cells for each dimension.
  ## - `T`: The data type of the elements in the global array.
  ## 
  ## Returns:
  ## A new instance of `GlobalArray[D, T]`.
  ## 
  ## Raises:
  ## - `ValueError`: If the GA allocation fails.
  ## 
  ## Example:
  ## ```nim
  ## let lattice = [8, 8, 8, 16]
  ## let mpigrid = [1, 1, 1, 2]
  ## let ghostgrid = [1, 1, 1, 1]
  ## var myGA = newGlobalArray(lattice, mpigrid, ghostgrid): float
  ## ```
  let handle = GA_Create_handle()
  var dims: array[D, cint]
  var chunks: array[D, cint]
  var widths: array[D, cint]

  for i in 0..<D:
    dims[i] = cint(lattice[i])
    chunks[i] = cint(lattice[i] div mpiGrid[i])
    widths[i] = cint(ghostGrid[i])
  
  handle.GA_Set_name(cast[ptr cchar](GLOBALNAME))
  handle.GA_Set_data(cint(D), addr dims[0], toGAType(T))
  handle.GA_Set_chunk(addr chunks[0])
  handle.GA_Set_ghosts(addr widths[0])

  let status = handle.GA_Allocate()
  if status == 0:
    let errMsg = "Error in GA " & $handle & " construction"
    raise newException(ValueError, errMsg & "; status code: " & $status)

  return GlobalArray[D, T](
    handle: handle,
    lattice: lattice,
    mpiGrid: mpiGrid,
    ghostGrid: ghostGrid
  )

proc conformable*[D: static[int], T](a, b: GlobalArray[D, T]): bool =
  ## Checks if two GlobalArrays are conformable
  ##
  ## Two GlobalArrays are conformable if they have the same lattice dimensions,
  ## MPI grid configuration, and ghost cell widths.
  ##
  ## Parameters:
  ## - `a`: The first GlobalArray.
  ## - `b`: The second GlobalArray.
  ##
  ## Returns:
  ## `true` if the two GlobalArrays are conformable, `false` otherwise.
  for i in 0..<D:
    if a.lattice[i] != b.lattice[i]: return false
    if a.mpiGrid[i] != b.mpiGrid[i]: return false
    if a.ghostGrid[i] != b.ghostGrid[i]: return false
  return true

proc `=destroy`*[D: static[int], T](ga: GlobalArray[D, T]) =
  ## Destructor for GlobalArray
  ## 
  ## Frees the resources associated with the GlobalArray.
  ##
  ## Parameters:
  ## - `ga`: The GlobalArray instance to be destroyed.
  if ga.handle > 0: ga.handle.GA_Destroy()

proc `=copy`*[D: static[int], T](
  dest: var GlobalArray[D, T], 
  src: GlobalArray[D, T]
) =
  ## Copy semantics for GlobalArray
  ## 
  ## Creates a copy of the source GlobalArray into the destination. All data and 
  ## fields are copied, except for the handle, which is preserved to ensure that
  ## each GlobalArray instance manages its own GA resource.
  ##
  ## Parameters:
  ## - `dest`: The destination GlobalArray to copy into.
  ## - `src`: The source GlobalArray to copy from.
  if dest.handle == src.handle: return
  if dest.handle != 0 and src.handle != 0 and conformable(dest, src):
    GA_Copy(src.handle, dest.handle)
  elif dest.handle == 0 and src.handle != 0:
    dest = newGlobalArray(src.lattice, src.mpiGrid, src.ghostGrid): T
    GA_Copy(src.handle, dest.handle)
  elif src.handle == 0:
    let errMsg = "Error in GA copy from " & $src.handle & " to " & $dest.handle
    raise newException(ValueError, errMsg & "; source is uninitialized")

#[ LD destructors, copy assignment ]#

proc `=destroy`*[D: static[int], T](local: LocalData[D, T]) =
  ## Destructor for LocalData - releases the GA access
  if local.ga_handle > 0:
    var lo_c: array[D, cint]
    var hi_c: array[D, cint]
    for i in 0..<D:
      lo_c[i] = cint(local.lo[i])
      hi_c[i] = cint(local.hi[i])
    NGA_Release(local.ga_handle, addr lo_c[0], addr hi_c[0])

proc `=copy`*[D: static[int], T](dest: var LocalData[D, T], src: LocalData[D, T]) {.error.}
  ## Prevent copying of LocalData - it represents exclusive access to GA data

#[ GA accessors ]#

proc getHandle*[D: static[int], T](ga: GlobalArray[D, T]): cint =
  ## Accessor for the GA handle
  ##
  ## Returns the underlying GA handle associated with the GlobalArray.
  ##
  ## Parameters:
  ## - `ga`: The GlobalArray instance.
  ##
  ## Returns:
  ## The GA handle as a `cint`.
  return ga.handle

proc getLattice*[D: static[int], T](ga: GlobalArray[D, T]): array[D, int] =
  ## Accessor for the lattice dimensions
  ##
  ## Returns the lattice dimensions of the GlobalArray.
  ##
  ## Parameters:
  ## - `ga`: The GlobalArray instance.
  ##
  ## Returns:
  ## An array representing the lattice dimensions.
  return ga.lattice

proc getMPIGrid*[D: static[int], T](ga: GlobalArray[D, T]): array[D, int] =
  ## Accessor for the MPI grid configuration
  ##
  ## Returns the MPI grid configuration of the GlobalArray.
  ##
  ## Parameters:
  ## - `ga`: The GlobalArray instance.
  ##
  ## Returns:
  ## An array representing the MPI grid configuration.
  return ga.mpiGrid

proc getGhostGrid*[D: static[int], T](ga: GlobalArray[D, T]): array[D, int] =
  ## Accessor for the ghost cell widths
  ##
  ## Returns the ghost cell widths of the GlobalArray.
  ##
  ## Parameters:
  ## - `ga`: The GlobalArray instance.
  ##
  ## Returns:
  ## An array representing the ghost cell widths.
  return ga.ghostGrid

proc isInitialized*[D: static[int], T](ga: GlobalArray[D, T]): bool =
  ## Checks if the GlobalArray is initialized
  ##
  ## A GlobalArray is considered initialized if its GA handle is valid (not zero).
  ##
  ## Parameters:
  ## - `ga`: The GlobalArray instance.
  ##
  ## Returns:
  ## `true` if the GlobalArray is initialized, `false` otherwise.
  return ga.handle != 0

#[ downcasting to local data ]#

proc downcast*[D: static[int], T](ga: GlobalArray[D, T]): LocalData[D, T] =
  ## Access the local portion of the GlobalArray on the current process
  ##
  ## Returns a LocalData object containing a pointer to the local data
  ## and its bounds. The data is automatically released when LocalData
  ## goes out of scope.
  ##
  ## Parameters:
  ## - `ga`: The GlobalArray to access
  ##
  ## Returns:
  ## LocalData containing pointer to local data, bounds, and leading dimensions
  ##
  ## Example:
  ## ```nim
  ## var myGA = newGlobalArray(lattice, mpigrid, ghostgrid, float)
  ## let local = <- myGA
  ## # Use local.data, local.lo, local.hi, local.ld
  ## # Automatically released when local goes out of scope
  ## ```
  var lo_c: array[D, cint]
  var hi_c: array[D, cint]
  var ld_c: array[D-1, cint]
  var p: pointer
  var local: LocalData[D, T]
  let pid = GA_Nodeid()

  NGA_Distribution(ga.handle, pid, addr lo_c[0], addr hi_c[0])
  NGA_Access(ga.handle, addr lo_c[0], addr hi_c[0], addr p, addr ld_c[0])
  
  local.ga_handle = ga.handle
  local.data = cast[ptr UncheckedArray[T]](p)
  for i in 0..<D:
    local.lo[i] = int(lo_c[i])
    local.hi[i] = int(hi_c[i])
  for i in 0..<(D-1): local.ld[i] = int(ld_c[i])
  
  return local

#[ LD misc ]#

proc `$`*[D:static[int], T](local: LocalData[D, T]): string =
  ## String representation of LocalData
  ##
  ## Returns a string summarizing the LocalData's GA handle and bounds.
  ##
  ## Parameters:
  ## - `local`: The LocalData instance.
  ##
  ## Returns:
  ## A string representation of the LocalData.
  result = "LocalData(ga_handle: " & $local.ga_handle & ", lo: ["
  for i in 0..<D:
    result.add($local.lo[i])
    if i < D-1: result.add(", ")
  result.add("], hi: [")
  for i in 0..<D:
    result.add($local.hi[i])
    if i < D-1: result.add(", ")
  result.add("])")

#[ unit tests ]#

test:
  let lattice = [8, 8, 8, 8*numRanks()]
  let mpigrid = [1, 1, 1, numRanks()]
  let ghostgrid = [1, 1, 1, 1]
  var testGA1 = newGlobalArray(lattice, mpigrid, ghostgrid): float
  var testGA2 = newGlobalArray(lattice, mpigrid, ghostgrid): float
  var testGA3 = newGlobalArray(lattice, mpigrid, ghostgrid): float32
  var testGA4 = newGlobalArray(lattice, mpigrid, ghostgrid): int

  assert(testGA1.isInitialized(), "GA initialization failed.")
  assert(testGA2.isInitialized(), "GA initialization failed.")
  assert(testGA3.isInitialized(), "GA initialization failed.")
  assert(testGA4.isInitialized(), "GA initialization failed.")

  assert(testGA1.getLattice() == lattice, "Lattice accessor failed.")
  assert(testGA1.getMPIGrid() == mpigrid, "MPI grid accessor failed.")
  assert(testGA1.getGhostGrid() == ghostgrid, "Ghost grid accessor failed.")

  testGA2 = testGA1

  assert(testGA1.handle != testGA2.handle, "GA copy must not copy handles.")

  testGA1 = testGA1

  let testLD1 = downcast(testGA1)
  let testLD2 = downcast(testGA2)

  assert(testLD1.ga_handle == testGA1.handle, "LocalData GA handle mismatch.")
  assert(testLD2.ga_handle == testGA2.handle, "LocalData GA handle mismatch.")

  let pid = myRank()
  let nnd = numRanks()
  echo "Process ", pid, "/", nnd, ": ", testLD1
  echo "Process ", pid, "/", nnd, ": ", testLD2

  for i in 0..<4:
    assert(testLD1.lo[i] >= 0 and testLD1.hi[i] < lattice[i], "LocalData bounds out of range.")
    assert(testLD2.lo[i] >= 0 and testLD2.hi[i] < lattice[i], "LocalData bounds out of range.")
  

