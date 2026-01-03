#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/globalarrays/gatypes.nim
  Contact: reliq-lft@proton.me

  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 reliq-lft
  
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.
  
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
]#

import gabase
import gawrap

import utils/[private]

when isMainModule:
  import gampi
  import utils/[commandline]

GlobalArrays: discard

const GLOBALNAME = cstring("ReliQ_GlobalArray")

type GlobalArray*[D: static[int], T] = object
  ## Represents a Global Array
  ##
  ## This object encapsulates a Global Array (GA) handle along with
  ## metadata about its globalGrid dimensions, MPI grid configuration,
  ## and ghost cell widths.
  ## 
  ## Fields:
  ## - `handle`: The underlying GA handle.
  ## - `globalGrid`: An array specifying the size of each dimension of the global array.
  ## - `mpiGrid`: An array specifying the distribution of the global array across MPI ranks.
  ## - `ghostGrid`: An array specifying the width of ghost cells for each dimension.
  ## 
  ## Note: The type parameter `T` indicates the data type of the elements in the global array.
  handle: cint
  globalGrid: array[D, int] 
  localGrid: array[D, int]
  mpiGrid: array[D, int]
  ghostGrid: array[D, int]
  lo, hi: array[D, int]

#[ global array constructor ]#

proc newGlobalArray*[D: static[int]](
  globalGrid: array[D, SomeInteger],
  mpiGrid: array[D, SomeInteger],
  ghostGrid: array[D, SomeInteger],
  T: typedesc
): GlobalArray[D, T] =
  ## Constructor for GlobalArray
  ## 
  ## Creates a new GlobalArray with the specified globalGrid dimensions,
  ## MPI grid configuration, ghost cell widths, and data type.
  ## 
  ## Parameters:
  ## - `globalGrid`: Array specifying the size of each dimension of the global array.
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
  ## let globalGrid = [8, 8, 8, 16]
  ## let mpigrid = [1, 1, 1, 2]
  ## let ghostgrid = [1, 1, 1, 1]
  ## var myGA = newGlobalArray(globalGrid, mpigrid, ghostgrid): float
  ## ```
  let handle = newHandle()
  var dims: array[D, cint]
  var chunks: array[D, cint]
  var widths: array[D, cint]
  var lo, hi: array[D, cint]
  var localGrid: array[D, cint]

  for i in 0..<D:
    dims[i] = cint(globalGrid[i])
    if mpiGrid[i] == -1: chunks[i] = cint(-1) # GlobalArrays decides the chunk size
    else: chunks[i] = cint(globalGrid[i] div mpiGrid[i])
    widths[i] = cint(ghostGrid[i])
  
  handle.setName(GLOBALNAME)
  handle.setData(dims): T
  handle.setChunk(chunks)
  handle.setGhosts(widths)

  alloc handle

  handle.localIndices(lo, hi)
  for i in 0..<D: localGrid[i] = hi[i] - lo[i] + 1

  return GlobalArray[D, T](
    handle: handle,
    globalGrid: globalGrid.mapTo(int),
    localGrid: localGrid.mapTo(int),
    lo: lo.mapTo(int),
    hi: hi.mapTo(int),
    mpiGrid: mpiGrid.mapTo(int),
    ghostGrid: ghostGrid.mapTo(int)
  )

#[ global array move semantics ]#

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
    if a.globalGrid[i] != b.globalGrid[i]: return false
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
  if ga.handle != 0: ga.handle.GA_Destroy()

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
    GA_Sync()
  elif dest.handle == 0 and src.handle != 0:
    dest = newGlobalArray(src.globalGrid, src.mpiGrid, src.ghostGrid): T
    GA_Copy(src.handle, dest.handle) 
    GA_Sync()
  elif src.handle == 0:
    let errMsg = "Error in GA copy from " & $src.handle & " to " & $dest.handle
    raise newException(ValueError): errMsg & "; source is uninitialized"

#[ accessors ]#

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

proc getGlobalGrid*[D: static[int], T](ga: GlobalArray[D, T]): array[D, int] =
  ## Accessor for the lattice dimensions
  ##
  ## Returns the lattice dimensions of the GlobalArray.
  ##
  ## Parameters:
  ## - `ga`: The GlobalArray instance.
  ##
  ## Returns:
  ## An array representing the lattice dimensions.
  return ga.globalGrid

proc getLocalGrid*[D: static[int], T](ga: GlobalArray[D, T]): array[D, int] =
  ## Accessor for the lattice dimensions
  ##
  ## Returns the lattice dimensions of the GlobalArray.
  ##
  ## Parameters:
  ## - `a`: The GlobalArray or HostView instance.
  ##
  ## Returns:
  ## An array representing the lattice dimensions.
  return ga.localGrid

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

proc numSites*[D: static[int], T](ga: GlobalArray[D, T]): int =
  ## Get the number of local sites for this process
  ##
  ## Parameters:
  ## - `ga`: The GlobalArray instance
  ##
  ## Returns:
  ## The number of local sites for the current process
  result = 1
  for i in 0..<D: result *= ga.hi[i] - ga.lo[i] + 1

proc getBounds*[D: static[int], T](
  ga: GlobalArray[D, T]
): (array[D, int], array[D, int]) =
  ## Get the local bounds for this process
  ##
  ## Parameters:
  ## - `ga`: The GlobalArray instance
  ##
  ## Returns:
  ## A tuple containing two arrays: the lower and upper bounds of the local segment
  return (ga.lo, ga.hi)

#[ halo exchange ]#

proc updateGhostDirection*[D: static[int], T](
  ga: GlobalArray[D, T],
  dir: SomeInteger,
  side: SomeInteger,
  update_corners: bool = true
) =
  ## Update ghost cells in a specific direction and side
  ##
  ## This procedure updates the ghost cells of the GlobalArray
  ## in a specified direction and side (lower or upper). It can also
  ## optionally update corner ghost cells.
  ##
  ## Parameters:
  ## - `ga`: The GlobalArray instance.
  ## - `dir`: The direction index for the ghost cell update.
  ## - `side`: The side index (0 for lower, 1 for upper).
  ## - `update_corners`: Whether to update corner ghost cells (default: true).
  let update_corners_c = (if update_corners: cint(1) else: cint(0))
  ga.handle.GA_Update_ghost_dir(
    cint(dir), 
    cint(side), 
    update_corners_c
  )
  GA_Sync()

proc updateGhosts*[D: static[int], T](
  ga: GlobalArray[D, T]; 
  update_corners: bool = true
) =
  ## Update the ghost cells of the GlobalArray
  ##
  ## This procedure triggers a ghost cell update for the GlobalArray,
  ## ensuring that the ghost cells contain up-to-date data from neighboring
  ## processes.
  ##
  ## Parameters:
  ## - `ga`: The GlobalArray instance.
  for mu in 0..<D:
    ga.updateGhostDirection(mu, 1, update_corners = update_corners)
    ga.updateGhostDirection(mu, -1, update_corners = update_corners)

#[ misc ]#

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

#[ unit tests ]#

when isMainModule:
  block:
    var argc = cargc()
    var argv = cargv(argc)
    
    # Explicit MPI and GA initialization sequence
    # This allows proper shutdown without mpirun warnings
    initMPI(addr argc, addr argv)
    initGA()
    
    # Scope all GA operations so destructors run before finalizeGA()
    block:
      # small local type initialization tests

      var gaTypeInt32 = toGAType(int32)
      var gaTypeInt64 = toGAType(int64)
      var gaTypeFloat32 = toGAType(float32)
      var gaTypeFloat64 = toGAType(float64)

      echo "GA type for int32: ", gaTypeInt32
      echo "GA type for int64: ", gaTypeInt64
      echo "GA type for float32: ", gaTypeFloat32
      echo "GA type for float64: ", gaTypeFloat64

      # full GlobalArray construction tests

      let lattice = [8, 8, 8, 8*GA_Nnodes()]
      let mpigrid = [1, 1, 1, GA_Nnodes()]
      let ghostgrid = [1, 1, 1, 1]
      var testGA1 = newGlobalArray(lattice, mpigrid, ghostgrid): float
      var testGA2 = newGlobalArray(lattice, mpigrid, ghostgrid): float64
      var testGA3 = newGlobalArray(lattice, mpigrid, ghostgrid): float32
      var testGA4 = newGlobalArray(lattice, mpigrid, ghostgrid): int
      var testGA5 = newGlobalArray(lattice, mpigrid, ghostgrid): int32
      var testGA6 = newGlobalArray(lattice, mpigrid, ghostgrid): int64

      assert testGA1.isInitialized(), "GA initialization failed."
      assert testGA2.isInitialized(), "GA initialization failed."
      assert testGA3.isInitialized(), "GA initialization failed."
      assert testGA4.isInitialized(), "GA initialization failed."
      assert testGA5.isInitialized(), "GA initialization failed."
      assert testGA6.isInitialized(), "GA initialization failed."

      assert testGA1.getGlobalGrid() == lattice, "Lattice accessor failed."
      assert testGA1.getMPIGrid() == mpigrid, "MPI grid accessor failed."
      assert testGA1.getGhostGrid() == ghostgrid, "Ghost grid accessor failed."

      echo "GA construction tests passed"

      testGA2 = testGA1

      assert testGA1.handle != testGA2.handle, "GA copy must not copy handles."
      
      echo "GA copy semantics tests passed"

      testGA1.updateGhosts()

      echo "GA ghost update tests passed"
    
    # All GlobalArrays are now destroyed, safe to finalize
    finalizeGA()
    finalizeMPI()
    
    echo "gatypes tests completed successfully"