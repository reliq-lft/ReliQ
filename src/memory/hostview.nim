#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/memory/hostview.nim
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

import globalarrays/[gatypes, gawrap]
import lattice/[indexing]
import utils/[private]

when isMainModule:
  import globalarrays/[gampi, gabase]
  import utils/[commandline]

type HostView*[D: static[int], T] = object
  ## Represents a local array on host memory
  ##
  ## This object encapsulates a local array along with metadata about its
  ## globalGrid dimensions, MPI grid configuration, and ghost cell widths.
  ## 
  ## Fields:
  ## - `data`: A pointer to an UncheckedArray of type `T` representing the local data.
  ## - `globalGrid`: An array specifying the size of each dimension of the global array.
  ## - `mpiGrid`: An array specifying the distribution of the global array across MPI ranks.
  ## - `ghostGrid`: An array specifying the width of ghost cells for each dimension.
  ## 
  ## Note: The type parameter `T` indicates the data type of the elements in the local array.
  handle: cint
  data: ptr UncheckedArray[T]
  localGrid: array[D, int]
  paddedGrid: array[D, int]
  ghostGrid: array[D, int]
  lo, hi: array[D, int]
  ld: array[D-1, int]
  hasPadding: bool

#[ constructors ]#

proc hostView*[D: static[int], T](
  ga: GlobalArray[D, T];
  pad: bool = false
): HostView[D, T] =
  ## Constructor for HostView
  ##
  ## Creates a new HostView associated with the given GlobalArray.
  ##
  ## Parameters:
  ## - `ga`: The GlobalArray instance to associate with the local array.
  ##
  ## Returns:
  ## A new instance of `HostView[D, T]`.
  let handle = ga.getHandle()
  var paddedGrid: array[D, cint]
  var lo_c, hi_c: array[D, cint]
  var lo, hi: array[D, int]
  var ld: array[D-1, cint]
  var p: pointer
  let pid = GA_Nodeid()

  handle.NGA_Distribution(pid, addr lo_c[0], addr hi_c[0])
  if pad: handle.NGA_Access_ghosts(addr paddedGrid[0], addr p, addr ld[0])
  else: 
    handle.NGA_Access(addr lo_c[0], addr hi_c[0], addr p, addr ld[0])
    paddedGrid = ga.getLocalGrid().mapTo(cint)

  (lo, hi) = (lo_c.mapTo(int), hi_c.mapTo(int))

  return HostView[D, T](
    handle: handle,
    data: cast[ptr UncheckedArray[T]](p),
    localGrid: ga.getLocalGrid(),
    paddedGrid: paddedGrid.mapTo(int),
    lo: lo,
    hi: hi,
    ld: ld.mapTo(int),
    ghostGrid: ga.getGhostGrid(),
    hasPadding: pad
  )

#[ destructor ]#

proc `=destroy`[D: static[int], T](hv: var HostView[D, T]) =
  ## Destructor for HostView
  ##
  ## Releases the Global Arrays handle when the HostView is destroyed.
  var (lo, hi) = (hv.lo.mapTo(cint), hv.hi.mapTo(cint))
  if hv.handle != 0:
    hv.handle.NGA_Release(addr lo[0], addr hi[0])

#[ accessors ]#

template `[]`*[D: static[int], T](hv: HostView[D, T]; idx: SomeInteger): T =
  ## Access operator for HostView
  ##
  ## Provides access to elements of the local array using multi-dimensional indices.
  ##
  ## Parameters:
  ## - `idx`: A variable number of integer indices specifying the position in each dimension.
  ##
  ## Returns:
  ## The element of type `T` at the specified indices.
  hv.data[idx]

proc `[]`*[D: static[int]; I: SomeInteger; T](
  hv: HostView[D, T]; 
  idx: array[D, I]
): T =
  ## Access operator for HostView
  ##
  ## Provides access to elements of the local array using an array of indices.
  ##
  ## Parameters:
  ## - `idx`: An array of integer indices specifying the position in each dimension.
  ##
  ## Returns:
  ## The element of type `T` at the specified indices.
  hv.data[idx.coordsToFlat(hv.paddedGrid)]

proc `[]=`*[D: static[int]; I: SomeInteger; T](
  hv: var HostView[D, T]; 
  idx: SomeInteger; 
  value: T
) =
  ## Assignment operator for HostView
  ##
  ## Allows assignment to elements of the local array using multi-dimensional indices.
  ##
  ## Parameters:
  ## - `idx`: A variable number of integer indices specifying the position in each dimension.
  ## - `value`: The value of type `T` to assign at the specified indices.
  hv.data[idx] = value

proc `[]=`*[D: static[int]; I: SomeInteger; T](
  hv: var HostView[D, T]; 
  idx: array[D, I]; 
  value: T
) =
  ## Assignment operator for HostView
  ##
  ## Allows assignment to elements of the local array using an array of indices.
  ##
  ## Parameters:
  ## - `idx`: An array of integer indices specifying the position in each dimension.
  ## - `value`: The value of type `T` to assign at the specified indices.
  hv.data[idx.coordsToFlat(hv.paddedGrid)] = value

proc numSites*[D: static[int], T](hv: HostView[D, T]): int =
  ## Get the number of local sites in the HostView
  ##
  ## Parameters:
  ## - `hv`: The HostView instance.
  ##
  ## Returns:
  ## The total number of local sites as an integer.
  return hv.paddedGrid.product()

proc numLocalSites*[D: static[int], T](hv: HostView[D, T]): int =
  ## Get the number of local sites excluding ghost cells in the HostView
  ##
  ## Parameters:
  ## - `hv`: The HostView instance.
  ##
  ## Returns:
  ## The total number of local sites excluding ghost cells as an integer.
  return hv.localGrid.product()

proc getData*[D: static[int], T](hv: HostView[D, T]): ptr UncheckedArray[T] =
  ## Get the raw data pointer of the HostView
  ##
  ## Parameters:
  ## - `hv`: The HostView instance.
  ##
  ## Returns:
  ## A pointer to the underlying UncheckedArray of type `T`.
  return hv.data

proc getHandle*[D: static[int], T](hv: HostView[D, T]): cint =
  ## Get the Global Arrays handle associated with the HostView
  ##
  ## Parameters:
  ## - `hv`: The HostView instance.
  ##
  ## Returns:
  ## The Global Arrays handle as an integer.
  return hv.handle

proc getBounds*[D: static[int], T](
  hv: HostView[D, T]
): (array[D, int], array[D, int]) =
  ## Get the local bounds for this process
  ##
  ## Parameters:
  ## - `hv`: The HostView instance.
  ##
  ## Returns:
  ## A tuple containing the lower and upper bounds as arrays of integers.
  return (hv.lo, hv.hi)

proc getLocalGrid*[D: static[int], T](hv: HostView[D, T]): array[D, int] =
  ## Get the local grid dimensions of the HostView
  ##
  ## Parameters:
  ## - `hv`: The HostView instance.
  ##
  ## Returns:
  ## An array representing the local grid dimensions.
  return hv.localGrid

proc getPaddedGrid*[D: static[int], T](hv: HostView[D, T]): array[D, int] =
  ## Get the padded grid dimensions of the HostView
  ##
  ## Parameters:
  ## - `hv`: The HostView instance.
  ##      
  ## Returns:
  ## An array representing the padded grid dimensions.
  return hv.paddedGrid

proc getGhostGrid*[D: static[int], T](hv: HostView[D, T]): array[D, int] =
  ## Get the ghost cell widths of the HostView
  ##
  ## Parameters:
  ## - `hv`: The HostView instance.
  ##
  ## Returns:
  ## An array representing the ghost cell widths for each dimension.
  return hv.ghostGrid

proc hasPadding*[D: static[int], T](hv: HostView[D, T]): bool =
  ## Check if the HostView has padding for ghost cells
  ##
  ## Parameters:
  ## - `hv`: The HostView instance.
  ## Returns:
  ## A boolean indicating whether the HostView has padding.
  return hv.hasPadding

proc getLd*[D: static[int], T](hv: HostView[D, T]): array[D-1, int] =
  ## Get the leading dimensions of the HostView
  ##
  ## Parameters:
  ## - `hv`: The HostView instance.
  ##
  ## Returns:
  ## An array representing the leading dimensions.
  return hv.ld

#[ tests ]#

when isMainModule:
  block:
    var argc = cargc()
    var argv = cargv(argc)
    
    # Explicit MPI and GA initialization sequence
    # This allows proper shutdown without mpirun warnings
    initMPI(addr argc, addr argv)
    initGA()

    block:
      let lattice = [8, 8, 8, 8*GA_Nnodes()]
      let mpigrid = [1, 1, 1, GA_Nnodes()]
      let ghostgrid = [1, 1, 1, 1]
      var testGA1 = newGlobalArray(lattice, mpigrid, ghostgrid): float
      var hv1 = testGA1.hostView()

      for n in 0..<hv1.numSites():
        let coords = n.flatToCoords(hv1.paddedGrid)
        hv1[coords] = float(n) * 1.5
        assert hv1[coords] == hv1[n]

    # All GlobalArrays are now destroyed, safe to finalize
    finalizeGA()
    finalizeMPI()
