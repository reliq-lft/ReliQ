#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/memory/deviceview.nim
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

import std/[sequtils]

import hostview

import device/[platforms]
import globalarrays/[gatypes]
import utils/[private]
import lattice/[indexing]

nvidia: import cuda/[cudawrap]
amd: import hip/[hipwrap]
cpu: import simd/[simdtypes]

# Import vectorWidth constant
cpu:
  const vectorWidth* {.intdefine.} = 8  # Should match the Makefile setting

when isMainModule:
  import globalarrays/[gampi, gabase, gawrap]
  import utils/[commandline]

type DeviceView*[D: static[int], T] = object
  ## Represents a local array on device memory
  ##
  ## This object encapsulates a local array along with metadata about its
  ## globalGrid dimensions, MPI grid configuration, and ghost cell widths.
  ## 
  ## Fields:
  ## - `data`: A pointer to memory on the device of type `T` representing the local data.
  ## - `host`: A pointer to an UncheckedArray of type `T` representing the host-side data.
  ## - `globalGrid`: An array specifying the size of each dimension of the global array.
  ## - `mpiGrid`: An array specifying the distribution of the global array across MPI ranks.
  ## - `ghostGrid`: An array specifying the width of ghost cells for each dimension.
  ## 
  ## Note: The type parameter `T` indicates the data type of the elements in the local array.
  data: pointer
  host: ptr UncheckedArray[T]
  localGrid: array[D, int]
  paddedGrid: array[D, int]
  ghostGrid: array[D, int]
  lo, hi: array[D, int]
  ld: array[D-1, int]

#[ constructors ]#

proc deviceView*[D: static[int], T](hv: HostView[D, T]): DeviceView[D, T] =
  ## Constructor for DeviceView
  ##
  ## Creates a new DeviceView associated with the given HostView.
  ##
  ## Parameters:
  ## - `hv`: The HostView instance to associate with the local array.
  let (lo, hi) = hv.getBounds()
  result = DeviceView[D, T](
    host: hv.getData(),
    localGrid: hv.getLocalGrid(),
    paddedGrid: hv.getPaddedGrid(),
    ghostGrid: hv.getGhostGrid(),
    lo: lo,
    hi: hi,
    ld: hv.getLd()
  )

  nvidia:
    let size = sizeof(T) * result.paddedGrid.product()
    handleError cudaMalloc(addr result.data, size)
    handleError cudaMemcpy(result.data, addr result.host[][0], size, cudaMemcpyHostToDevice)
  amd:
    let size = sizeof(T) * result.paddedGrid.product()
    handleError hipMalloc(addr result.data, size)
    handleError hipMemcpy(result.data, addr result.host[][0], size, hipMemcpyHostToDevice)
  cpu: result.data = cast[pointer](result.host)

proc deviceView*[D: static[int], T](ga: GlobalArray[D, T]): DeviceView[D, T] =
  ## Constructor for DeviceView
  ##
  ## Creates a new DeviceView associated with the given GlobalArray.
  ##
  ## Parameters:
  ## - `ga`: The GlobalArray instance to associate with the local array.
  var hv = hostView(ga)
  return deviceView(hv)

proc hostView*[D: static[int], T](dv: DeviceView[D, T]): HostView[D, T] =
  ## Converts a DeviceView back to a HostView
  ##
  ## Parameters:
  ## - `dv`: The DeviceView instance to convert.
  ##
  ## Returns:
  ## A new instance of `HostView[D, T]`.
  nvidia: 
    let size = sizeof(T) * dv.paddedGrid.product()
    handleError cudaMemcpy(dv.data, addr dv.host[][0], size, cudaMemcpyDeviceToHost)
  amd: 
    let size = sizeof(T) * dv.paddedGrid.product()
    handleError hipMemcpy(dv.data, addr dv.host[][0], size, hipMemcpyDeviceToHost)

  return HostView[D, T](
    data: dv.host,
    localGrid: dv.localGrid,
    paddedGrid: dv.paddedGrid,
    lo: dv.lo,
    hi: dv.hi,
    ld: dv.ld,
    ghostGrid: dv.ghostGrid
  )

#[ move semantics ]#

proc `=destroy`*[D: static[int], T](dv: var DeviceView[D, T]) =
  ## Destructor for DeviceView
  ##
  ## Frees the device memory associated with the DeviceView.
  nvidia: cudaFree(dv.data)
  amd: hipFree(dv.data)
  dv.data = nil
  dv.host = nil

#[ accessors ]#

template `[]`*[D: static[int], T](dv: DeviceView[D, T]; idx: SomeInteger): untyped =
  ## Access an element in the DeviceView using flat indexing
  ##
  ## Parameters:
  ## - `idx`: The flat index specifying the location of the element.
  ##
  ## Returns:
  ## The element of type `T` at the specified flat index.
  nvidia: cast[ptr UncheckedArray[T]](dv.data)[idx]
  amd: cast[ptr UncheckedArray[T]](dv.data)[idx]
  cpu: newSIMD(addr cast[ptr UncheckedArray[T]](dv.data)[idx])

template `[]`*[D: static[int], T](dv: DeviceView[D, T], coords: array[D, int]): untyped =
  ## Access an element in the DeviceView using multi-dimensional coordinates
  ##
  ## Parameters:
  ## - `coords`: An array of coordinates specifying the location of the element.
  ##
  ## Returns:
  ## The element of type `T` at the specified coordinates.
  nvidia: cast[ptr UncheckedArray[T]](dv.data)[coords.coordsToFlat(dv.paddedGrid)]
  amd: cast[ptr UncheckedArray[T]](dv.data)[coords.coordsToFlat(dv.paddedGrid)]
  cpu: newSIMD(addr cast[ptr UncheckedArray[T]](dv.data)[coords.coordsToFlat(dv.paddedGrid)])

template `[]=`*[D: static[int], T](
  dv: var DeviceView[D, T]; 
  idx: SomeInteger; 
  value: untyped
) =
  ## Set an element in the DeviceView using flat indexing
  ##
  ## Parameters:
  ## - `idx`: The flat index specifying the location of the element.
  ## - `value`: The value of type `T` to assign at the specified index.
  cast[ptr UncheckedArray[T]](dv.data)[idx] = value

template `[]=`*[D: static[int], T](
  dv: var DeviceView[D, T]; 
  coords: array[D, int]; 
  value: untyped
) =
  ## Set an element in the DeviceView using multi-dimensional coordinates
  ##
  ## Parameters:
  ## - `coords`: An array of coordinates specifying the location of the element.
  ## - `value`: The value of type `T` to assign at the specified coordinates.
  cast[ptr UncheckedArray[T]](dv.data)[coords.coordsToFlat(dv.paddedGrid)] = value

proc numSites*[D: static[int], T](hv: DeviceView[D, T]): int =
  ## Get the number of local sites in the HostView
  ##
  ## Parameters:
  ## - `hv`: The HostView instance.
  ##
  ## Returns:
  ## The total number of local sites as an integer.
  return hv.paddedGrid.product()

proc numLocalSites*[D: static[int], T](hv: DeviceView[D, T]): int =
  ## Get the number of local sites excluding ghost cells in the HostView
  ##
  ## Parameters:
  ## - `hv`: The HostView instance.
  ##
  ## Returns:
  ## The total number of local sites excluding ghost cells as an integer.
  return hv.localGrid.product()

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
      var hv1 = testGA1.deviceView()

      cpu:
        for n in 0..<hv1.numSites() div vectorWidth:
          let coords = n.flatToCoords(hv1.paddedGrid)
          discard hv1[coords]# = float(n) * 1.5
          #assert hv1[coords] == hv1[n]
      
    # All GlobalArrays are now destroyed, safe to finalize
    finalizeGA()
    finalizeMPI()