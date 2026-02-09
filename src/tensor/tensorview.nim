#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/types/tensorview.nim
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

## TensorFieldView - Device-Side Views for Backend Dispatch
## ===========================================================
##
## This module provides `TensorFieldView[L,T]`, the type that the
## `each` macro operates on.  A view wraps a `LocalTensorField` in
## device-side buffers (OpenCL `cl_mem`, SYCL buffers, or raw host
## pointers for OpenMP) and handles the AoSoA layout transformation
## needed for efficient SIMD / GPU execution.
##
## Key capabilities:
##
## - **Construction**: `newTensorFieldView(local, ioKind)` allocates
##   device buffers and optionally synchronises data from the parent
##   local tensor (read/readwrite) or just allocates (write)
## - **AoSoA layout**: `transformToAoSoA` / `transformFromAoSoA` convert
##   between natural Array-of-Structures order and the blocked
##   Array-of-Structures-of-Arrays layout used by kernels
## - **Backend dispatch**: the `each` macro inspects view arguments at
##   compile time and emits OpenCL, SYCL, or OpenMP code accordingly
## - **Stencil integration**: views can be passed to `each` together
##   with a `LatticeStencil` for neighbor access in kernels
## - **Destruction**: on scope exit, write/readwrite views synchronise
##   data back to the parent local tensor field
##
## The backend is selected at compile time:
##
## ==========  ==========================  ====================
## Backend     Compile flag                Buffer type
## ==========  ==========================  ====================
## OpenCL      *(default)*                 ``cl_mem``
## SYCL        ``-d:UseSycl``              ``SyclBuffer``
## OpenMP      ``-d:UseOpenMP``            raw ``pointer``
## ==========  ==========================  ====================
##
## Example
## ^^^^^^^
##
## .. code-block:: nim
##   var local = field.newLocalTensorField()
##   var vSrc = local.newTensorFieldView(iokRead)
##   var vDst = local.newTensorFieldView(iokWrite)
##   each vDst, vSrc, n:
##     vDst[n] = 2.0 * vSrc[n]

import lattice
import localtensor
import globaltensor
import sitetensor
import lattice/stencil
export stencil

# SIMD layout infrastructure
import simd/simdlayout
export simdlayout

# Backend selection via compile-time flags
# Use -d:UseSycl for SYCL backend
# Use -d:UseOpenMP for OpenMP backend (CPU only)
# Default is OpenCL
const UseSycl* {.booldefine.} = false
const UseOpenMP* {.booldefine.} = false

when UseOpenMP:
  import openmp/[ompbase, ompdisp, ompsimd]
  export ompdisp, ompsimd
  # OpenMP uses host memory directly - no separate device buffers
  type
    BackendBuffer* = pointer
    BackendQueue* = pointer  # Dummy type for API compatibility
elif UseSycl:
  import sycl/[syclbase, sycldisp]
  export sycldisp
  # Type aliases for backend-agnostic code
  type
    BackendBuffer* = SyclBuffer
    BackendQueue* = SyclQueue
else:
  import opencl/[clwrap, clbase, cldisp]
  export cldisp
  # Type aliases for backend-agnostic code
  type
    BackendBuffer* = PMem
    BackendQueue* = PCommandQueue

import utils/[complex]

export sitetensor

#[ Vectorization: AoSoA layout for SIMD efficiency ]#

const VectorWidth* {.intdefine.} = 8
  ## Number of sites processed together in a vector group.
  ## Set via -d:VectorWidth=4 for AVX2, -d:VectorWidth=8 for AVX-512, etc.
  ## Default is 8 for good GPU and modern CPU performance.

when isMainModule:
  import std/[unittest]
  import parallel
  from lattice/simplecubiclattice import SimpleCubicLattice

type IOKind* = enum iokRead, iokWrite, iokReadWrite

type DeviceStorage* = object
  ## Device memory storage representation (type-erased)
  buffers*: seq[BackendBuffer]       
  queues*: seq[BackendQueue]
  sitesPerDevice*: seq[int] 
  totalSites*: int          
  elementsPerSite*: int     ## Scalar elements per site (for OpenCL - complex counts as 2)
  tensorElementsPerSite*: int  ## Tensor elements per site (for memory - Complex64 counts as 1)
  elementSize*: int         ## sizeof(T)
  hostPtr*: pointer         
  hostOffsets*: seq[int]
  siteOffsets*: seq[int]    ## Precomputed flat offsets for each lex site in padded GA memory
  destroyed*: bool          ## Flag to prevent double destruction
  simdLayout*: SimdLatticeLayout  ## SIMD layout for vectorized AoSoA access
  aosoaData*: pointer       ## Pointer to AoSoA transformed data (OpenMP)
  aosoaSeqRef*: RootRef     ## Reference to keep AoSoA seq alive (OpenMP)

type TensorFieldView*[L, T] = object
  ## Tensor field view on device memory
  ## 
  ## L is the lattice type, T is the scalar element type.
  ## Shape is stored at runtime to avoid static generic issues.
  ioKind*: IOKind
  lattice*: L
  dims*: int                # Number of lattice dimensions (was D)
  rank*: int                # Tensor rank (was R)
  shape*: seq[int]          # Tensor shape at each site
  data*: DeviceStorage
  hasPadding*: bool
  simdGrid*: seq[int]       ## Runtime SIMD lane grid (e.g., [1,2,2,2] for 8 lanes)

#[ constructor/destructor helpers ]#

proc computeElementsPerSite*(shape: openArray[int], isComplex: bool = false): int =
  ## Compute the number of elements per lattice site
  result = 1
  for dim in shape: result *= dim
  if isComplex: result *= 2

proc computeTotalLatticeSites*(localGrid: openArray[int]): int =
  ## Compute the total number of lattice sites from the local grid
  result = 1
  for dim in localGrid: result *= dim

proc splitLatticeSites*(totalSites: int, numDevices: int): seq[int] =
  ## Split lattice sites as evenly as possible among devices
  result = newSeq[int](numDevices)
  let baseSites = totalSites div numDevices
  let remainder = totalSites mod numDevices
  for i in 0..<numDevices:
    result[i] = baseSites
    if i < remainder: result[i] += 1

#[ AoSoA layout transformation functions ]#

proc numVectorGroups*(numSites: int): int {.inline.} =
  ## Compute number of vector groups (ceiling division)
  (numSites + VectorWidth - 1) div VectorWidth

proc transformAoStoAoSoA*[T](src: pointer, numSites, elemsPerSite: int, siteOffsets: seq[int]): seq[T] =
  ## Transform data from AoS (Array of Structures) to AoSoA layout.
  ## 
  ## AoS layout: site0[e0,e1,...], site1[e0,e1,...], ...
  ## AoSoA layout: group0[e0: s0,s1,...,sV-1, e1: s0,s1,...,sV-1, ...], group1[...]
  ## 
  ## Each vector group contains VectorWidth sites with elements interleaved.
  ## This enables SIMD-friendly memory access patterns.
  ## Uses siteOffsets to handle padded GA memory strides.
  ## Note: siteOffsets are in storage-type units (float64 for Complex64,
  ## float32 for Complex32), not in T units.
  let numGroups = numVectorGroups(numSites)
  let paddedSites = numGroups * VectorWidth
  result = newSeq[T](paddedSites * elemsPerSite)
  
  # For complex types, siteOffsets are in storage-type units (float64 for Complex64,
  # float32 for Complex32), but T is Complex64/Complex32 which is 2x the storage size.
  # We must read using the storage type and assemble complex values.
  when T is Complex64:
    let srcData = cast[ptr UncheckedArray[float64]](src)
    for site in 0..<numSites:
      let g = site div VectorWidth
      let lane = site mod VectorWidth
      let srcBase = siteOffsets[site]
      for e in 0..<elemsPerSite:
        let aosoaIdx = g * (VectorWidth * elemsPerSite) + e * VectorWidth + lane
        result[aosoaIdx] = complex64(srcData[srcBase + e * 2], srcData[srcBase + e * 2 + 1])
  elif T is Complex32:
    let srcData = cast[ptr UncheckedArray[float32]](src)
    for site in 0..<numSites:
      let g = site div VectorWidth
      let lane = site mod VectorWidth
      let srcBase = siteOffsets[site]
      for e in 0..<elemsPerSite:
        let aosoaIdx = g * (VectorWidth * elemsPerSite) + e * VectorWidth + lane
        result[aosoaIdx] = complex32(srcData[srcBase + e * 2], srcData[srcBase + e * 2 + 1])
  else:
    let srcData = cast[ptr UncheckedArray[T]](src)
    for site in 0..<numSites:
      let g = site div VectorWidth
      let lane = site mod VectorWidth
      let srcBase = siteOffsets[site]
      for e in 0..<elemsPerSite:
        let aosoaIdx = g * (VectorWidth * elemsPerSite) + e * VectorWidth + lane
        result[aosoaIdx] = srcData[srcBase + e]
  
  # Padding lanes (for partial last group) are left as zero-initialized

proc transformAoSoAtoAoS*[T](src: pointer, numSites, elemsPerSite: int): seq[T] =
  ## Transform data from AoSoA back to flat contiguous AoS layout.
  ## 
  ## This is used by ``updateGlobalTensorField`` for manual write-back.
  result = newSeq[T](numSites * elemsPerSite)
  
  let srcData = cast[ptr UncheckedArray[T]](src)
  
  for site in 0..<numSites:
    let g = site div VectorWidth      # vector group index
    let lane = site mod VectorWidth   # lane within group
    for e in 0..<elemsPerSite:
      # AoSoA index: g * (VectorWidth * elemsPerSite) + e * VectorWidth + lane
      # AoS index: site * elemsPerSite + e
      let aosoaIdx = g * (VectorWidth * elemsPerSite) + e * VectorWidth + lane
      let aosIdx = site * elemsPerSite + e
      result[aosIdx] = srcData[aosoaIdx]

#[ SIMD-aware AoSoA layout transformation functions ]#

proc transformAoStoAoSoASimd*[T](src: pointer, layout: SimdLatticeLayout, elemsPerSite: int, siteOffsets: seq[int]): seq[T] =
  ## Transform data from AoS to AoSoA layout using SIMD layout with configurable lane grid.
  ## 
  ## Unlike the simple VectorWidth-based AoSoA, this uses the SimdLatticeLayout to
  ## properly map lattice coordinates to SIMD lanes based on the user-specified simdGrid.
  ## Uses siteOffsets to handle padded GA memory strides.
  ##
  ## The AoSoA buffer always uses VectorWidth as the inner (lane) dimension so
  ## that SIMD loads/stores in the C kernel are contiguous and aligned.
  ## When ``nSitesInner < VectorWidth``, the extra lanes are zero-padded.
  ##
  ## AoS layout: site0[e0,e1,...], site1[e0,e1,...], ...
  ## AoSoA layout: outer0[e0: lane0..laneVW-1, e1: lane0..laneVW-1, ...], outer1[...]
  ##
  ## Parameters:
  ##   src: Pointer to GA memory (padded strides)
  ##   layout: SimdLatticeLayout with innerGeom (SIMD lanes) and outerGeom (vector groups)
  ##   elemsPerSite: Number of tensor elements per site (in T units)
  ##   siteOffsets: Precomputed flat offsets for each lexicographic site in padded GA memory
  ##               (in storage-type units, e.g. float64 for Complex64 fields)
  let vw = VectorWidth
  let nGroups = numVectorGroups(layout.nSites)
  let totalElements = nGroups * vw * elemsPerSite
  # Pad allocation to prevent GCC -mavx512f auto-vectorized 512-bit stores from
  # overflowing into adjacent heap objects.  64 bytes = one ZMM register width.
  let padElements = (64 + sizeof(T) - 1) div sizeof(T)
  result = newSeq[T](totalElements + padElements)
  
  # Fill only the valid lanes (0..<nSitesInner) — extra lanes stay zero.
  let srcData = cast[ptr UncheckedArray[T]](src)
  for outerIdx in 0..<layout.nSitesOuter:
    for lane in 0..<layout.nSitesInner:
      let localSite = outerInnerToLocal(outerIdx, lane, layout)
      let srcBase = siteOffsets[localSite]
      for e in 0..<elemsPerSite:
        # Use VW (not nSitesInner) as the lane stride so the kernel's
        # simd_load/simd_store accesses contiguous VW-wide blocks.
        let aosoaIdx = outerIdx * (elemsPerSite * vw) + e * vw + lane
        result[aosoaIdx] = srcData[srcBase + e]

when not UseOpenMP:
  # For non-OpenMP backends, provide local AoSoA↔AoS transform.
  # OpenMP gets this from ompsimd (imported above).
  proc transformAoSoAtoAoSSimd[T](src: pointer, layout: SimdLatticeLayout, elemsPerSite: int): seq[T] =
    ## Transform data from AoSoA back to flat contiguous AoS layout using SIMD layout.
    ##
    ## This is used by ``updateGlobalTensorField`` for manual write-back.
    ## Inverse of transformAoStoAoSoASimd.
    let totalElements = layout.nSites * elemsPerSite
    result = newSeq[T](totalElements)
    
    let srcData = cast[ptr UncheckedArray[T]](src)
    let vw = VectorWidth
    
    for outerIdx in 0..<layout.nSitesOuter:
      for lane in 0..<layout.nSitesInner:
        let localSite = outerInnerToLocal(outerIdx, lane, layout)
        for e in 0..<elemsPerSite:
          let aosoaIdx = outerIdx * (elemsPerSite * vw) + e * vw + lane
          let aosIdx = localSite * elemsPerSite + e
          result[aosIdx] = srcData[aosoaIdx]

proc defaultSimdGrid*[D: static[int]](localGrid: array[D, int]): seq[int] =
  ## Generate default SIMD grid that distributes VectorWidth lanes across dimensions.
  ## Prefers faster-varying (lower index) dimensions.
  result = newSeq[int](D)
  for d in 0..<D: result[d] = 1
  
  var remainingLanes = VectorWidth
  
  for d in 0..<D:
    if remainingLanes <= 1: break
    var lanesFordim = 1
    var candidate = 2
    while candidate <= remainingLanes and candidate <= localGrid[d]:
      if localGrid[d] mod candidate == 0 and remainingLanes mod candidate == 0:
        lanesFordim = candidate
      candidate *= 2
    result[d] = lanesFordim
    remainingLanes = remainingLanes div lanesFordim

proc defaultSimdGrid*(localGrid: seq[int]): seq[int] =
  ## Generate default SIMD grid that distributes VectorWidth lanes across dimensions.
  ## Overload for seq input.
  let D = localGrid.len
  result = newSeq[int](D)
  for d in 0..<D: result[d] = 1
  
  var remainingLanes = VectorWidth
  
  for d in 0..<D:
    if remainingLanes <= 1: break
    var lanesFordim = 1
    var candidate = 2
    while candidate <= remainingLanes and candidate <= localGrid[d]:
      if localGrid[d] mod candidate == 0 and remainingLanes mod candidate == 0:
        lanesFordim = candidate
      candidate *= 2
    result[d] = lanesFordim
    remainingLanes = remainingLanes div lanesFordim

#[ constructors/destructors ]#

when UseOpenMP:
  # OpenMP backend: works directly on host memory with SIMD-aware AoSoA layout
  
  template newTensorFieldView*[D: static[int], R: static[int], L, T](
    tensor: LocalTensorField[D, R, L, T];
    io: IOKind;
    simdGrid: openArray[int]
  ): TensorFieldView[L, T] =
    ## Create tensor field view from local tensor field (OpenMP backend) with custom SIMD grid.
    ##
    ## The simdGrid parameter specifies how SIMD lanes are distributed across dimensions.
    ## For example, simdGrid=[1,2,2,2] distributes 8 SIMD lanes as: 1 in dim 0, 2 in dims 1-3.
    ##
    ## Parameters:
    ##   tensor: Local tensor field to create view from
    ##   io: IOKind (iokRead, iokWrite, iokReadWrite)
    ##   simdGrid: SIMD lane grid per dimension (e.g., [1,2,2,2] for 8 lanes on 4D lattice)
    ##
    ## Example:
    ##   var view = localField.newTensorFieldView(iokRead, [1, 2, 2, 2])
    block:
      let totalSites = computeTotalLatticeSites(tensor.localGrid)
      let isComplex = isComplex32(T) or isComplex64(T)
      let elementsPerSite = computeElementsPerSite(tensor.shape, isComplex)
      let tensorElementsPerSite = computeElementsPerSite(tensor.shape, false)
      let elemSize = sizeof(T)
      
      # Create SIMD layout from local grid and user-specified simdGrid
      let layout = newSimdLatticeLayout(@(tensor.localGrid), @simdGrid)
      
      var view = TensorFieldView[L, T](
        ioKind: io,
        lattice: tensor.lattice,
        dims: D,
        rank: R,
        shape: @(tensor.shape),
        hasPadding: tensor.hasPadding,
        simdGrid: @simdGrid
      )
      
      # AoSoA buffer must use the storage scalar type (float64 for Complex64,
      # float32 for Complex32, T for real types) so that SIMD kernels can
      # load contiguous VW-wide lanes of the same scalar element.
      # For complex types this splits re and im into separate AoSoA elements.
      # Pad allocation to prevent GCC -mavx512f auto-vectorized 512-bit stores from
      # overflowing into adjacent heap objects.  64 bytes = one ZMM register width.
      when T is Complex64:
        type StorageScalar = float64
      elif T is Complex32:
        type StorageScalar = float32
      else:
        type StorageScalar = T
      
      type SeqHolder = ref object of RootObj
        data: seq[StorageScalar]
      
      var holder = SeqHolder()
      let padElements = (64 + sizeof(StorageScalar) - 1) div sizeof(StorageScalar)
      # Allocate VW-padded buffer: numVectorGroups * VW * elementsPerSite
      # so that partial last groups have valid (zeroed) lanes for SIMD access.
      let paddedSites = numVectorGroups(totalSites) * VectorWidth
      case io
      of iokRead, iokReadWrite:
        holder.data = transformAoStoAoSoASimd[StorageScalar](cast[pointer](tensor.data), layout, elementsPerSite, @(tensor.siteOffsets))
      of iokWrite:
        holder.data = newSeq[StorageScalar](paddedSites * elementsPerSite + padElements)
      
      view.data = DeviceStorage(
        buffers: @[cast[pointer](addr holder.data[0])],
        queues: @[nil.BackendQueue],
        sitesPerDevice: @[totalSites],
        totalSites: totalSites,
        elementsPerSite: elementsPerSite,
        tensorElementsPerSite: tensorElementsPerSite,
        elementSize: elemSize,
        hostPtr: cast[pointer](tensor.data),
        hostOffsets: @[0],
        siteOffsets: @(tensor.siteOffsets),
        simdLayout: layout,
        aosoaData: cast[pointer](addr holder.data[0]),
        aosoaSeqRef: holder  # Keep the ref alive
      )
      
      move(view)
  
  template newTensorFieldView*[D: static[int], R: static[int], L, T](
    tensor: LocalTensorField[D, R, L, T];
    io: IOKind
  ): TensorFieldView[L, T] =
    ## Create tensor field view from local tensor field (OpenMP backend)
    ## Uses default SIMD grid based on VectorWidth for AoSoA vectorized layout.
    block:
      let totalSites = computeTotalLatticeSites(tensor.localGrid)
      let isComplex = isComplex32(T) or isComplex64(T)
      let elementsPerSite = computeElementsPerSite(tensor.shape, isComplex)
      let tensorElementsPerSite = computeElementsPerSite(tensor.shape, false)
      let elemSize = sizeof(T)
      
      # Compute default SIMD grid from VectorWidth and lattice dimensions
      let simdGrid = defaultSimdGrid(@(tensor.localGrid))
      
      # Create SIMD layout from local grid and computed simdGrid
      let layout = newSimdLatticeLayout(@(tensor.localGrid), simdGrid)
      
      var view = TensorFieldView[L, T](
        ioKind: io,
        lattice: tensor.lattice,
        dims: D,
        rank: R,
        shape: @(tensor.shape),
        hasPadding: tensor.hasPadding,
        simdGrid: simdGrid
      )
      
      # AoSoA buffer must use the storage scalar type (float64 for Complex64,
      # float32 for Complex32, T for real types) so that SIMD kernels can
      # load contiguous VW-wide lanes of the same scalar element.
      # For complex types this splits re and im into separate AoSoA elements.
      # Pad allocation to prevent GCC -mavx512f auto-vectorized 512-bit stores from
      # overflowing into adjacent heap objects.  64 bytes = one ZMM register width.
      when T is Complex64:
        type StorageScalar = float64
      elif T is Complex32:
        type StorageScalar = float32
      else:
        type StorageScalar = T
      
      type SeqHolder = ref object of RootObj
        data: seq[StorageScalar]
      
      var holder = SeqHolder()
      let padElements = (64 + sizeof(StorageScalar) - 1) div sizeof(StorageScalar)
      let paddedSites = numVectorGroups(totalSites) * VectorWidth
      case io
      of iokRead, iokReadWrite:
        holder.data = transformAoStoAoSoASimd[StorageScalar](cast[pointer](tensor.data), layout, elementsPerSite, @(tensor.siteOffsets))
      of iokWrite:
        holder.data = newSeq[StorageScalar](paddedSites * elementsPerSite + padElements)
      
      view.data = DeviceStorage(
        buffers: @[cast[pointer](addr holder.data[0])],
        queues: @[nil.BackendQueue],
        sitesPerDevice: @[totalSites],
        totalSites: totalSites,
        elementsPerSite: elementsPerSite,
        tensorElementsPerSite: tensorElementsPerSite,
        elementSize: elemSize,
        hostPtr: cast[pointer](tensor.data),
        hostOffsets: @[0],
        siteOffsets: @(tensor.siteOffsets),
        simdLayout: layout,
        aosoaData: cast[pointer](addr holder.data[0]),
        aosoaSeqRef: holder  # Keep the ref alive
      )
      
      move(view)

else:
  # OpenCL/SYCL backend: uses device memory with AoSoA layout
  template newTensorFieldView*[D: static[int], R: static[int], L, T](
    tensor: LocalTensorField[D, R, L, T];
    io: IOKind
  ): TensorFieldView[L, T] =
    ## Create tensor field view from local tensor field
    ## Uses AoSoA layout on device for SIMD-friendly access patterns.
    block:
      let numDevices = clQueues.len
      let totalSites = computeTotalLatticeSites(tensor.localGrid)
      let isComplex = isComplex32(T) or isComplex64(T)
      # elementsPerSite counts individual scalar values (floats) for OpenCL
      let elementsPerSite = computeElementsPerSite(tensor.shape, isComplex)
      # tensorElementsPerSite counts T elements (may be Complex64)
      let tensorElementsPerSite = computeElementsPerSite(tensor.shape, false)
      let sitesPerDevice = splitLatticeSites(totalSites, numDevices)
      let elemSize = sizeof(T)
      
      # For AoSoA, we allocate padded buffers (rounded up to VectorWidth)
      var hostOffsets = newSeq[int](numDevices)
      var offset = 0
      for i in 0..<numDevices:
        hostOffsets[i] = offset
        # Use tensor elements (T type) for byte offset calculation (original AoS offsets for host)
        offset += sitesPerDevice[i] * tensorElementsPerSite * elemSize

      var view = TensorFieldView[L, T](
        ioKind: io,
        lattice: tensor.lattice,
        dims: D,
        rank: R,
        shape: @(tensor.shape),
        hasPadding: tensor.hasPadding
      )
      
      view.data = DeviceStorage(
        buffers: newSeq[BackendBuffer](numDevices),
        queues: clQueues,
        sitesPerDevice: sitesPerDevice,
        totalSites: totalSites,
        elementsPerSite: elementsPerSite,  # scalar elements for OpenCL
        tensorElementsPerSite: tensorElementsPerSite,  # T elements for memory
        elementSize: elemSize,
        hostPtr: cast[pointer](tensor.data),
        hostOffsets: hostOffsets,
        siteOffsets: @(tensor.siteOffsets)
      )
      
      var siteStart = 0
      for deviceIdx in 0..<numDevices:
        let numSites = sitesPerDevice[deviceIdx]
        let numTensorElements = numSites * tensorElementsPerSite
        
        # For AoSoA, allocate padded buffer (rounded up to VectorWidth)
        let numGroups = numVectorGroups(numSites)
        let paddedSites = numGroups * VectorWidth
        let paddedElements = paddedSites * tensorElementsPerSite
        let paddedBufferSize = paddedElements * elemSize
        
        view.data.buffers[deviceIdx] = buffer[T](clContext, paddedElements)
        
        case io
        of iokRead, iokReadWrite:
          # Transform from padded GA to AoSoA before uploading
          # Build device-local siteOffsets slice
          let deviceOffsets = tensor.siteOffsets[siteStart ..< siteStart + numSites]
          var aosoaData = transformAoStoAoSoA[T](cast[pointer](tensor.data), numSites, tensorElementsPerSite, deviceOffsets)
          clQueues[deviceIdx].write(addr aosoaData[0], view.data.buffers[deviceIdx], paddedBufferSize)
          check finish(clQueues[deviceIdx])
        of iokWrite: discard

        siteStart += numSites
      
      move(view)

template newTensorFieldView*[D: static[int], R: static[int], L, T](
  tensor: TensorField[D, R, L, T];
  io: IOKind
): TensorFieldView[L, T] =
  ## Create tensor field view from global tensor field
  tensor.newLocalTensorField().newTensorFieldView(io)

template newScalarFieldView*[D: static[int], L, T](
  tensor: TensorField[D, 1, L, T];
  io: IOKind
): TensorFieldView[L, T] =
  ## Create scalar field view from global scalar field
  tensor.newLocalScalarField().newTensorFieldView(io)

when UseOpenMP:
  template newTensorFieldView*[D: static[int], R: static[int], L, T](
    tensor: TensorField[D, R, L, T];
    io: IOKind;
    simdGrid: openArray[int]
  ): TensorFieldView[L, T] =
    ## Create tensor field view from global tensor field with custom SIMD grid
    ##
    ## Parameters:
    ##   tensor: Global tensor field
    ##   io: IOKind (iokRead, iokWrite, iokReadWrite)
    ##   simdGrid: SIMD lane grid per dimension (e.g., [1,2,2,2] for 8 lanes)
    ##
    ## Example:
    ##   var view = field.newTensorFieldView(iokRead, [1, 2, 2, 2])
    tensor.newLocalTensorField().newTensorFieldView(io, simdGrid)

# Prevent copying of DeviceStorage to avoid double-free
proc `=copy`*(dest: var DeviceStorage, src: DeviceStorage) 
  {.error: "DeviceStorage cannot be copied".}

#[ accessor helpers ]#

proc numSites*[L, T](view: TensorFieldView[L, T]): int {.inline.} =
  ## Returns the total number of lattice sites in the tensor field view
  view.data.totalSites

proc elementsPerSite*[L, T](view: TensorFieldView[L, T]): int {.inline.} =
  ## Returns the number of elements per lattice site
  view.data.elementsPerSite

template all*[L, T](view: TensorFieldView[L, T]): untyped =
  ## Returns a range over all sites: ``0 ..< numSites``.
  ## Use with ``each`` loops: ``for n in each view.all:``
  0 ..< view.numSites()

proc totalElements*[L, T](view: TensorFieldView[L, T]): int {.inline.} =
  ## Returns the total number of elements across all sites
  view.numSites * view.elementsPerSite

proc buffers*[L, T](view: TensorFieldView[L, T]): seq[BackendBuffer] =
  ## Returns the underlying device memory buffers
  view.data.buffers

proc sitesPerDevice*[L, T](view: TensorFieldView[L, T]): seq[int] =
  ## Returns the number of sites assigned to each device
  view.data.sitesPerDevice

#[ SIMD Layout Accessors ]#

proc simdLayout*[L, T](view: TensorFieldView[L, T]): SimdLatticeLayout =
  ## Returns the SIMD layout for vectorized access
  view.data.simdLayout

proc nSitesInner*[L, T](view: TensorFieldView[L, T]): int {.inline.} =
  ## Returns the number of SIMD lanes (sites per vector group)
  view.data.simdLayout.nSitesInner

proc nSitesOuter*[L, T](view: TensorFieldView[L, T]): int {.inline.} =
  ## Returns the number of vector groups (outer loop iterations)
  view.data.simdLayout.nSitesOuter

proc hasSimdLayout*[L, T](view: TensorFieldView[L, T]): bool {.inline.} =
  ## Returns true if this view was created with a SIMD layout
  view.simdGrid.len > 0

proc aosoaDataPtr*[L, T](view: TensorFieldView[L, T]): ptr UncheckedArray[T] {.inline.} =
  ## Returns pointer to AoSoA data buffer (for SIMD views)
  cast[ptr UncheckedArray[T]](view.data.aosoaData)

#[ CPU fallback support for printing ]#

when UseOpenMP:
  proc readSiteData*[L, T](view: TensorFieldView[L, T], site: int): RuntimeSiteData[T] =
    ## Read tensor data for a single site (OpenMP backend).
    ## Handles both AoS (hostPtr) and AoSoA (aosoaData) layouts.
    result.shape = view.shape
    result.rank = view.rank
    
    let elemsPerSite = view.data.tensorElementsPerSite
    result.data = newSeq[T](elemsPerSite)
    
    if view.simdGrid.len > 0 and not view.data.aosoaData.isNil:
      # SIMD layout: read from AoSoA data using SIMD indexing
      let aosoaPtr = cast[ptr UncheckedArray[T]](view.data.aosoaData)
      let layout = view.data.simdLayout
      for e in 0..<elemsPerSite:
        let idx = aosoaIndexFromLocal(site, e, elemsPerSite, layout)
        result.data[e] = aosoaPtr[idx]
    else:
      # AoS layout: direct linear access to host memory
      let hostData = cast[ptr UncheckedArray[T]](view.data.hostPtr)
      let baseIdx = site * elemsPerSite
      for e in 0..<elemsPerSite:
        result.data[e] = hostData[baseIdx + e]

  proc formatSiteData*[L, T](view: TensorFieldView[L, T], site: int): string =
    ## Read and format tensor data for a single site.
    let data = readSiteData(view, site)
    return $data

  proc writeSiteElement*[L, T](view: TensorFieldView[L, T], site: int, elementIdx: int, value: T) =
    ## Write a single element at a specific site (OpenMP backend).
    ## Handles both AoS and AoSoA layouts.
    let elemsPerSite = view.data.tensorElementsPerSite
    if view.simdGrid.len > 0 and not view.data.aosoaData.isNil:
      let aosoaPtr = cast[ptr UncheckedArray[T]](view.data.aosoaData)
      let layout = view.data.simdLayout
      let idx = aosoaIndexFromLocal(site, elementIdx, elemsPerSite, layout)
      aosoaPtr[idx] = value
    else:
      let hostData = cast[ptr UncheckedArray[T]](view.data.hostPtr)
      let idx = site * elemsPerSite + elementIdx
      hostData[idx] = value

else:
  proc readSiteData*[L, T](view: TensorFieldView[L, T], site: int): RuntimeSiteData[T] =
    ## Read tensor data for a single site from device memory.
    ## Used for CPU fallback when print statements are present.
    result.shape = view.shape
    result.rank = view.rank
    
    let elemsPerSite = view.data.tensorElementsPerSite
    result.data = newSeq[T](elemsPerSite)
    
    # Find which device has this site
    var deviceIdx = 0
    var siteOffset = site
    for i in 0..<view.data.sitesPerDevice.len:
      if siteOffset < view.data.sitesPerDevice[i]:
        deviceIdx = i
        break
      siteOffset -= view.data.sitesPerDevice[i]
    
    let numSitesOnDev = view.data.sitesPerDevice[deviceIdx]
    let numGroups = numVectorGroups(numSitesOnDev)
    let paddedBufferSize = numGroups * VectorWidth * elemsPerSite * sizeof(T)
    
    # Read entire device buffer (small overhead for debug printing)
    var aosoaData = newSeq[T](numGroups * VectorWidth * elemsPerSite)
    read(view.data.queues[deviceIdx], addr(aosoaData[0]), 
                view.data.buffers[deviceIdx], paddedBufferSize)
    discard finish(view.data.queues[deviceIdx])
    
    # Transform back from AoSoA and extract this site's data
    let g = siteOffset div VectorWidth
    let lane = siteOffset mod VectorWidth
    for e in 0..<elemsPerSite:
      let aosoaIdx = g * (VectorWidth * elemsPerSite) + e * VectorWidth + lane
      result.data[e] = aosoaData[aosoaIdx]

  proc formatSiteData*[L, T](view: TensorFieldView[L, T], site: int): string =
    ## Read and format tensor data for a single site.
    ## Returns a nicely formatted string representation.
    let data = readSiteData(view, site)
    return $data

  proc writeSiteElement*[L, T](view: TensorFieldView[L, T], site: int, elementIdx: int, value: T) =
    ## Write a single element of a tensor at a specific site to device memory.
    ## Used for element-level writes like view[n][i,j] = value during CPU fallback.
    ## This is an immediate write that syncs with the device.
    
    # Find which device has this site
    var deviceIdx = 0
    var siteOffset = site
    for i in 0..<view.data.sitesPerDevice.len:
      if siteOffset < view.data.sitesPerDevice[i]:
        deviceIdx = i
        break
      siteOffset -= view.data.sitesPerDevice[i]
    
    let elemsPerSite = view.data.tensorElementsPerSite
    let numSitesOnDev = view.data.sitesPerDevice[deviceIdx]
    let numGroups = numVectorGroups(numSitesOnDev)
    let paddedBufferSize = numGroups * VectorWidth * elemsPerSite * sizeof(T)
    
    # Read entire device buffer
    var aosoaData = newSeq[T](numGroups * VectorWidth * elemsPerSite)
    read(view.data.queues[deviceIdx], addr(aosoaData[0]), 
         view.data.buffers[deviceIdx], paddedBufferSize)
    discard finish(view.data.queues[deviceIdx])
    
    # Calculate AoSoA index and write the value
    let g = siteOffset div VectorWidth
    let lane = siteOffset mod VectorWidth
    let aosoaIdx = g * (VectorWidth * elemsPerSite) + elementIdx * VectorWidth + lane
    aosoaData[aosoaIdx] = value
    
    # Write back to device
    write(view.data.queues[deviceIdx], addr(aosoaData[0]), 
          view.data.buffers[deviceIdx], paddedBufferSize)
    discard finish(view.data.queues[deviceIdx])

# Phantom [] operators for TensorFieldView
# These are never called at runtime - they provide type info for OpenCL codegen
# The view's rank field determines interpretation:
#   rank=1 → vector field (SiteVec)
#   rank=2 → matrix field (SiteMat)  
#   else   → scalar field (T)

#[ TensorFieldView operators ]#
# For OpenMP: real implementations that access host memory directly
# For OpenCL/SYCL: phantom operators for kernel codegen

when UseOpenMP:
  # Helper to compute AoSoA element offset from local site index
  proc aosoaOffset[L, T](view: TensorFieldView[L, T], site: int, elemIdx: int): int {.inline.} =
    ## Convert (site, elemIdx) to offset in AoSoA buffer
    ## For a view with SIMD layout, uses outerIdx * nSitesInner * elemsPerSite + elemIdx * nSitesInner + innerIdx
    let layout = view.data.simdLayout
    let (outer, inner) = localToOuterInner(site, layout)
    let nSitesInner = layout.nSitesInner
    let elemsPerSite = view.data.tensorElementsPerSite
    result = outer * nSitesInner * elemsPerSite + elemIdx * nSitesInner + inner
  
  # Helper to get data pointer (AoSoA or AoS based on layout)
  proc getDataPtr[L, T](view: TensorFieldView[L, T]): ptr UncheckedArray[T] {.inline.} =
    if view.simdGrid.len > 0:
      cast[ptr UncheckedArray[T]](view.data.aosoaData)
    else:
      cast[ptr UncheckedArray[T]](view.data.hostPtr)
  
  # Helper to compute element offset (handles both AoS and AoSoA)
  proc elemOffset[L, T](view: TensorFieldView[L, T], site: int, elemIdx: int): int {.inline.} =
    if view.simdGrid.len > 0:
      view.aosoaOffset(site, elemIdx)
    else:
      site * view.data.tensorElementsPerSite + elemIdx
  
  # Helper to read element from proxy (handles both layouts)
  proc readProxyElem[L, T](proxy: TensorSiteProxy[L, T], elemIdx: int): T {.inline.} =
    let data = cast[ptr UncheckedArray[T]](proxy.hostPtr)
    if proxy.hasSimdLayout:
      let view = cast[ptr TensorFieldView[L, T]](proxy.view)
      data[view[].aosoaOffset(proxy.site, elemIdx)]
    else:
      data[proxy.site * proxy.elemsPerSite + elemIdx]
  
  # Helper to write element to view (handles both layouts)
  proc writeViewElem[L, T](view: TensorFieldView[L, T], site: int, elemIdx: int, value: T) {.inline.} =
    let data = view.getDataPtr()
    data[view.elemOffset(site, elemIdx)] = value

  # OpenMP backend: returns TensorSiteProxy for uniform API
  # TensorSiteProxy operators in sitetensor.nim have real implementations for OpenMP
  
  proc `[]`*[L, T](view: TensorFieldView[L, T], site: int): TensorSiteProxy[L, T] =
    ## Returns a proxy for site tensor access
    ## For OpenMP, the proxy stores view/site info for direct memory access
    result.view = cast[pointer](unsafeAddr view)
    result.site = site
    result.runtimeData = readSiteData(view, site)
    # OpenMP-specific: store info needed for direct element access
    # For SIMD layout, we store the AoSoA pointer so TensorSiteProxy knows to use SIMD indexing
    if view.simdGrid.len > 0:
      result.hostPtr = view.data.aosoaData
      result.hasSimdLayout = true
      result.simdLayoutPtr = cast[pointer](unsafeAddr view.data.simdLayout)
      result.nSitesInner = view.data.simdLayout.nSitesInner
    else:
      result.hostPtr = view.data.hostPtr
      result.hasSimdLayout = false
      result.simdLayoutPtr = nil
      result.nSitesInner = 0
    result.shape = view.shape
    result.elemsPerSite = view.data.tensorElementsPerSite
  
  proc `[]=`*[L, T](view: TensorFieldView[L, T], site: int, value: TensorSiteProxy[L, T]) {.inline.} =
    ## Copy site tensor from one view to another
    ## Handles both AoS and AoSoA layouts
    let elemsPerSite = view.data.tensorElementsPerSite
    
    if view.simdGrid.len > 0:
      # SIMD layout: use AoSoA indexing
      let dstData = cast[ptr UncheckedArray[T]](view.data.aosoaData)
      let srcData = cast[ptr UncheckedArray[T]](value.hostPtr)
      if value.hasSimdLayout:
        # Source also uses SIMD layout - compute both offsets
        let srcView = cast[ptr TensorFieldView[L, T]](value.view)
        for e in 0..<elemsPerSite:
          let dstOffset = view.aosoaOffset(site, e)
          let srcOffset = srcView[].aosoaOffset(value.site, e)
          dstData[dstOffset] = srcData[srcOffset]
      else:
        # Source uses AoS, dest uses AoSoA
        let srcBase = value.site * elemsPerSite
        for e in 0..<elemsPerSite:
          let dstOffset = view.aosoaOffset(site, e)
          dstData[dstOffset] = srcData[srcBase + e]
    else:
      # AoS layout: direct linear indexing
      let srcData = cast[ptr UncheckedArray[T]](value.hostPtr)
      let dstData = cast[ptr UncheckedArray[T]](view.data.hostPtr)
      let srcBase = value.site * elemsPerSite
      let dstBase = site * elemsPerSite
      for e in 0..<elemsPerSite:
        dstData[dstBase + e] = srcData[srcBase + e]

  proc `[]=`*[L, T](view: TensorFieldView[L, T], site: int, value: T) {.inline.} =
    ## Set scalar value at site (for scalar fields)
    if view.simdGrid.len > 0:
      let hostData = cast[ptr UncheckedArray[T]](view.data.aosoaData)
      hostData[view.aosoaOffset(site, 0)] = value
    else:
      let hostData = cast[ptr UncheckedArray[T]](view.data.hostPtr)
      hostData[site] = value
  
  proc `[]=`*[L, T](view: TensorFieldView[L, T], site: int, value: MatAddResult[L, T]) {.inline.} =
    ## Matrix/vector addition: C[n] = A[n] + B[n] or C[n] = A[n] - B[n]
    ## Handles both direct proxy access and computed intermediate results
    let elemsPerSite = view.data.tensorElementsPerSite
    
    # Handle computed buffers vs direct memory access
    if value.hasComputedA and value.hasComputedB:
      # Both operands are from computed results (e.g., A*B + C*D)
      if value.isSubtraction:
        for e in 0..<elemsPerSite:
          view.writeViewElem(site, e, value.computedA[e] - value.computedB[e])
      else:
        for e in 0..<elemsPerSite:
          view.writeViewElem(site, e, value.computedA[e] + value.computedB[e])
    elif value.hasComputedA and not value.hasComputedB:
      # A is computed, B is from proxy (e.g., (A*B + C*D) - E[n])
      if value.isSubtraction:
        for e in 0..<elemsPerSite:
          view.writeViewElem(site, e, value.computedA[e] - value.proxyB.readProxyElem(e))
      else:
        for e in 0..<elemsPerSite:
          view.writeViewElem(site, e, value.computedA[e] + value.proxyB.readProxyElem(e))
    elif not value.hasComputedA and value.hasComputedB:
      # A is from proxy, B is computed
      if value.isSubtraction:
        for e in 0..<elemsPerSite:
          view.writeViewElem(site, e, value.proxyA.readProxyElem(e) - value.computedB[e])
      else:
        for e in 0..<elemsPerSite:
          view.writeViewElem(site, e, value.proxyA.readProxyElem(e) + value.computedB[e])
    else:
      # Both from proxies (simple case: A[n] + B[n])
      if value.isSubtraction:
        for e in 0..<elemsPerSite:
          view.writeViewElem(site, e, value.proxyA.readProxyElem(e) - value.proxyB.readProxyElem(e))
      else:
        for e in 0..<elemsPerSite:
          view.writeViewElem(site, e, value.proxyA.readProxyElem(e) + value.proxyB.readProxyElem(e))
  
  proc `[]=`*[L, T](view: TensorFieldView[L, T], site: int, value: VecAddResult[L, T]) {.inline.} =
    ## Vector addition: same as MatAddResult
    let elemsPerSite = view.data.tensorElementsPerSite
    
    if value.hasComputedA and value.hasComputedB:
      if value.isSubtraction:
        for e in 0..<elemsPerSite:
          view.writeViewElem(site, e, value.computedA[e] - value.computedB[e])
      else:
        for e in 0..<elemsPerSite:
          view.writeViewElem(site, e, value.computedA[e] + value.computedB[e])
    elif value.hasComputedA and not value.hasComputedB:
      if value.isSubtraction:
        for e in 0..<elemsPerSite:
          view.writeViewElem(site, e, value.computedA[e] - value.proxyB.readProxyElem(e))
      else:
        for e in 0..<elemsPerSite:
          view.writeViewElem(site, e, value.computedA[e] + value.proxyB.readProxyElem(e))
    elif not value.hasComputedA and value.hasComputedB:
      if value.isSubtraction:
        for e in 0..<elemsPerSite:
          view.writeViewElem(site, e, value.proxyA.readProxyElem(e) - value.computedB[e])
      else:
        for e in 0..<elemsPerSite:
          view.writeViewElem(site, e, value.proxyA.readProxyElem(e) + value.computedB[e])
    else:
      if value.isSubtraction:
        for e in 0..<elemsPerSite:
          view.writeViewElem(site, e, value.proxyA.readProxyElem(e) - value.proxyB.readProxyElem(e))
      else:
        for e in 0..<elemsPerSite:
          view.writeViewElem(site, e, value.proxyA.readProxyElem(e) + value.proxyB.readProxyElem(e))
  
  proc `[]=`*[L, T](view: TensorFieldView[L, T], site: int, value: MatMulResult[L, T]) {.inline.} =
    ## Matrix multiplication: C[n] = A[n] * B[n]
    ## C[i,j] = sum_k(A[i,k] * B[k,j])
    let elemsPerSite = view.data.tensorElementsPerSite
    
    # Get matrix dimensions from shape
    let rows = view.shape[0]
    let cols = if view.shape.len > 1: view.shape[1] else: 1
    let innerDim = if value.proxyA.shape.len > 1: value.proxyA.shape[1] else: 1
    
    for i in 0..<rows:
      for j in 0..<cols:
        var sum: T = T(0)
        for k in 0..<innerDim:
          let aElemIdx = i * innerDim + k
          let bElemIdx = k * cols + j
          sum += value.proxyA.readProxyElem(aElemIdx) * value.proxyB.readProxyElem(bElemIdx)
        let dstElemIdx = i * cols + j
        view.writeViewElem(site, dstElemIdx, sum)
  
  proc `[]=`*[L, T](view: TensorFieldView[L, T], site: int, value: MatVecResult[L, T]) {.inline.} =
    ## Matrix-vector multiplication: v_out[n] = M[n] * v[n]
    ## v_out[i] = sum_j(M[i,j] * v[j])
    let rows = value.proxyMat.shape[0]
    let cols = if value.proxyMat.shape.len > 1: value.proxyMat.shape[1] else: 1
    
    for i in 0..<rows:
      var sum: T = T(0)
      for j in 0..<cols:
        let mElemIdx = i * cols + j
        sum += value.proxyMat.readProxyElem(mElemIdx) * value.proxyVec.readProxyElem(j)
      view.writeViewElem(site, i, sum)
  
  proc `[]=`*[L, T](view: TensorFieldView[L, T], site: int, value: ScalarMulResult[L, T]) {.inline.} =
    ## Scalar multiplication: C[n] = scalar * A[n]
    let elemsPerSite = view.data.tensorElementsPerSite
    for e in 0..<elemsPerSite:
      view.writeViewElem(site, e, value.scalar * value.proxy.readProxyElem(e))
  
  proc `[]=`*[L, T](view: TensorFieldView[L, T], site: int, value: ScalarAddResult[L, T]) {.inline.} =
    ## Scalar addition: C[n] = A[n] + scalar
    let elemsPerSite = view.data.tensorElementsPerSite
    for e in 0..<elemsPerSite:
      view.writeViewElem(site, e, value.proxy.readProxyElem(e) + value.scalar)

else:
  # OpenCL/SYCL backend: phantom operators for kernel codegen
  # These operators interact with TensorFieldView and produce/consume proxy types
  # defined in sitetensor.nim

  # TensorFieldView[] returns a proxy with runtime data for printing
  proc `[]`*[L, T](view: TensorFieldView[L, T], site: int): TensorSiteProxy[L, T] = 
    result.view = cast[pointer](unsafeAddr view)
    result.site = site
    result.runtimeData = readSiteData(view, site)

  # TensorFieldView[]= accepts proxy (copy), marker types, or scalar
  proc `[]=`*[L, T](view: TensorFieldView[L, T], site: int, value: TensorSiteProxy[L, T]) = 
    raise newException(Defect, "TensorFieldView[]= TensorSiteProxy phantom operator")

  proc `[]=`*[L, T](view: TensorFieldView[L, T], site: int, value: T) = 
    raise newException(Defect, "TensorFieldView[]= scalar phantom operator")

  proc `[]=`*[L, T](view: TensorFieldView[L, T], site: int, value: MatMulResult[L, T]) = 
    raise newException(Defect, "TensorFieldView[]= MatMulResult phantom operator")

  proc `[]=`*[L, T](view: TensorFieldView[L, T], site: int, value: MatAddResult[L, T]) = 
    raise newException(Defect, "TensorFieldView[]= MatAddResult phantom operator")

  proc `[]=`*[L, T](view: TensorFieldView[L, T], site: int, value: VecAddResult[L, T]) = 
    raise newException(Defect, "TensorFieldView[]= VecAddResult phantom operator")

  proc `[]=`*[L, T](view: TensorFieldView[L, T], site: int, value: MatVecResult[L, T]) = 
    raise newException(Defect, "TensorFieldView[]= MatVecResult phantom operator")

  proc `[]=`*[L, T](view: TensorFieldView[L, T], site: int, value: ScalarMulResult[L, T]) = 
    raise newException(Defect, "TensorFieldView[]= ScalarMulResult phantom operator")

  proc `[]=`*[L, T](view: TensorFieldView[L, T], site: int, value: ScalarAddResult[L, T]) = 
    raise newException(Defect, "TensorFieldView[]= ScalarAddResult phantom operator")

#[ Legacy marker functions - still supported ]#

proc matmul*[L, T](a, b: TensorFieldView[L, T], site: int): MatMulResult[L, T] {.inline.} =
  ## Matrix multiplication marker for OpenCL codegen
  ## Usage: mViewC[n] = matmul(mViewA, mViewB, n)
  ## Prefer using: mViewC[n] = mViewA[n] * mViewB[n]
  raise newException(Defect, "matmul is a phantom op for OpenCL codegen")

proc matadd*[L, T](a, b: TensorFieldView[L, T], site: int): MatAddResult[L, T] {.inline.} =
  ## Matrix addition marker for OpenCL codegen
  raise newException(Defect, "matadd is a phantom op for OpenCL codegen")

proc vecadd*[L, T](a, b: TensorFieldView[L, T], site: int): VecAddResult[L, T] {.inline.} =
  ## Vector addition marker for OpenCL codegen
  raise newException(Defect, "vecadd is a phantom op for OpenCL codegen")

proc matvec*[L, T](mat, vec: TensorFieldView[L, T], site: int): MatVecResult[L, T] {.inline.} =
  ## Matrix-vector multiplication marker for OpenCL codegen
  raise newException(Defect, "matvec is a phantom op for OpenCL codegen")

# Prevent copying of TensorFieldView to avoid double-free
proc `=copy`*[L, T](dest: var TensorFieldView[L, T], src: TensorFieldView[L, T]) 
  {.error: "TensorFieldView cannot be copied".}

when UseOpenMP:
  proc `=destroy`*[L, T](view: var TensorFieldView[L, T]) =
    ## Destructor for TensorFieldView (OpenMP backend)
    ## For SIMD views, transforms AoSoA data back to AoS and scatters
    ## into the padded GA memory via siteOffsets.  Because the host pointer
    ## points directly into GA memory, this write-back immediately updates
    ## the Global Array — no separate flush is needed.
    if view.data.destroyed:
      return  # Already destroyed, skip
    
    # Check if this is a SIMD view with AoSoA data
    if view.simdGrid.len > 0 and not view.data.aosoaData.isNil:
      case view.ioKind:
        of iokWrite, iokReadWrite:
          # Transform AoSoA back to flat contiguous AoS
          # For complex types the AoSoA buffer uses the storage scalar type
          # (float64 for Complex64, float32 for Complex32) with elementsPerSite
          # counting individual re/im scalars.
          let layout = view.data.simdLayout
          when T is Complex64:
            type StorageScalar = float64
          elif T is Complex32:
            type StorageScalar = float32
          else:
            type StorageScalar = T
          let elemsPerSite = view.data.elementsPerSite
          let aosData = transformAoSoAtoAoSSimd[StorageScalar](
            view.data.aosoaData, layout, elemsPerSite
          )
          
          # Scatter flat AoS into padded GA memory using siteOffsets
          # siteOffsets are in storage-scalar units, and the GA memory
          # also stores data as storage scalars, so this is a direct copy.
          let destPtr = cast[ptr UncheckedArray[StorageScalar]](view.data.hostPtr)
          let nSites = view.data.totalSites
          for site in 0..<nSites:
            let dstBase = view.data.siteOffsets[site]
            let srcBase = site * elemsPerSite
            for e in 0..<elemsPerSite:
              destPtr[dstBase + e] = aosData[srcBase + e]
        of iokRead:
          discard
    
    # Mark as destroyed
    view.data.destroyed = true

else:
  proc `=destroy`*[L, T](view: var TensorFieldView[L, T]) =
    ## Destructor for TensorFieldView (OpenCL/SYCL backend)
    ## Reads device data back, transforms from AoSoA to AoS, and scatters
    ## into the padded GA memory via siteOffsets.  Because the host pointer
    ## points directly into GA memory, this write-back immediately updates
    ## the Global Array — no separate flush is needed.
    ## Releases all device buffers after write-back.
    if view.data.destroyed:
      return  # Already destroyed, skip
    if view.data.buffers.len > 0:
      case view.ioKind:
        of iokWrite, iokReadWrite:
          if not view.data.hostPtr.isNil and view.data.queues.len > 0:
            var siteStart = 0
            for deviceIdx in 0..<view.data.buffers.len:
              let buf = view.data.buffers[deviceIdx]
              if not buf.isNil:
                let numSites = view.data.sitesPerDevice[deviceIdx]
                if numSites == 0: continue
                let tensorElementsPerSite = view.data.tensorElementsPerSite
                
                # Read padded AoSoA data from device
                let numGroups = numVectorGroups(numSites)
                let paddedSites = numGroups * VectorWidth
                let paddedElements = paddedSites * tensorElementsPerSite
                let paddedBufferSize = paddedElements * view.data.elementSize
                
                var aosoaData = newSeq[T](paddedElements)
                try:
                  let q = view.data.queues[deviceIdx]
                  if q.isNil:
                    discard  # Queue is nil, skip read-back
                  else:
                    q.read(addr aosoaData[0], buf, paddedBufferSize)
                    check finish(q)
                    check finish(q)
                except CatchableError:
                  discard  # Best-effort read-back during destruction
                except:
                  discard  # Handle any exception during destruction
                
                # Transform AoSoA back to flat contiguous AoS
                let aosData = transformAoSoAtoAoS[T](addr aosoaData[0], numSites, tensorElementsPerSite)
                
                # Scatter flat AoS into padded GA memory using siteOffsets
                # siteOffsets are in storage-type units (float64 for Complex64,
                # float32 for Complex32), not in T units.
                when T is Complex64:
                  let destData = cast[ptr UncheckedArray[float64]](view.data.hostPtr)
                  for site in 0..<numSites:
                    let dstBase = view.data.siteOffsets[siteStart + site]
                    let srcBase = site * tensorElementsPerSite
                    for e in 0..<tensorElementsPerSite:
                      destData[dstBase + e * 2] = aosData[srcBase + e].re
                      destData[dstBase + e * 2 + 1] = aosData[srcBase + e].im
                elif T is Complex32:
                  let destData = cast[ptr UncheckedArray[float32]](view.data.hostPtr)
                  for site in 0..<numSites:
                    let dstBase = view.data.siteOffsets[siteStart + site]
                    let srcBase = site * tensorElementsPerSite
                    for e in 0..<tensorElementsPerSite:
                      destData[dstBase + e * 2] = aosData[srcBase + e].re
                      destData[dstBase + e * 2 + 1] = aosData[srcBase + e].im
                else:
                  let destPtr = cast[ptr UncheckedArray[T]](view.data.hostPtr)
                  for site in 0..<numSites:
                    let dstBase = view.data.siteOffsets[siteStart + site]
                    let srcBase = site * tensorElementsPerSite
                    for e in 0..<tensorElementsPerSite:
                      destPtr[dstBase + e] = aosData[srcBase + e]
                
                siteStart += numSites
        of iokRead: discard
      
      # Release all buffers and set to nil to prevent double-free
      for i in 0..<view.data.buffers.len:
        let buf = view.data.buffers[i]
        if not buf.isNil:
          release(buf)
          view.data.buffers[i] = nil
    
    # Mark as destroyed
    view.data.destroyed = true

proc numGlobalSites*[L, T](view: TensorFieldView[L, T]): int {.inline.} =
  ## Returns the total number of global lattice sites across all MPI ranks
  result = 1
  for d in 0..<view.dims:
    result *= view.lattice.globalGrid[d]

when isMainModule:
  parallel:
    block: # All tensor fields must be destroyed before GA finalization
      let dims: array[4, int] = [8, 8, 8, 16]
      let testLattice = newSimpleCubicLattice(dims)

      var realTensorField1 = testLattice.newTensorField([3, 3]): float64
      var complexTensorField1 = testLattice.newTensorField([3, 3]): Complex64

      var localRealTensorField1 = realTensorField1.newLocalTensorField()
      var localComplexTensorField1 = complexTensorField1.newLocalTensorField()

      # transfer local tensor fields to device memory
      block:
        var deviceRealTensorView1 = localRealTensorField1.newTensorFieldView(iokRead)
        var deviceComplexTensorView1 = localComplexTensorField1.newTensorFieldView(iokRead)
        var deviceRealTensorView2 = realTensorField1.newTensorFieldView(iokWrite)
        var deviceComplexTensorView2 = complexTensorField1.newTensorFieldView(iokWrite)
      # device views go out of scope and are destroyed here

      suite "TensorFieldView each loop dispatch":
        
        test "Vector addition with TensorFieldView":
          # Create tensor fields for testing
          var tensorA = testLattice.newTensorField([1]): float64
          var tensorB = testLattice.newTensorField([1]): float64
          var tensorC = testLattice.newTensorField([1]): float64
          
          # Get local views
          var localA = tensorA.newLocalTensorField()
          var localB = tensorB.newLocalTensorField()
          var localC = tensorC.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          # Initialize host data
          for i in 0..<numSites:
            localA.data[localA.siteOffsets[i]] = float64(i)
            localB.data[localB.siteOffsets[i]] = float64(i * 2)
            localC.data[localC.siteOffsets[i]] = 0.0
          
          block:
            # Create device views
            var tViewA = localA.newTensorFieldView(iokRead)
            var tViewB = localB.newTensorFieldView(iokRead)
            var tViewC = localC.newTensorFieldView(iokReadWrite)
            
            # Execute kernel: C = A + B
            for i in each 0..<tViewC.numSites():
              tViewC[i] = tViewA[i] + tViewB[i]
          # Views destroyed here, tViewC writes back to localC
          
          # Verify results
          for i in 0..<numSites:
            check localC.data[localC.siteOffsets[i]] == float64(i) + float64(i * 2)
        
        test "Scalar multiplication with TensorFieldView":
          var tensorA = testLattice.newTensorField([1]): float64
          var tensorB = testLattice.newTensorField([1]): float64
          
          var localA = tensorA.newLocalTensorField()
          var localB = tensorB.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          for i in 0..<numSites:
            localA.data[localA.siteOffsets[i]] = float64(i)
            localB.data[localB.siteOffsets[i]] = 0.0
          
          block:
            var tViewA = localA.newTensorFieldView(iokRead)
            var tViewB = localB.newTensorFieldView(iokReadWrite)
            
            # Execute kernel: B = A * 3.0
            for i in each 0..<tViewB.numSites():
              tViewB[i] = tViewA[i] * 3.0
          
          for i in 0..<numSites:
            check localB.data[localB.siteOffsets[i]] == float64(i) * 3.0
        
        test "In-place update with TensorFieldView":
          var tensorA = testLattice.newTensorField([1]): float64
          
          var localA = tensorA.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          for i in 0..<numSites:
            localA.data[localA.siteOffsets[i]] = float64(i)
          
          block:
            var tViewA = localA.newTensorFieldView(iokReadWrite)
            
            # Execute kernel: A = A + 1.0
            for i in each 0..<tViewA.numSites():
              tViewA[i] = tViewA[i] + 1.0
          
          for i in 0..<numSites:
            check localA.data[localA.siteOffsets[i]] == float64(i) + 1.0
        
        test "numSites returns correct count":
          var tensorA = testLattice.newTensorField([1]): float64
          var localA = tensorA.newLocalTensorField()
          
          let expectedSites = localA.localGrid[0] * localA.localGrid[1] * 
                              localA.localGrid[2] * localA.localGrid[3]
          
          block:
            var tViewA = localA.newTensorFieldView(iokRead)
            check tViewA.numSites() == expectedSites
        
        test "Matrix multiplication with TensorFieldView":
          # Create 2x2 matrix fields for testing
          var tensorA = testLattice.newTensorField([2, 2]): float64
          var tensorB = testLattice.newTensorField([2, 2]): float64
          var tensorC = testLattice.newTensorField([2, 2]): float64
          
          var localA = tensorA.newLocalTensorField()
          var localB = tensorB.newLocalTensorField()
          var localC = tensorC.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          # Initialize: A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]] at each site
          # C = A * B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
          for site in 0..<numSites:
            let base = localA.siteOffsets[site]
            localA.data[base + 0] = 1.0  # [0,0]
            localA.data[base + 1] = 2.0  # [0,1]
            localA.data[base + 2] = 3.0  # [1,0]
            localA.data[base + 3] = 4.0  # [1,1]
            
            localB.data[base + 0] = 5.0  # [0,0]
            localB.data[base + 1] = 6.0  # [0,1]
            localB.data[base + 2] = 7.0  # [1,0]
            localB.data[base + 3] = 8.0  # [1,1]
            
            localC.data[base + 0] = 0.0
            localC.data[base + 1] = 0.0
            localC.data[base + 2] = 0.0
            localC.data[base + 3] = 0.0
          
          block:
            var mViewA = localA.newTensorFieldView(iokRead)
            var mViewB = localB.newTensorFieldView(iokRead)
            var mViewC = localC.newTensorFieldView(iokWrite)
            
            # Matrix multiplication using natural syntax: mat1[n] * mat2[n]
            for n in each 0..<mViewC.numSites():
              mViewC[n] = mViewA[n] * mViewB[n]
          
          # Verify results: C = A * B = [[19, 22], [43, 50]]
          for site in 0..<numSites:
            let base = localC.siteOffsets[site]
            check localC.data[base + 0] == 19.0  # [0,0]
            check localC.data[base + 1] == 22.0  # [0,1]
            check localC.data[base + 2] == 43.0  # [1,0]
            check localC.data[base + 3] == 50.0  # [1,1]
        
        test "Matrix addition with TensorFieldView":
          # Create 2x2 matrix fields: C = A + B
          var tensorA = testLattice.newTensorField([2, 2]): float64
          var tensorB = testLattice.newTensorField([2, 2]): float64
          var tensorC = testLattice.newTensorField([2, 2]): float64
          
          var localA = tensorA.newLocalTensorField()
          var localB = tensorB.newLocalTensorField()
          var localC = tensorC.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          # A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
          # C = A + B = [[6, 8], [10, 12]]
          for site in 0..<numSites:
            let base = localA.siteOffsets[site]
            localA.data[base + 0] = 1.0
            localA.data[base + 1] = 2.0
            localA.data[base + 2] = 3.0
            localA.data[base + 3] = 4.0
            localB.data[base + 0] = 5.0
            localB.data[base + 1] = 6.0
            localB.data[base + 2] = 7.0
            localB.data[base + 3] = 8.0
            localC.data[base + 0] = 0.0
            localC.data[base + 1] = 0.0
            localC.data[base + 2] = 0.0
            localC.data[base + 3] = 0.0
          
          block:
            var mViewA = localA.newTensorFieldView(iokRead)
            var mViewB = localB.newTensorFieldView(iokRead)
            var mViewC = localC.newTensorFieldView(iokWrite)
            
            for n in each 0..<mViewC.numSites():
              mViewC[n] = mViewA[n] + mViewB[n]
          
          for site in 0..<numSites:
            let base = localC.siteOffsets[site]
            check localC.data[base + 0] == 6.0
            check localC.data[base + 1] == 8.0
            check localC.data[base + 2] == 10.0
            check localC.data[base + 3] == 12.0
        
        test "Matrix subtraction with TensorFieldView":
          # Create 2x2 matrix fields: C = A - B
          var tensorA = testLattice.newTensorField([2, 2]): float64
          var tensorB = testLattice.newTensorField([2, 2]): float64
          var tensorC = testLattice.newTensorField([2, 2]): float64
          
          var localA = tensorA.newLocalTensorField()
          var localB = tensorB.newLocalTensorField()
          var localC = tensorC.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          # A = [[10, 20], [30, 40]], B = [[1, 2], [3, 4]]
          # C = A - B = [[9, 18], [27, 36]]
          for site in 0..<numSites:
            let base = localA.siteOffsets[site]
            localA.data[base + 0] = 10.0
            localA.data[base + 1] = 20.0
            localA.data[base + 2] = 30.0
            localA.data[base + 3] = 40.0
            localB.data[base + 0] = 1.0
            localB.data[base + 1] = 2.0
            localB.data[base + 2] = 3.0
            localB.data[base + 3] = 4.0
            localC.data[base + 0] = 0.0
            localC.data[base + 1] = 0.0
            localC.data[base + 2] = 0.0
            localC.data[base + 3] = 0.0
          
          block:
            var mViewA = localA.newTensorFieldView(iokRead)
            var mViewB = localB.newTensorFieldView(iokRead)
            var mViewC = localC.newTensorFieldView(iokWrite)
            
            for n in each 0..<mViewC.numSites():
              mViewC[n] = mViewA[n] - mViewB[n]
          
          for site in 0..<numSites:
            let base = localC.siteOffsets[site]
            check localC.data[base + 0] == 9.0
            check localC.data[base + 1] == 18.0
            check localC.data[base + 2] == 27.0
            check localC.data[base + 3] == 36.0
        
        test "Vector addition with rank-1 tensors":
          # Create length-4 vector fields: C = A + B
          var tensorA = testLattice.newTensorField([4]): float64
          var tensorB = testLattice.newTensorField([4]): float64
          var tensorC = testLattice.newTensorField([4]): float64
          
          var localA = tensorA.newLocalTensorField()
          var localB = tensorB.newLocalTensorField()
          var localC = tensorC.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          # A = [1, 2, 3, 4], B = [10, 20, 30, 40]
          # C = A + B = [11, 22, 33, 44]
          for site in 0..<numSites:
            let base = localA.siteOffsets[site]
            localA.data[base + 0] = 1.0
            localA.data[base + 1] = 2.0
            localA.data[base + 2] = 3.0
            localA.data[base + 3] = 4.0
            localB.data[base + 0] = 10.0
            localB.data[base + 1] = 20.0
            localB.data[base + 2] = 30.0
            localB.data[base + 3] = 40.0
            localC.data[base + 0] = 0.0
            localC.data[base + 1] = 0.0
            localC.data[base + 2] = 0.0
            localC.data[base + 3] = 0.0
          
          block:
            var vViewA = localA.newTensorFieldView(iokRead)
            var vViewB = localB.newTensorFieldView(iokRead)
            var vViewC = localC.newTensorFieldView(iokWrite)
            
            for n in each 0..<vViewC.numSites():
              vViewC[n] = vViewA[n] + vViewB[n]
          
          for site in 0..<numSites:
            let base = localC.siteOffsets[site]
            check localC.data[base + 0] == 11.0
            check localC.data[base + 1] == 22.0
            check localC.data[base + 2] == 33.0
            check localC.data[base + 3] == 44.0
        
        test "Matrix-vector multiplication":
          # C = A * v where A is 2x2, v is length-2, C is length-2
          # A = [[1, 2], [3, 4]], v = [5, 6]
          # C = [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
          var tensorA = testLattice.newTensorField([2, 2]): float64
          var tensorV = testLattice.newTensorField([2]): float64
          var tensorC = testLattice.newTensorField([2]): float64
          
          var localA = tensorA.newLocalTensorField()
          var localV = tensorV.newLocalTensorField()
          var localC = tensorC.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          for site in 0..<numSites:
            let baseA = localA.siteOffsets[site]
            let baseV = localV.siteOffsets[site]
            localA.data[baseA + 0] = 1.0  # [0,0]
            localA.data[baseA + 1] = 2.0  # [0,1]
            localA.data[baseA + 2] = 3.0  # [1,0]
            localA.data[baseA + 3] = 4.0  # [1,1]
            localV.data[baseV + 0] = 5.0
            localV.data[baseV + 1] = 6.0
            localC.data[baseV + 0] = 0.0
            localC.data[baseV + 1] = 0.0
          
          block:
            var mViewA = localA.newTensorFieldView(iokRead)
            var vViewV = localV.newTensorFieldView(iokRead)
            var vViewC = localC.newTensorFieldView(iokWrite)
            
            # Matrix-vector multiplication uses same * operator
            for n in each 0..<vViewC.numSites():
              vViewC[n] = mViewA[n] * vViewV[n]
          
          for site in 0..<numSites:
            let base = localC.siteOffsets[site]
            check localC.data[base + 0] == 17.0  # 1*5 + 2*6
            check localC.data[base + 1] == 39.0  # 3*5 + 4*6
        
        test "Scalar times matrix":
          # C = 2.0 * A
          var tensorA = testLattice.newTensorField([2, 2]): float64
          var tensorC = testLattice.newTensorField([2, 2]): float64
          
          var localA = tensorA.newLocalTensorField()
          var localC = tensorC.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          for site in 0..<numSites:
            let base = localA.siteOffsets[site]
            localA.data[base + 0] = 1.0
            localA.data[base + 1] = 2.0
            localA.data[base + 2] = 3.0
            localA.data[base + 3] = 4.0
            localC.data[base + 0] = 0.0
            localC.data[base + 1] = 0.0
            localC.data[base + 2] = 0.0
            localC.data[base + 3] = 0.0
          
          block:
            var mViewA = localA.newTensorFieldView(iokRead)
            var mViewC = localC.newTensorFieldView(iokWrite)
            
            for n in each 0..<mViewC.numSites():
              mViewC[n] = 2.0 * mViewA[n]
          
          for site in 0..<numSites:
            let base = localC.siteOffsets[site]
            check localC.data[base + 0] == 2.0
            check localC.data[base + 1] == 4.0
            check localC.data[base + 2] == 6.0
            check localC.data[base + 3] == 8.0
        
        test "Matrix times scalar":
          # C = A * 3.0
          var tensorA = testLattice.newTensorField([2, 2]): float64
          var tensorC = testLattice.newTensorField([2, 2]): float64
          
          var localA = tensorA.newLocalTensorField()
          var localC = tensorC.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          for site in 0..<numSites:
            let base = localA.siteOffsets[site]
            localA.data[base + 0] = 1.0
            localA.data[base + 1] = 2.0
            localA.data[base + 2] = 3.0
            localA.data[base + 3] = 4.0
            localC.data[base + 0] = 0.0
            localC.data[base + 1] = 0.0
            localC.data[base + 2] = 0.0
            localC.data[base + 3] = 0.0
          
          block:
            var mViewA = localA.newTensorFieldView(iokRead)
            var mViewC = localC.newTensorFieldView(iokWrite)
            
            for n in each 0..<mViewC.numSites():
              mViewC[n] = mViewA[n] * 3.0
          
          for site in 0..<numSites:
            let base = localC.siteOffsets[site]
            check localC.data[base + 0] == 3.0
            check localC.data[base + 1] == 6.0
            check localC.data[base + 2] == 9.0
            check localC.data[base + 3] == 12.0
        
        test "Scalar times vector":
          # C = 2.5 * v
          var tensorV = testLattice.newTensorField([4]): float64
          var tensorC = testLattice.newTensorField([4]): float64
          
          var localV = tensorV.newLocalTensorField()
          var localC = tensorC.newLocalTensorField()
          
          let numSites = localV.localGrid[0] * localV.localGrid[1] * 
                         localV.localGrid[2] * localV.localGrid[3]
          
          for site in 0..<numSites:
            let base = localV.siteOffsets[site]
            localV.data[base + 0] = 2.0
            localV.data[base + 1] = 4.0
            localV.data[base + 2] = 6.0
            localV.data[base + 3] = 8.0
            localC.data[base + 0] = 0.0
            localC.data[base + 1] = 0.0
            localC.data[base + 2] = 0.0
            localC.data[base + 3] = 0.0
          
          block:
            var vViewV = localV.newTensorFieldView(iokRead)
            var vViewC = localC.newTensorFieldView(iokWrite)
            
            for n in each 0..<vViewC.numSites():
              vViewC[n] = 2.5 * vViewV[n]
          
          for site in 0..<numSites:
            let base = localC.siteOffsets[site]
            check localC.data[base + 0] == 5.0
            check localC.data[base + 1] == 10.0
            check localC.data[base + 2] == 15.0
            check localC.data[base + 3] == 20.0
        
        test "Combined: (A * B) then add C":
          # D = A * B, then E = D + C using separate loops
          # A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]], C = [[1, 1], [1, 1]]
          # D = A*B = [[19, 22], [43, 50]]
          # E = D + C = [[20, 23], [44, 51]]
          var tensorA = testLattice.newTensorField([2, 2]): float64
          var tensorB = testLattice.newTensorField([2, 2]): float64
          var tensorC = testLattice.newTensorField([2, 2]): float64
          var tensorD = testLattice.newTensorField([2, 2]): float64
          var tensorE = testLattice.newTensorField([2, 2]): float64
          
          var localA = tensorA.newLocalTensorField()
          var localB = tensorB.newLocalTensorField()
          var localC = tensorC.newLocalTensorField()
          var localD = tensorD.newLocalTensorField()
          var localE = tensorE.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          for site in 0..<numSites:
            let base = localA.siteOffsets[site]
            localA.data[base + 0] = 1.0
            localA.data[base + 1] = 2.0
            localA.data[base + 2] = 3.0
            localA.data[base + 3] = 4.0
            localB.data[base + 0] = 5.0
            localB.data[base + 1] = 6.0
            localB.data[base + 2] = 7.0
            localB.data[base + 3] = 8.0
            localC.data[base + 0] = 1.0
            localC.data[base + 1] = 1.0
            localC.data[base + 2] = 1.0
            localC.data[base + 3] = 1.0
          
          block:
            var mViewA = localA.newTensorFieldView(iokRead)
            var mViewB = localB.newTensorFieldView(iokRead)
            var mViewD = localD.newTensorFieldView(iokWrite)
            
            for n in each 0..<mViewD.numSites():
              mViewD[n] = mViewA[n] * mViewB[n]
          
          block:
            var mViewC = localC.newTensorFieldView(iokRead)
            var mViewD = localD.newTensorFieldView(iokRead)
            var mViewE = localE.newTensorFieldView(iokWrite)
            
            for n in each 0..<mViewE.numSites():
              mViewE[n] = mViewD[n] + mViewC[n]
          
          for site in 0..<numSites:
            let base = localE.siteOffsets[site]
            check localE.data[base + 0] == 20.0
            check localE.data[base + 1] == 23.0
            check localE.data[base + 2] == 44.0
            check localE.data[base + 3] == 51.0
        
        test "3x3 matrix multiplication":
          # Test with larger matrices
          var tensorA = testLattice.newTensorField([3, 3]): float64
          var tensorB = testLattice.newTensorField([3, 3]): float64
          var tensorC = testLattice.newTensorField([3, 3]): float64
          
          var localA = tensorA.newLocalTensorField()
          var localB = tensorB.newLocalTensorField()
          var localC = tensorC.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          # A = [[1,0,0], [0,1,0], [0,0,1]] (identity)
          # B = [[1,2,3], [4,5,6], [7,8,9]]
          # C = A * B = B
          for site in 0..<numSites:
            let base = localA.siteOffsets[site]
            # Identity matrix A
            localA.data[base + 0] = 1.0
            localA.data[base + 1] = 0.0
            localA.data[base + 2] = 0.0
            localA.data[base + 3] = 0.0
            localA.data[base + 4] = 1.0
            localA.data[base + 5] = 0.0
            localA.data[base + 6] = 0.0
            localA.data[base + 7] = 0.0
            localA.data[base + 8] = 1.0
            # B
            localB.data[base + 0] = 1.0
            localB.data[base + 1] = 2.0
            localB.data[base + 2] = 3.0
            localB.data[base + 3] = 4.0
            localB.data[base + 4] = 5.0
            localB.data[base + 5] = 6.0
            localB.data[base + 6] = 7.0
            localB.data[base + 7] = 8.0
            localB.data[base + 8] = 9.0
          
          block:
            var mViewA = localA.newTensorFieldView(iokRead)
            var mViewB = localB.newTensorFieldView(iokRead)
            var mViewC = localC.newTensorFieldView(iokWrite)
            
            for n in each 0..<mViewC.numSites():
              mViewC[n] = mViewA[n] * mViewB[n]
          
          # C should equal B (since A is identity)
          for site in 0..<numSites:
            let base = localC.siteOffsets[site]
            check localC.data[base + 0] == 1.0
            check localC.data[base + 1] == 2.0
            check localC.data[base + 2] == 3.0
            check localC.data[base + 3] == 4.0
            check localC.data[base + 4] == 5.0
            check localC.data[base + 5] == 6.0
            check localC.data[base + 6] == 7.0
            check localC.data[base + 7] == 8.0
            check localC.data[base + 8] == 9.0
        
        test "3x3 matrix times 3-vector":
          var tensorA = testLattice.newTensorField([3, 3]): float64
          var tensorV = testLattice.newTensorField([3]): float64
          var tensorC = testLattice.newTensorField([3]): float64
          
          var localA = tensorA.newLocalTensorField()
          var localV = tensorV.newLocalTensorField()
          var localC = tensorC.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          # A = [[1,2,3], [4,5,6], [7,8,9]], v = [1, 1, 1]
          # C = A*v = [1+2+3, 4+5+6, 7+8+9] = [6, 15, 24]
          for site in 0..<numSites:
            let baseA = localA.siteOffsets[site]
            let baseV = localV.siteOffsets[site]
            localA.data[baseA + 0] = 1.0
            localA.data[baseA + 1] = 2.0
            localA.data[baseA + 2] = 3.0
            localA.data[baseA + 3] = 4.0
            localA.data[baseA + 4] = 5.0
            localA.data[baseA + 5] = 6.0
            localA.data[baseA + 6] = 7.0
            localA.data[baseA + 7] = 8.0
            localA.data[baseA + 8] = 9.0
            localV.data[baseV + 0] = 1.0
            localV.data[baseV + 1] = 1.0
            localV.data[baseV + 2] = 1.0
          
          block:
            var mViewA = localA.newTensorFieldView(iokRead)
            var vViewV = localV.newTensorFieldView(iokRead)
            var vViewC = localC.newTensorFieldView(iokWrite)
            
            for n in each 0..<vViewC.numSites():
              vViewC[n] = mViewA[n] * vViewV[n]
          
          for site in 0..<numSites:
            let base = localC.siteOffsets[site]
            check localC.data[base + 0] == 6.0
            check localC.data[base + 1] == 15.0
            check localC.data[base + 2] == 24.0
        
        test "Scalar add to matrix":
          # C = A + 10.0 (add scalar to each element)
          var tensorA = testLattice.newTensorField([2, 2]): float64
          var tensorC = testLattice.newTensorField([2, 2]): float64
          
          var localA = tensorA.newLocalTensorField()
          var localC = tensorC.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          for site in 0..<numSites:
            let base = localA.siteOffsets[site]
            localA.data[base + 0] = 1.0
            localA.data[base + 1] = 2.0
            localA.data[base + 2] = 3.0
            localA.data[base + 3] = 4.0
            localC.data[base + 0] = 0.0
            localC.data[base + 1] = 0.0
            localC.data[base + 2] = 0.0
            localC.data[base + 3] = 0.0
          
          block:
            var mViewA = localA.newTensorFieldView(iokRead)
            var mViewC = localC.newTensorFieldView(iokWrite)
            
            for n in each 0..<mViewC.numSites():
              mViewC[n] = mViewA[n] + 10.0
          
          for site in 0..<numSites:
            let base = localC.siteOffsets[site]
            check localC.data[base + 0] == 11.0
            check localC.data[base + 1] == 12.0
            check localC.data[base + 2] == 13.0
            check localC.data[base + 3] == 14.0
        
        test "Scalar add to vector":
          # C = 100.0 + v (scalar on left)
          var tensorV = testLattice.newTensorField([4]): float64
          var tensorC = testLattice.newTensorField([4]): float64
          
          var localV = tensorV.newLocalTensorField()
          var localC = tensorC.newLocalTensorField()
          
          let numSites = localV.localGrid[0] * localV.localGrid[1] * 
                         localV.localGrid[2] * localV.localGrid[3]
          
          for site in 0..<numSites:
            let base = localV.siteOffsets[site]
            localV.data[base + 0] = 1.0
            localV.data[base + 1] = 2.0
            localV.data[base + 2] = 3.0
            localV.data[base + 3] = 4.0
            localC.data[base + 0] = 0.0
            localC.data[base + 1] = 0.0
            localC.data[base + 2] = 0.0
            localC.data[base + 3] = 0.0
          
          block:
            var vViewV = localV.newTensorFieldView(iokRead)
            var vViewC = localC.newTensorFieldView(iokWrite)
            
            for n in each 0..<vViewC.numSites():
              vViewC[n] = 100.0 + vViewV[n]
          
          for site in 0..<numSites:
            let base = localC.siteOffsets[site]
            check localC.data[base + 0] == 101.0
            check localC.data[base + 1] == 102.0
            check localC.data[base + 2] == 103.0
            check localC.data[base + 3] == 104.0

        test "Element-level matrix write (set individual elements)":
          # Set individual matrix elements: mView[n][i,j] = value
          var tensorM = testLattice.newTensorField([2, 2]): float64
          
          var localM = tensorM.newLocalTensorField()
          
          let numSites = localM.localGrid[0] * localM.localGrid[1] * 
                         localM.localGrid[2] * localM.localGrid[3]
          
          # Initialize to zeros
          for site in 0..<numSites:
            let base = localM.siteOffsets[site]
            for i in 0..<4:
              localM.data[base + i] = 0.0
          
          block:
            var mView = localM.newTensorFieldView(iokReadWrite)
            
            # Set identity matrix at each site
            for n in each 0..<mView.numSites():
              mView[n][0, 0] = 1.0
              mView[n][0, 1] = 0.0
              mView[n][1, 0] = 0.0
              mView[n][1, 1] = 1.0
          
          # Verify identity matrices
          for site in 0..<numSites:
            let base = localM.siteOffsets[site]
            check localM.data[base + 0] == 1.0  # [0,0]
            check localM.data[base + 1] == 0.0  # [0,1]
            check localM.data[base + 2] == 0.0  # [1,0]
            check localM.data[base + 3] == 1.0  # [1,1]

        test "Element-level vector write (set individual elements)":
          # Set individual vector elements: vView[n][i] = value
          var tensorV = testLattice.newTensorField([3]): float64
          
          var localV = tensorV.newLocalTensorField()
          
          let numSites = localV.localGrid[0] * localV.localGrid[1] * 
                         localV.localGrid[2] * localV.localGrid[3]
          
          # Initialize to zeros
          for site in 0..<numSites:
            let base = localV.siteOffsets[site]
            for i in 0..<3:
              localV.data[base + i] = 0.0
          
          block:
            var vView = localV.newTensorFieldView(iokReadWrite)
            
            # Set [1, 2, 3] at each site
            for n in each 0..<vView.numSites():
              vView[n][0] = 1.0
              vView[n][1] = 2.0
              vView[n][2] = 3.0
          
          # Verify vectors
          for site in 0..<numSites:
            let base = localV.siteOffsets[site]
            check localV.data[base + 0] == 1.0
            check localV.data[base + 1] == 2.0
            check localV.data[base + 2] == 3.0

        test "AoSoA vectorization verification":
          # This test verifies that the AoSoA layout is working correctly by
          # checking that different sites within the same vector group get
          # processed correctly. If lanes weren't working, we'd see incorrect
          # interleaving of results.
          
          # Use a lattice large enough that each MPI rank has at least VectorWidth
          # sites after domain decomposition.  [4,4,4,4]=256 sites total;
          # with 4 ranks each rank gets 64 sites = 8 vector groups of width 8.
          let smallDims: array[4, int] = [4, 4, 4, 4]
          let smallLattice = newSimpleCubicLattice(smallDims)
          
          var tensorA = smallLattice.newTensorField([2]): float64  # 2-element vectors
          var tensorB = smallLattice.newTensorField([2]): float64
          
          var localA = tensorA.newLocalTensorField()
          var localB = tensorB.newLocalTensorField()
          
          # Get actual local sites (may be less than 16 with MPI distribution)
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          # Initialize A with site-specific values
          # Site n gets [n, n+100]
          for site in 0..<numSites:
            let base = localA.siteOffsets[site]
            localA.data[base + 0] = float64(site)
            localA.data[base + 1] = float64(site + 100)
            localB.data[base + 0] = 0.0
            localB.data[base + 1] = 0.0
          
          block:
            var vViewA = localA.newTensorFieldView(iokRead)
            var vViewB = localB.newTensorFieldView(iokWrite)
            
            # B = A * 2 (tests that each lane processes its own site correctly)
            for n in each 0..<vViewB.numSites():
              vViewB[n] = 2.0 * vViewA[n]
              # Debug: uncomment to see thread/work-group info
              # echo "group:", group_id, " local:", local_id, " gid:", gid, " site:", n
          
          # Verify: if AoSoA and lanes work correctly, B[site] = [2*site, 2*(site+100)]
          # If lanes were broken, we'd see scrambled results
          for site in 0..<numSites:
            let base = localB.siteOffsets[site]
            check localB.data[base + 0] == float64(2 * site)
            check localB.data[base + 1] == float64(2 * (site + 100))
          
          # Additional check: verify vector groups
          echo "  VectorWidth = ", VectorWidth
          echo "  numSites = ", numSites
          echo "  numVectorGroups = ", numVectorGroups(numSites)

        test "Print support with echo in each loop":
          # Test that echo statements trigger CPU fallback and $view[n] works
          var tensorM = testLattice.newTensorField([2, 2]): float64
          
          var localM = tensorM.newLocalTensorField()
          
          let numSites = localM.localGrid[0] * localM.localGrid[1] * 
                         localM.localGrid[2] * localM.localGrid[3]
          
          # Initialize to identity matrix at each site
          for site in 0..<numSites:
            let base = localM.siteOffsets[site]
            localM.data[base + 0] = 1.0  # [0,0]
            localM.data[base + 1] = 0.0  # [0,1]
            localM.data[base + 2] = 0.0  # [1,0]
            localM.data[base + 3] = 1.0  # [1,1]
          
          block:
            var mView = localM.newTensorFieldView(iokReadWrite)
            
            # This triggers CPU fallback due to echo statement
            # Only print first 2 sites to keep output manageable
            for n in each 0..<mView.numSites():
              if n < 2:
                echo "  Site ", n, " matrix:\n", $mView[n]
          
          # Verify data is still correct after CPU fallback loop
          for site in 0..<numSites:
            let base = localM.siteOffsets[site]
            check localM.data[base + 0] == 1.0
            check localM.data[base + 1] == 0.0
            check localM.data[base + 2] == 0.0
            check localM.data[base + 3] == 1.0

        test "Print support for vectors":
          # Test vector printing with $vView[n]
          var tensorV = testLattice.newTensorField([3]): float64
          
          var localV = tensorV.newLocalTensorField()
          
          let numSites = localV.localGrid[0] * localV.localGrid[1] * 
                         localV.localGrid[2] * localV.localGrid[3]
          
          # Initialize vector at each site to [1, 2, 3]
          for site in 0..<numSites:
            let base = localV.siteOffsets[site]
            localV.data[base + 0] = 1.0
            localV.data[base + 1] = 2.0
            localV.data[base + 2] = 3.0
          
          block:
            var vView = localV.newTensorFieldView(iokReadWrite)
            
            for n in each 0..<vView.numSites():
              if n < 2:
                echo "  Site ", n, " vector: ", $vView[n]
          
          # Verify data
          for site in 0..<numSites:
            let base = localV.siteOffsets[site]
            check localV.data[base + 0] == 1.0
            check localV.data[base + 1] == 2.0
            check localV.data[base + 2] == 3.0

        test "Long chain: A*B + C*D - E on single line":
          # Test complex single-line expression: R = A*B + C*D - E
          # All 2x2 matrices, computed in one kernel call
          # A = [[1,2],[3,4]], B = [[1,0],[0,1]] => A*B = [[1,2],[3,4]]
          # C = [[2,0],[0,2]], D = [[1,1],[1,1]] => C*D = [[2,2],[2,2]]
          # E = [[1,1],[1,1]]
          # R = A*B + C*D - E = [[1,2],[3,4]] + [[2,2],[2,2]] - [[1,1],[1,1]] = [[2,3],[4,5]]
          var tensorA = testLattice.newTensorField([2, 2]): float64
          var tensorB = testLattice.newTensorField([2, 2]): float64
          var tensorC = testLattice.newTensorField([2, 2]): float64
          var tensorD = testLattice.newTensorField([2, 2]): float64
          var tensorE = testLattice.newTensorField([2, 2]): float64
          var tensorR = testLattice.newTensorField([2, 2]): float64
          
          var localA = tensorA.newLocalTensorField()
          var localB = tensorB.newLocalTensorField()
          var localC = tensorC.newLocalTensorField()
          var localD = tensorD.newLocalTensorField()
          var localE = tensorE.newLocalTensorField()
          var localR = tensorR.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          for site in 0..<numSites:
            let base = localA.siteOffsets[site]
            localA.data[base + 0] = 1.0; localA.data[base + 1] = 2.0
            localA.data[base + 2] = 3.0; localA.data[base + 3] = 4.0
            localB.data[base + 0] = 1.0; localB.data[base + 1] = 0.0
            localB.data[base + 2] = 0.0; localB.data[base + 3] = 1.0
            localC.data[base + 0] = 2.0; localC.data[base + 1] = 0.0
            localC.data[base + 2] = 0.0; localC.data[base + 3] = 2.0
            localD.data[base + 0] = 1.0; localD.data[base + 1] = 1.0
            localD.data[base + 2] = 1.0; localD.data[base + 3] = 1.0
            localE.data[base + 0] = 1.0; localE.data[base + 1] = 1.0
            localE.data[base + 2] = 1.0; localE.data[base + 3] = 1.0
          
          block:
            var mViewA = localA.newTensorFieldView(iokRead)
            var mViewB = localB.newTensorFieldView(iokRead)
            var mViewC = localC.newTensorFieldView(iokRead)
            var mViewD = localD.newTensorFieldView(iokRead)
            var mViewE = localE.newTensorFieldView(iokRead)
            var mViewR = localR.newTensorFieldView(iokWrite)
            
            for n in each 0..<mViewR.numSites():
              mViewR[n] = mViewA[n] * mViewB[n] + mViewC[n] * mViewD[n] - mViewE[n]
          
          for site in 0..<numSites:
            let base = localR.siteOffsets[site]
            check localR.data[base + 0] == 2.0
            check localR.data[base + 1] == 3.0
            check localR.data[base + 2] == 4.0
            check localR.data[base + 3] == 5.0

        # ================================================================
        # Multi-type tests - float32, int32, int64 support
        # ================================================================

        test "Float32 vector addition":
          ## Test that float32 type works correctly for vector operations
          var tensorA = testLattice.newTensorField([3]): float32
          var tensorB = testLattice.newTensorField([3]): float32
          var tensorC = testLattice.newTensorField([3]): float32
          
          var localA = tensorA.newLocalTensorField()
          var localB = tensorB.newLocalTensorField()
          var localC = tensorC.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          for site in 0..<numSites:
            let base = localA.siteOffsets[site]
            localA.data[base + 0] = 1.0'f32
            localA.data[base + 1] = 2.0'f32
            localA.data[base + 2] = 3.0'f32
            localB.data[base + 0] = 0.5'f32
            localB.data[base + 1] = 1.0'f32
            localB.data[base + 2] = 1.5'f32
          
          block:
            var viewA = localA.newTensorFieldView(iokRead)
            var viewB = localB.newTensorFieldView(iokRead)
            var viewC = localC.newTensorFieldView(iokWrite)
            
            for n in each 0..<viewC.numSites():
              viewC[n] = viewA[n] + viewB[n]
          
          for site in 0..<numSites:
            let base = localC.siteOffsets[site]
            check localC.data[base + 0] == 1.5'f32
            check localC.data[base + 1] == 3.0'f32
            check localC.data[base + 2] == 4.5'f32

        test "Float32 scalar multiplication":
          ## Test scalar * vector with float32
          var tensorA = testLattice.newTensorField([2]): float32
          var tensorB = testLattice.newTensorField([2]): float32
          
          var localA = tensorA.newLocalTensorField()
          var localB = tensorB.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          for site in 0..<numSites:
            let base = localA.siteOffsets[site]
            localA.data[base + 0] = 2.0'f32
            localA.data[base + 1] = 4.0'f32
          
          block:
            var viewA = localA.newTensorFieldView(iokRead)
            var viewB = localB.newTensorFieldView(iokWrite)
            
            for n in each 0..<viewB.numSites():
              viewB[n] = 2.5'f32 * viewA[n]
          
          for site in 0..<numSites:
            let base = localB.siteOffsets[site]
            check localB.data[base + 0] == 5.0'f32
            check localB.data[base + 1] == 10.0'f32

        test "Float32 2x2 matrix multiplication":
          ## Test 2x2 matrix multiplication with float32
          # A = [[1, 2], [3, 4]], B = [[2, 0], [0, 2]] => C = [[2, 4], [6, 8]]
          var tensorA = testLattice.newTensorField([2, 2]): float32
          var tensorB = testLattice.newTensorField([2, 2]): float32
          var tensorC = testLattice.newTensorField([2, 2]): float32
          
          var localA = tensorA.newLocalTensorField()
          var localB = tensorB.newLocalTensorField()
          var localC = tensorC.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          for site in 0..<numSites:
            let base = localA.siteOffsets[site]
            localA.data[base + 0] = 1.0'f32; localA.data[base + 1] = 2.0'f32
            localA.data[base + 2] = 3.0'f32; localA.data[base + 3] = 4.0'f32
            localB.data[base + 0] = 2.0'f32; localB.data[base + 1] = 0.0'f32
            localB.data[base + 2] = 0.0'f32; localB.data[base + 3] = 2.0'f32
          
          block:
            var viewA = localA.newTensorFieldView(iokRead)
            var viewB = localB.newTensorFieldView(iokRead)
            var viewC = localC.newTensorFieldView(iokWrite)
            
            for n in each 0..<viewC.numSites():
              viewC[n] = viewA[n] * viewB[n]
          
          for site in 0..<numSites:
            let base = localC.siteOffsets[site]
            check localC.data[base + 0] == 2.0'f32
            check localC.data[base + 1] == 4.0'f32
            check localC.data[base + 2] == 6.0'f32
            check localC.data[base + 3] == 8.0'f32

        test "Int32 vector addition":
          ## Test int32 integer arithmetic on GPU
          var tensorA = testLattice.newTensorField([3]): int32
          var tensorB = testLattice.newTensorField([3]): int32
          var tensorC = testLattice.newTensorField([3]): int32
          
          var localA = tensorA.newLocalTensorField()
          var localB = tensorB.newLocalTensorField()
          var localC = tensorC.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          for site in 0..<numSites:
            let base = localA.siteOffsets[site]
            localA.data[base + 0] = 10'i32
            localA.data[base + 1] = 20'i32
            localA.data[base + 2] = 30'i32
            localB.data[base + 0] = 5'i32
            localB.data[base + 1] = 15'i32
            localB.data[base + 2] = 25'i32
          
          block:
            var viewA = localA.newTensorFieldView(iokRead)
            var viewB = localB.newTensorFieldView(iokRead)
            var viewC = localC.newTensorFieldView(iokWrite)
            
            for n in each 0..<viewC.numSites():
              viewC[n] = viewA[n] + viewB[n]
          
          for site in 0..<numSites:
            let base = localC.siteOffsets[site]
            check localC.data[base + 0] == 15'i32
            check localC.data[base + 1] == 35'i32
            check localC.data[base + 2] == 55'i32

        test "Int32 vector subtraction":
          ## Test int32 subtraction on GPU
          var tensorA = testLattice.newTensorField([2]): int32
          var tensorB = testLattice.newTensorField([2]): int32
          var tensorC = testLattice.newTensorField([2]): int32
          
          var localA = tensorA.newLocalTensorField()
          var localB = tensorB.newLocalTensorField()
          var localC = tensorC.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          for site in 0..<numSites:
            let base = localA.siteOffsets[site]
            localA.data[base + 0] = 100'i32
            localA.data[base + 1] = 50'i32
            localB.data[base + 0] = 30'i32
            localB.data[base + 1] = 20'i32
          
          block:
            var viewA = localA.newTensorFieldView(iokRead)
            var viewB = localB.newTensorFieldView(iokRead)
            var viewC = localC.newTensorFieldView(iokWrite)
            
            for n in each 0..<viewC.numSites():
              viewC[n] = viewA[n] - viewB[n]
          
          for site in 0..<numSites:
            let base = localC.siteOffsets[site]
            check localC.data[base + 0] == 70'i32
            check localC.data[base + 1] == 30'i32

        test "Int64 vector addition":
          ## Test int64 with large numbers on GPU
          var tensorA = testLattice.newTensorField([2]): int64
          var tensorB = testLattice.newTensorField([2]): int64
          var tensorC = testLattice.newTensorField([2]): int64
          
          var localA = tensorA.newLocalTensorField()
          var localB = tensorB.newLocalTensorField()
          var localC = tensorC.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          # Use values larger than int32 can represent
          let bigVal1: int64 = 3_000_000_000'i64  # > 2^31
          let bigVal2: int64 = 4_000_000_000'i64
          
          for site in 0..<numSites:
            let base = localA.siteOffsets[site]
            localA.data[base + 0] = bigVal1
            localA.data[base + 1] = 100'i64
            localB.data[base + 0] = bigVal2
            localB.data[base + 1] = 200'i64
          
          block:
            var viewA = localA.newTensorFieldView(iokRead)
            var viewB = localB.newTensorFieldView(iokRead)
            var viewC = localC.newTensorFieldView(iokWrite)
            
            for n in each 0..<viewC.numSites():
              viewC[n] = viewA[n] + viewB[n]
          
          for site in 0..<numSites:
            let base = localC.siteOffsets[site]
            check localC.data[base + 0] == bigVal1 + bigVal2  # 7_000_000_000
            check localC.data[base + 1] == 300'i64

        test "Int64 scalar multiplication":
          ## Test int64 scalar multiply on GPU
          var tensorA = testLattice.newTensorField([2]): int64
          var tensorB = testLattice.newTensorField([2]): int64
          
          var localA = tensorA.newLocalTensorField()
          var localB = tensorB.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          for site in 0..<numSites:
            let base = localA.siteOffsets[site]
            localA.data[base + 0] = 1_000_000'i64
            localA.data[base + 1] = 2_000_000'i64
          
          block:
            var viewA = localA.newTensorFieldView(iokRead)
            var viewB = localB.newTensorFieldView(iokWrite)
            
            for n in each 0..<viewB.numSites():
              viewB[n] = 3'i64 * viewA[n]
          
          for site in 0..<numSites:
            let base = localB.siteOffsets[site]
            check localB.data[base + 0] == 3_000_000'i64
            check localB.data[base + 1] == 6_000_000'i64

        test "Mixed precision not supported (separate fields)":
          ## Verify operations work with same types (no mixing)
          ## Float64 and float32 fields should both work independently
          var tensorF64 = testLattice.newTensorField([2]): float64
          var tensorF32 = testLattice.newTensorField([2]): float32
          
          var localF64 = tensorF64.newLocalTensorField()
          var localF32 = tensorF32.newLocalTensorField()
          
          let numSites = localF64.localGrid[0] * localF64.localGrid[1] * 
                         localF64.localGrid[2] * localF64.localGrid[3]
          
          for site in 0..<numSites:
            let base = localF64.siteOffsets[site]
            localF64.data[base + 0] = 1.5
            localF64.data[base + 1] = 2.5
            let baseF32 = localF32.siteOffsets[site]
            localF32.data[baseF32 + 0] = 1.5'f32
            localF32.data[baseF32 + 1] = 2.5'f32
          
          # Test float64 independently
          block:
            var viewA = localF64.newTensorFieldView(iokRead)
            var viewB = localF64.newTensorFieldView(iokReadWrite)
            for site in 0..<numSites:
              let base = localF64.siteOffsets[site]
              localF64.data[base + 0] = 0.0
              localF64.data[base + 1] = 0.0
            block:
              var viewBWrite = localF64.newTensorFieldView(iokWrite)
              for n in each 0..<viewBWrite.numSites():
                viewBWrite[n] = viewA[n]
          
          # Test float32 independently
          block:
            var viewA = localF32.newTensorFieldView(iokRead)
            var viewBWrite = localF32.newTensorFieldView(iokReadWrite)
            for n in each 0..<viewBWrite.numSites():
              viewBWrite[n] = 2.0'f32 * viewA[n]
          
          for site in 0..<numSites:
            let base = localF32.siteOffsets[site]
            check localF32.data[base + 0] == 3.0'f32
            check localF32.data[base + 1] == 5.0'f32

      #[ ============================================================================
         Stencil API Tests - Unified stencil operations for all backends
         ============================================================================ ]#
      suite "Stencil API with TensorFieldView":
        
        test "Stencil types are exported from tensorview":
          ## Verify stencil types are accessible via tensorview import
          let pattern = nearestNeighborStencil[4]()
          check pattern.nPoints == 8
          check pattern.points.len == 8
        
        test "LatticeStencil construction from lattice":
          ## Create stencil from the test lattice
          let stencil = newLatticeStencil(nearestNeighborStencil[4](), testLattice)
          check stencil.nSites > 0
          check stencil.nPoints == 8
          check stencil.nLocalSites > 0
        
        test "StencilShift type via shift/fwd/bwd API":
          ## Test the StencilShift type returned by shift(), fwd(), bwd()
          let stencil = newLatticeStencil(nearestNeighborStencil[4](), testLattice)
          
          let site = 0
          let shiftFwd = stencil.shift(site, +1, 0)  # Forward in x
          let shiftBwd = stencil.shift(site, -1, 0)  # Backward in x
          
          check shiftFwd.neighborIdx >= 0
          check shiftBwd.neighborIdx >= 0
          check shiftFwd.neighborIdx != shiftBwd.neighborIdx
          
          # fwd() and bwd() shorthands match shift()
          check stencil.fwd(site, 0).neighborIdx == shiftFwd.neighborIdx
          check stencil.bwd(site, 0).neighborIdx == shiftBwd.neighborIdx
        
        test "Stencil neighbor indices are within bounds":
          ## All neighbor indices should be within [0, nPaddedSites)
          let stencil = newLatticeStencil(nearestNeighborStencil[4](), testLattice)
          
          for site in 0..<stencil.nLocalSites:
            for p in 0..<stencil.nPoints:
              let nbrIdx = stencil.neighbor(site, p)
              check nbrIdx >= 0
              check nbrIdx < stencil.nPaddedSites
        
        test "Stencil iteration helpers":
          ## Test sites() and points() iterators
          let stencil = newLatticeStencil(nearestNeighborStencil[4](), testLattice)
          
          var siteCount = 0
          for site in stencil.sites:
            siteCount += 1
          check siteCount == stencil.nLocalSites
          
          var pointCount = 0
          for p in stencil.points:
            pointCount += 1
          check pointCount == stencil.nPoints
        
        test "Stencil shift index matches neighbor lookup":
          ## Verify shift().idx gives same result as neighbor()
          let stencil = newLatticeStencil(nearestNeighborStencil[4](), testLattice)
          
          for site in 0..<min(50, stencil.nSites):
            for dir in 0..<4:
              let fwdShift = stencil.fwd(site, dir)
              let fwdPoint = 2 * dir
              check fwdShift.idx == stencil.neighbor(site, fwdPoint)
              
              let bwdShift = stencil.bwd(site, dir)
              let bwdPoint = 2 * dir + 1
              check bwdShift.idx == stencil.neighbor(site, bwdPoint)
        
        test "Forward and backward stencil patterns":
          ## Test different stencil pattern constructors
          let fwdPattern = forwardStencil[4]()
          check fwdPattern.nPoints == 4
          
          let bwdPattern = backwardStencil[4]()
          check bwdPattern.nPoints == 4
          
          let fwdStencil = newLatticeStencil(fwdPattern, testLattice)
          let bwdStencil = newLatticeStencil(bwdPattern, testLattice)
          
          # All neighbor indices should be valid
          for site in 0..<min(100, fwdStencil.nSites):
            for p in 0..<4:
              check fwdStencil.neighbor(site, p) >= 0
              check fwdStencil.neighbor(site, p) < fwdStencil.nPaddedSites
              check bwdStencil.neighbor(site, p) >= 0
              check bwdStencil.neighbor(site, p) < bwdStencil.nPaddedSites
        
        test "Stencil forward != backward in each direction":
          ## For interior sites, forward and backward neighbors should differ
          let stencil = newLatticeStencil(nearestNeighborStencil[4](), testLattice)
          
          # Pick a site that's well in the interior
          if stencil.nSites > 0:
            let site = stencil.nSites div 2
            for dir in 0..<4:
              let fwd = stencil.fwd(site, dir)
              let bwd = stencil.bwd(site, dir)
              check fwd.idx != bwd.idx
        
        test "Stencil different directions give different neighbors":
          ## Forward neighbors in different directions should be different sites
          let stencil = newLatticeStencil(nearestNeighborStencil[4](), testLattice)
          
          if stencil.nSites > 0:
            let site = stencil.nSites div 2
            let fwd0 = stencil.fwd(site, 0).idx
            let fwd1 = stencil.fwd(site, 1).idx
            let fwd2 = stencil.fwd(site, 2).idx
            let fwd3 = stencil.fwd(site, 3).idx
            check fwd0 != fwd1
            check fwd0 != fwd2
            check fwd0 != fwd3
            check fwd1 != fwd2
            check fwd1 != fwd3
            check fwd2 != fwd3
        
        test "Stencil neighbor is self-inverse (periodic BC)":
          ## Going forward then backward should return to the same site
          ## This requires no ghost width (periodic wrapping within local domain)
          let stencil = newLatticeStencil(nearestNeighborStencil[4](), testLattice)
          
          for site in 0..<min(50, stencil.nSites):
            for dir in 0..<4:
              # Get forward neighbor index
              let fwdIdx = stencil.fwd(site, dir).idx
              # Get backward neighbor of that forward neighbor
              let backAgain = stencil.bwd(fwdIdx, dir).idx
              check backAgain == site
        
        test "Stencil with 2D lattice":
          ## Test stencil on a 2D lattice (fewer dimensions)
          let dims2D: array[2, int] = [8, 8]
          let lat2D = newSimpleCubicLattice(dims2D)
          
          let stencil2D = newLatticeStencil(nearestNeighborStencil[2](), lat2D)
          check stencil2D.nPoints == 4  # ±x, ±y
          check stencil2D.nSites > 0
          
          for site in 0..<stencil2D.nSites:
            for p in 0..<4:
              let nbr = stencil2D.neighbor(site, p)
              check nbr >= 0 and nbr < stencil2D.nPaddedSites
        
        test "Stencil with 3D lattice":
          ## Test stencil on a 3D lattice
          let dims3D: array[3, int] = [4, 4, 4]
          let lat3D = newSimpleCubicLattice(dims3D)
          
          let stencil3D = newLatticeStencil(nearestNeighborStencil[3](), lat3D)
          check stencil3D.nPoints == 6  # ±x, ±y, ±z
          check stencil3D.nSites > 0
          
          for site in 0..<stencil3D.nSites:
            for p in 0..<6:
              let nbr = stencil3D.neighbor(site, p)
              check nbr >= 0 and nbr < stencil3D.nPaddedSites
        
        test "Stencil neighbor copy via each loop":
          ## Copy from neighbor site using stencil.neighbor() inside each loop
          ## Works on all backends: the stencil.neighbor() call evaluates to an int
          ## which is then used as the view index
          var tensorSrc = testLattice.newTensorField([1]): float64
          var tensorDst = testLattice.newTensorField([1]): float64
          
          var localSrc = tensorSrc.newLocalTensorField()
          var localDst = tensorDst.newLocalTensorField()
          
          let numSites = localSrc.localGrid[0] * localSrc.localGrid[1] * 
                         localSrc.localGrid[2] * localSrc.localGrid[3]
          
          # Initialize source to 7.0, destination to 0.0
          for i in 0..<numSites:
            localSrc.data[localSrc.siteOffsets[i]] = 7.0
            localDst.data[localDst.siteOffsets[i]] = 0.0
          
          let stencil = newLatticeStencil(nearestNeighborStencil[4](), testLattice)
          
          block:
            var vSrc = localSrc.newTensorFieldView(iokRead)
            var vDst = localDst.newTensorFieldView(iokWrite)
            
            # Copy forward-x neighbor to destination
            for n in each 0..<vDst.numSites():
              let nbrIdx = stencil.neighbor(n, 0)
              vDst[n] = vSrc[nbrIdx]
          
          # All destinations should be 7.0 (copied from neighbor in constant field)
          for i in 0..<numSites:
            check localDst.data[localDst.siteOffsets[i]] == 7.0
        
        test "Stencil neighbor copy with matrix fields via each loop":
          ## Copy neighbor's 2x2 matrix via stencil in each loop
          var tensorSrc = testLattice.newTensorField([2, 2]): float64
          var tensorDst = testLattice.newTensorField([2, 2]): float64
          
          var localSrc = tensorSrc.newLocalTensorField()
          var localDst = tensorDst.newLocalTensorField()
          
          let numSites = localSrc.localGrid[0] * localSrc.localGrid[1] * 
                         localSrc.localGrid[2] * localSrc.localGrid[3]
          
          # Initialize source matrices to identity
          for site in 0..<numSites:
            let base = localSrc.siteOffsets[site]
            localSrc.data[base + 0] = 1.0
            localSrc.data[base + 1] = 0.0
            localSrc.data[base + 2] = 0.0
            localSrc.data[base + 3] = 1.0
          
          let stencil = newLatticeStencil(nearestNeighborStencil[4](), testLattice)
          
          block:
            var vSrc = localSrc.newTensorFieldView(iokRead)
            var vDst = localDst.newTensorFieldView(iokWrite)
            
            # Copy backward-t neighbor to destination
            for n in each 0..<vDst.numSites():
              let nbrIdx = stencil.neighbor(n, 7)  # Backward t is point 7
              vDst[n] = vSrc[nbrIdx]
          
          # Verify all destination matrices are identity
          for site in 0..<numSites:
            let base = localDst.siteOffsets[site]
            check localDst.data[base + 0] == 1.0
            check localDst.data[base + 1] == 0.0
            check localDst.data[base + 2] == 0.0
            check localDst.data[base + 3] == 1.0
        
        test "Stencil neighbor copy with vector fields via each loop":
          ## Copy neighbor's rank-1 tensor via stencil in each loop
          var tensorSrc = testLattice.newTensorField([3]): float64
          var tensorDst = testLattice.newTensorField([3]): float64
          
          var localSrc = tensorSrc.newLocalTensorField()
          var localDst = tensorDst.newLocalTensorField()
          
          let numSites = localSrc.localGrid[0] * localSrc.localGrid[1] * 
                         localSrc.localGrid[2] * localSrc.localGrid[3]
          
          # Initialize source vectors to (3, 6, 9)
          for site in 0..<numSites:
            let base = localSrc.siteOffsets[site]
            localSrc.data[base + 0] = 3.0
            localSrc.data[base + 1] = 6.0
            localSrc.data[base + 2] = 9.0
          
          let stencil = newLatticeStencil(nearestNeighborStencil[4](), testLattice)
          
          block:
            var vSrc = localSrc.newTensorFieldView(iokRead)
            var vDst = localDst.newTensorFieldView(iokWrite)
            
            # Copy forward-y neighbor
            for n in each 0..<vDst.numSites():
              let nbrIdx = stencil.neighbor(n, 2)  # Forward y is point 2
              vDst[n] = vSrc[nbrIdx]
          
          # Verify vectors copied correctly
          for site in 0..<numSites:
            let base = localDst.siteOffsets[site]
            check localDst.data[base + 0] == 3.0
            check localDst.data[base + 1] == 6.0
            check localDst.data[base + 2] == 9.0
        
        test "Stencil neighbor copy with float32 via each loop":
          ## Test stencil copy with float32 type
          var tensorSrc = testLattice.newTensorField([2]): float32
          var tensorDst = testLattice.newTensorField([2]): float32
          
          var localSrc = tensorSrc.newLocalTensorField()
          var localDst = tensorDst.newLocalTensorField()
          
          let numSites = localSrc.localGrid[0] * localSrc.localGrid[1] * 
                         localSrc.localGrid[2] * localSrc.localGrid[3]
          
          for site in 0..<numSites:
            let base = localSrc.siteOffsets[site]
            localSrc.data[base + 0] = 1.5'f32
            localSrc.data[base + 1] = 2.5'f32
          
          let stencil = newLatticeStencil(nearestNeighborStencil[4](), testLattice)
          
          block:
            var vSrc = localSrc.newTensorFieldView(iokRead)
            var vDst = localDst.newTensorFieldView(iokWrite)
            
            for n in each 0..<vDst.numSites():
              let nbrIdx = stencil.neighbor(n, 5)  # Backward z is point 5
              vDst[n] = vSrc[nbrIdx]
          
          for site in 0..<numSites:
            let base = localDst.siteOffsets[site]
            check localDst.data[base + 0] == 1.5'f32
            check localDst.data[base + 1] == 2.5'f32
        
        test "Stencil combined with arithmetic in each loop":
          ## Multiply neighbor value by scalar in each loop
          var tensorSrc = testLattice.newTensorField([1]): float64
          var tensorDst = testLattice.newTensorField([1]): float64
          
          var localSrc = tensorSrc.newLocalTensorField()
          var localDst = tensorDst.newLocalTensorField()
          
          let numSites = localSrc.localGrid[0] * localSrc.localGrid[1] * 
                         localSrc.localGrid[2] * localSrc.localGrid[3]
          
          for i in 0..<numSites:
            localSrc.data[localSrc.siteOffsets[i]] = 5.0
            localDst.data[localDst.siteOffsets[i]] = 0.0
          
          let stencil = newLatticeStencil(nearestNeighborStencil[4](), testLattice)
          
          block:
            var vSrc = localSrc.newTensorFieldView(iokRead)
            var vDst = localDst.newTensorFieldView(iokWrite)
            
            # dst[n] = 3.0 * src[neighbor(n)]
            for n in each 0..<vDst.numSites():
              let nbrIdx = stencil.neighbor(n, 0)
              vDst[n] = 3.0 * vSrc[nbrIdx]
          
          # 3.0 * 5.0 = 15.0
          for i in 0..<numSites:
            check localDst.data[localDst.siteOffsets[i]] == 15.0
        
        test "Stencil add neighbor and self in each loop":
          ## C[n] = A[n] + A[neighbor(n)]
          ## For constant field this should be 2*constant
          var tensorSrc = testLattice.newTensorField([1]): float64
          var tensorDst = testLattice.newTensorField([1]): float64
          
          var localSrc = tensorSrc.newLocalTensorField()
          var localDst = tensorDst.newLocalTensorField()
          
          let numSites = localSrc.localGrid[0] * localSrc.localGrid[1] * 
                         localSrc.localGrid[2] * localSrc.localGrid[3]
          
          for i in 0..<numSites:
            localSrc.data[localSrc.siteOffsets[i]] = 4.0
            localDst.data[localDst.siteOffsets[i]] = 0.0
          
          let stencil = newLatticeStencil(nearestNeighborStencil[4](), testLattice)
          
          block:
            var vSrc = localSrc.newTensorFieldView(iokRead)
            var vDst = localDst.newTensorFieldView(iokWrite)
            
            for n in each 0..<vDst.numSites():
              let nbrIdx = stencil.neighbor(n, 0)
              vDst[n] = vSrc[n] + vSrc[nbrIdx]
          
          # 4.0 + 4.0 = 8.0
          for i in 0..<numSites:
            check localDst.data[localDst.siteOffsets[i]] == 8.0

      #[ ============================================================================
         SIMD Vectorized TensorFieldView Tests (OpenMP backend)
         ============================================================================ ]#
      when UseOpenMP:
        suite "SIMD Vectorized TensorFieldView":
          
          test "Create SIMD view with custom lane grid [1,2,2,2]":
            ## Test creating a SIMD-vectorized view with user-specified lane grid
            var tensorA = testLattice.newTensorField([1]): float64
            var localA = tensorA.newLocalTensorField()
            
            let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                           localA.localGrid[2] * localA.localGrid[3]
            
            # Initialize data
            for i in 0..<numSites:
              localA.data[localA.siteOffsets[i]] = float64(i)
            
            # Create SIMD view with [1, 2, 2, 2] = 8 lanes
            block:
              let simdGrid = @[1, 2, 2, 2]
              var simdView = localA.newTensorFieldView(iokRead, simdGrid)
              
              check simdView.hasSimdLayout
              check simdView.nSitesInner == 8
              check simdView.nSitesOuter == numSites div 8
              check simdView.simdLayout.innerGeom == @[1, 2, 2, 2]
          
          test "SIMD view roundtrip preserves data":
            ## Test that data survives roundtrip through SIMD view (AoS -> AoSoA -> AoS)
            var tensorA = testLattice.newTensorField([2]): float64
            var localA = tensorA.newLocalTensorField()
            
            let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                           localA.localGrid[2] * localA.localGrid[3]
            let numElements = numSites * 2
            
            # Initialize with pattern (using siteOffsets for padded memory)
            for site in 0..<numSites:
              let base = localA.siteOffsets[site]
              localA.data[base + 0] = float64(site * 2 * 7 + 3)
              localA.data[base + 1] = float64((site * 2 + 1) * 7 + 3)
            
            # Save original values (using siteOffsets for padded memory)
            var originalData = newSeq[(float64, float64)](numSites)
            for site in 0..<numSites:
              let base = localA.siteOffsets[site]
              originalData[site] = (localA.data[base + 0], localA.data[base + 1])
            
            # Create SIMD view (transforms to AoSoA), then destroy (transforms back)
            block:
              let simdGrid = @[1, 2, 2, 2]
              var simdView = localA.newTensorFieldView(iokReadWrite, simdGrid)
              check simdView.hasSimdLayout
              # View will transform back to AoS on destruction
            
            # Verify data matches after roundtrip
            for site in 0..<numSites:
              let base = localA.siteOffsets[site]
              check localA.data[base + 0] == originalData[site][0]
              check localA.data[base + 1] == originalData[site][1]
          
          test "SIMD view AoSoA data transformation":
            ## Verify AoSoA layout actually rearranges data
            var tensorA = testLattice.newTensorField([1]): float64
            var localA = tensorA.newLocalTensorField()
            
            let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                           localA.localGrid[2] * localA.localGrid[3]
            
            # Initialize with site index
            for i in 0..<numSites:
              localA.data[localA.siteOffsets[i]] = float64(i)
            
            block:
              let simdGrid = @[1, 2, 2, 2]  # 8 lanes
              var simdView = localA.newTensorFieldView(iokRead, simdGrid)
              
              # Get AoSoA pointer
              let aosoaPtr = cast[ptr UncheckedArray[float64]](simdView.aosoaDataPtr)
              check aosoaPtr != nil
              
              # In AoSoA layout, the first 8 values should be the 8 sites
              # that form the first vector group. Verify they match the layout.
              let layout = simdView.simdLayout
              for lane in 0..<8:
                let localSite = outerInnerToLocal(0, lane, layout)
                # The value at aosoaPtr[lane] should equal the original value at localSite
                check aosoaPtr[lane] == float64(localSite)
          
          test "SIMD nSitesInner matches product of simdGrid":
            ## Verify nSitesInner calculation
            var tensorA = testLattice.newTensorField([3, 3]): float64
            var localA = tensorA.newLocalTensorField()
            
            let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                           localA.localGrid[2] * localA.localGrid[3]
            
            # simdGrid product must equal VectorWidth so AoSoA stride matches kernel
            block:
              let simdGrid = @[2, 2, 2, 1]  # 8 lanes = VectorWidth
              var simdView = localA.newTensorFieldView(iokRead, simdGrid)
              check simdView.nSitesInner == 8
              check simdView.nSitesOuter == numSites div 8
            
            block:
              let simdGrid = @[1, 1, 2, 4]  # 8 lanes = VectorWidth
              var simdView = localA.newTensorFieldView(iokRead, simdGrid)
              check simdView.nSitesInner == 8
              check simdView.nSitesOuter == numSites div 8

    # End of block - all tensor fields destroyed here before GA finalization