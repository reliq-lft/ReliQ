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

import lattice
import localtensor
import globaltensor
import sitetensor

import opencl/[clwrap, clbase, cldisp]
import utils/[complex]

export cldisp
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
  buffers*: seq[PMem]       
  queues*: seq[PCommandQueue]
  sitesPerDevice*: seq[int] 
  totalSites*: int          
  elementsPerSite*: int     ## Scalar elements per site (for OpenCL - complex counts as 2)
  tensorElementsPerSite*: int  ## Tensor elements per site (for memory - Complex64 counts as 1)
  elementSize*: int         ## sizeof(T)
  hostPtr*: pointer         
  hostOffsets*: seq[int]
  destroyed*: bool          ## Flag to prevent double destruction    

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

proc transformAoStoAoSoA*[T](src: pointer, numSites, elemsPerSite: int): seq[T] =
  ## Transform data from AoS (Array of Structures) to AoSoA layout.
  ## 
  ## AoS layout: site0[e0,e1,...], site1[e0,e1,...], ...
  ## AoSoA layout: group0[e0: s0,s1,...,sV-1, e1: s0,s1,...,sV-1, ...], group1[...]
  ## 
  ## Each vector group contains VectorWidth sites with elements interleaved.
  ## This enables SIMD-friendly memory access patterns.
  let numGroups = numVectorGroups(numSites)
  let paddedSites = numGroups * VectorWidth
  result = newSeq[T](paddedSites * elemsPerSite)
  
  let srcData = cast[ptr UncheckedArray[T]](src)
  
  for site in 0..<numSites:
    let g = site div VectorWidth      # vector group index
    let lane = site mod VectorWidth   # lane within group
    for e in 0..<elemsPerSite:
      # AoS index: site * elemsPerSite + e
      # AoSoA index: g * (VectorWidth * elemsPerSite) + e * VectorWidth + lane
      let aosIdx = site * elemsPerSite + e
      let aosoaIdx = g * (VectorWidth * elemsPerSite) + e * VectorWidth + lane
      result[aosoaIdx] = srcData[aosIdx]
  
  # Padding lanes (for partial last group) are left as zero-initialized

proc transformAoSoAtoAoS*[T](src: pointer, numSites, elemsPerSite: int): seq[T] =
  ## Transform data from AoSoA back to AoS layout.
  ## 
  ## This is used when reading device data back to host.
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

#[ constructors/destructors ]#

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
      buffers: newSeq[PMem](numDevices),
      queues: clQueues,
      sitesPerDevice: sitesPerDevice,
      totalSites: totalSites,
      elementsPerSite: elementsPerSite,  # scalar elements for OpenCL
      tensorElementsPerSite: tensorElementsPerSite,  # T elements for memory
      elementSize: elemSize,
      hostPtr: cast[pointer](tensor.data),
      hostOffsets: hostOffsets
    )
    
    var hostOffset = 0
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
        # Transform from AoS to AoSoA before uploading
        let srcPtr = cast[pointer](addr tensor.data[hostOffset])
        var aosoaData = transformAoStoAoSoA[T](srcPtr, numSites, tensorElementsPerSite)
        clQueues[deviceIdx].write(addr aosoaData[0], view.data.buffers[deviceIdx], paddedBufferSize)
        check clwrap.finish(clQueues[deviceIdx])
      of iokWrite: discard

      hostOffset += numTensorElements
    
    move(view)

template newTensorFieldView*[D: static[int], R: static[int], L, T](
  tensor: TensorField[D, R, L, T];
  io: IOKind
): TensorFieldView[L, T] =
  ## Create tensor field view from global tensor field
  tensor.newLocalTensorField().newTensorFieldView(io)

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

proc totalElements*[L, T](view: TensorFieldView[L, T]): int {.inline.} =
  ## Returns the total number of elements across all sites
  view.numSites * view.elementsPerSite

proc buffers*[L, T](view: TensorFieldView[L, T]): seq[PMem] =
  ## Returns the underlying device memory buffers
  view.data.buffers

proc sitesPerDevice*[L, T](view: TensorFieldView[L, T]): seq[int] =
  ## Returns the number of sites assigned to each device
  view.data.sitesPerDevice

# Phantom [] operators for TensorFieldView
# These are never called at runtime - they provide type info for OpenCL codegen
# The view's rank field determines interpretation:
#   rank=1 → vector field (SiteVec)
#   rank=2 → matrix field (SiteMat)  
#   else   → scalar field (T)

#[ TensorFieldView phantom operators for OpenCL codegen ]#
# These operators interact with TensorFieldView and produce/consume proxy types
# defined in sitetensor.nim

# TensorFieldView[] returns a proxy
proc `[]`*[L, T](view: TensorFieldView[L, T], site: int): TensorSiteProxy[L, T] = 
  raise newException(Defect, "TensorFieldView[] phantom operator")

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

proc `=destroy`*[L, T](view: var TensorFieldView[L, T]) =
  ## Destructor for TensorFieldView
  ## Writes device data back to host if ioKind is iokWrite or iokReadWrite,
  ## transforms from AoSoA back to AoS layout, then releases all device buffers.
  if view.data.destroyed:
    return  # Already destroyed, skip
  if view.data.buffers.len > 0:
    case view.ioKind:
      of iokWrite, iokReadWrite:
        if not view.data.hostPtr.isNil:
          for deviceIdx in 0..<view.data.buffers.len:
            let buf = view.data.buffers[deviceIdx]
            if not buf.isNil:
              let numSites = view.data.sitesPerDevice[deviceIdx]
              let tensorElementsPerSite = view.data.tensorElementsPerSite
              
              # Read padded AoSoA data from device
              let numGroups = numVectorGroups(numSites)
              let paddedSites = numGroups * VectorWidth
              let paddedElements = paddedSites * tensorElementsPerSite
              let paddedBufferSize = paddedElements * view.data.elementSize
              
              var aosoaData = newSeq[T](paddedElements)
              view.data.queues[deviceIdx].read(addr aosoaData[0], buf, paddedBufferSize)
              check clwrap.finish(view.data.queues[deviceIdx])
              
              # Transform AoSoA back to AoS
              let aosData = transformAoSoAtoAoS[T](addr aosoaData[0], numSites, tensorElementsPerSite)
              
              # Copy to host buffer
              let destPtr = cast[ptr UncheckedArray[T]](
                cast[pointer](cast[int](view.data.hostPtr) + view.data.hostOffsets[deviceIdx])
              )
              for i in 0..<(numSites * tensorElementsPerSite):
                destPtr[i] = aosData[i]
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
            localA.data[i] = float64(i)
            localB.data[i] = float64(i * 2)
            localC.data[i] = 0.0
          
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
            check localC.data[i] == float64(i) + float64(i * 2)
        
        test "Scalar multiplication with TensorFieldView":
          var tensorA = testLattice.newTensorField([1]): float64
          var tensorB = testLattice.newTensorField([1]): float64
          
          var localA = tensorA.newLocalTensorField()
          var localB = tensorB.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          for i in 0..<numSites:
            localA.data[i] = float64(i)
            localB.data[i] = 0.0
          
          block:
            var tViewA = localA.newTensorFieldView(iokRead)
            var tViewB = localB.newTensorFieldView(iokReadWrite)
            
            # Execute kernel: B = A * 3.0
            for i in each 0..<tViewB.numSites():
              tViewB[i] = tViewA[i] * 3.0
          
          for i in 0..<numSites:
            check localB.data[i] == float64(i) * 3.0
        
        test "In-place update with TensorFieldView":
          var tensorA = testLattice.newTensorField([1]): float64
          
          var localA = tensorA.newLocalTensorField()
          
          let numSites = localA.localGrid[0] * localA.localGrid[1] * 
                         localA.localGrid[2] * localA.localGrid[3]
          
          for i in 0..<numSites:
            localA.data[i] = float64(i)
          
          block:
            var tViewA = localA.newTensorFieldView(iokReadWrite)
            
            # Execute kernel: A = A + 1.0
            for i in each 0..<tViewA.numSites():
              tViewA[i] = tViewA[i] + 1.0
          
          for i in 0..<numSites:
            check localA.data[i] == float64(i) + 1.0
        
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
            let base = site * 4
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
            let base = site * 4
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
            let base = site * 4
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
            let base = site * 4
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
            let base = site * 4
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
            let base = site * 4
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
            let base = site * 4
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
            let base = site * 4
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
            let baseA = site * 4
            let baseV = site * 2
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
            let base = site * 2
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
            let base = site * 4
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
            let base = site * 4
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
            let base = site * 4
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
            let base = site * 4
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
            let base = site * 4
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
            let base = site * 4
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
            let base = site * 4
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
            let base = site * 4
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
            let base = site * 9
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
            let base = site * 9
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
            let baseA = site * 9
            let baseV = site * 3
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
            let base = site * 3
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
            let base = site * 4
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
            let base = site * 4
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
            let base = site * 4
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
            let base = site * 4
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
            let base = site * 4
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
            let base = site * 4
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
            let base = site * 3
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
            let base = site * 3
            check localV.data[base + 0] == 1.0
            check localV.data[base + 1] == 2.0
            check localV.data[base + 2] == 3.0

        test "AoSoA vectorization verification":
          # This test verifies that the AoSoA layout is working correctly by
          # checking that different sites within the same vector group get
          # processed correctly. If lanes weren't working, we'd see incorrect
          # interleaving of results.
          
          # Use a small lattice that's a multiple of VectorWidth (8)
          let smallDims: array[4, int] = [2, 2, 2, 2]  # 16 sites = 2 vector groups
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
            let base = site * 2
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
            let base = site * 2
            check localB.data[base + 0] == float64(2 * site)
            check localB.data[base + 1] == float64(2 * (site + 100))
          
          # Additional check: verify vector groups
          echo "  VectorWidth = ", VectorWidth
          echo "  numSites = ", numSites
          echo "  numVectorGroups = ", numVectorGroups(numSites)

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
            let base = site * 4
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
            let base = site * 4
            check localR.data[base + 0] == 2.0
            check localR.data[base + 1] == 3.0
            check localR.data[base + 2] == 4.0
            check localR.data[base + 3] == 5.0

    # End of block - all tensor fields destroyed here before GA finalization