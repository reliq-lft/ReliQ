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

import opencl/[clwrap, clbase]
import utils/[complex]

when isMainModule:
  import globaltensor
  import globalarrays/[gampi, gabase]
  import utils/[commandline]
  from lattice/simplecubiclattice import SimpleCubicLattice

type IOKind = enum AccRead, AccWrite, AccReadWrite

type DeviceStorage*[T] = object
  ## Device memory storage representation
  ##
  ## Represents data split across multiple device memory buffers.
  ## Each buffer holds a contiguous portion of the lattice sites,
  ## with tensor indices and data type indices remaining intact per site.
  buffers*: seq[PMem]       # OpenCL memory buffers, one per device
  sitesPerDevice*: seq[int] # Number of lattice sites assigned to each device
  elementsPerSite*: int     # Elements per lattice site (tensor shape * data type size)
  hostPtr*: pointer         # Pointer to host memory for write-back
  hostOffsets*: seq[int]    # Byte offsets into host memory for each device's portion

# Prevent copying of DeviceStorage to avoid double-free
proc `=copy`*[T](dest: var DeviceStorage[T], src: DeviceStorage[T]) {.error: "DeviceStorage cannot be copied".}

type TensorFieldView*[D: static[int], R: static[int], L, T] = object
  ## Local tensor field on device memory
  ## 
  ## Represents a local tensor field on device memory defined on a lattice with 
  ## specified dimensions and data type. The lattice sites are split among 
  ## available devices while keeping tensor and data type indices contiguous.
  ioKind: IOKind
  lattice*: L
  shape*: array[R, int]
  when isComplex32(T): data*: DeviceStorage[float32]
  elif isComplex64(T): data*: DeviceStorage[float64]
  else: data*: DeviceStorage[T]
  hasPadding: bool

# Prevent copying of TensorFieldView to avoid double-free
proc `=copy`*[D: static[int], R: static[int], L, T](
  dest: var TensorFieldView[D, R, L, T], 
  src: TensorFieldView[D, R, L, T]
) {.error: "TensorFieldView cannot be copied".}

#[ constructor/destructor helpers ]#

proc computeElementsPerSite*[R: static[int], T](shape: array[R, int]): int =
  ## Compute the number of elements per lattice site
  ##
  ## This includes tensor indices (middle) and data type indices (last).
  ## For complex types, the data type contributes a factor of 2.
  result = 1
  for dim in shape: result *= dim
  when isComplex32(T) or isComplex64(T): result *= 2

proc computeTotalLatticeSites*[D: static[int]](localGrid: array[D, int]): int =
  ## Compute the total number of lattice sites from the local grid
  result = 1
  for dim in localGrid: result *= dim

proc splitLatticeSites*(totalSites: int, numDevices: int): seq[int] =
  ## Split lattice sites as evenly as possible among devices
  ##
  ## Returns a sequence with the number of sites assigned to each device.
  ## Sites are distributed to minimize imbalance.
  result = newSeq[int](numDevices)
  let baseSites = totalSites div numDevices
  let remainder = totalSites mod numDevices
  
  for i in 0..<numDevices:
    result[i] = baseSites
    if i < remainder: result[i] += 1

#[ constructors/destructors ]#

template newDeviceStorage*[D: static[int], R: static[int], T](
  data: HostStorage[T],
  localGrid: array[D, int],
  shape: array[R, int],
  io: IOKind
): DeviceStorage[T] =
  ## Create device storage from host storage, splitting across devices
  ## 
  ## Allocates device memory on each available device and copies the 
  ## corresponding portion of host data. Lattice sites (leftmost indices)
  ## are split among devices, while tensor indices (middle) and data type
  ## indices (last) remain contiguous for each site.
  ##
  ## Uses global variables clContext and clQueues from initCL.
  let numDevices = clQueues.len
  let totalSites = computeTotalLatticeSites[D](localGrid)
  let elementsPerSite = computeElementsPerSite[R, T](shape)
  let sitesPerDevice = splitLatticeSites(totalSites, numDevices)
  
  # Compute host offsets for each device's portion
  var hostOffsets = newSeq[int](numDevices)
  var offset = 0
  for i in 0..<numDevices:
    hostOffsets[i] = offset
    offset += sitesPerDevice[i] * elementsPerSite * sizeof(T)
  
  var storage = DeviceStorage[T](
    buffers: newSeq[PMem](numDevices),
    sitesPerDevice: sitesPerDevice,
    elementsPerSite: elementsPerSite,
    hostPtr: cast[pointer](data),
    hostOffsets: hostOffsets
  )
  
  var hostOffset = 0
  for deviceIdx in 0..<numDevices:
    let numSites = sitesPerDevice[deviceIdx]
    let numElements = numSites * elementsPerSite
    let bufferSize = numElements * sizeof(T)
    
    # Allocate device buffer
    storage.buffers[deviceIdx] = buffer[T](clContext, numElements)
    
    # Copy data from host to device
    let srcPtr = cast[pointer](addr data[hostOffset])
    case io
    of AccRead, AccReadWrite:
      clQueues[deviceIdx].write(srcPtr, storage.buffers[deviceIdx], bufferSize)
    of AccWrite: discard

    hostOffset += numElements
  
  move(storage)

template newTensorFieldView*[D: static[int], R: static[int], L, T](
  tensor: LocalTensorField[D, R, L, T];
  io: IOKind
): TensorFieldView[D, R, L, T] =
  ## Create tensor field view
  ## 
  ## Creates a tensor field view on device memory from a local tensor field on 
  ## host memory by allocating device memory and copying data from host to device.
  ## The lattice sites are automatically split among available devices.
  ##
  ## Uses global variables clContext and clQueues from initCL.
  var data = tensor.data
  var localGrid = tensor.localGrid
  var shape = tensor.shape

  var view = TensorFieldView[D, R, L, T](
    ioKind: io,
    lattice: tensor.lattice,
    shape: shape,
    hasPadding: tensor.hasPadding
  )

  when isComplex32(T):
    view.data = newDeviceStorage[D, R, float32](data, localGrid, shape, io)
  elif isComplex64(T):
    view.data = newDeviceStorage[D, R, float64](data, localGrid, shape, io)
  else: view.data = newDeviceStorage[D, R, T](data, localGrid, shape, io)
  
  move(view)

template newTensorFieldView*[D: static[int], R: static[int], L, T](
  tensor: TensorField[D, R, L, T];
  io: IOKind
): TensorFieldView[D, R, L, T] =
  ## Create tensor field view from global tensor field
  ##
  ## Creates a tensor field view on device memory from a global tensor field by 
  ## first downcasting to a local tensor field on host memory, then allocating 
  ## device memory and copying data from host to device. The lattice sites are 
  ## automatically split among available devices.
  ##
  ## Uses global variables clContext and clQueues from initCL.
  tensor.newLocalTensorField().newTensorFieldView(io)

template `=destroy`*[D: static[int], R: static[int], L, T](
  view: var TensorFieldView[D, R, L, T]
) =
  ## Destructor for TensorFieldView
  ##
  ## Writes device data back to host memory if ioKind is AccWrite or AccReadWrite,
  ## then releases all device buffers.
  if view.data.buffers.len > 0:
    case view.ioKind:
      of AccWrite, AccReadWrite:
        if not view.data.hostPtr.isNil:
          for deviceIdx in 0..<view.data.buffers.len:
            let buf = view.data.buffers[deviceIdx]
            if not buf.isNil:
              let numElements = view.data.sitesPerDevice[deviceIdx] * view.data.elementsPerSite
              let bufferSize = numElements * sizeof(T)
              let destPtr = cast[pointer](cast[int](view.data.hostPtr) + view.data.hostOffsets[deviceIdx])
          clQueues[deviceIdx].read(destPtr, buf, bufferSize)
      of AccRead: discard
    
    # Release all buffers
    for buf in view.data.buffers:
      if not buf.isNil: release(buf)
    view.data.buffers.setLen(0)

when isMainModule:
  block:
    var argc = cargc()
    var argv = cargv(argc)
    
    initMPI(addr argc, addr argv)
    initGA()
    initCL()
    
    block:
      let dims: array[4, int] = [8, 8, 8, 16]
      let lattice = newSimpleCubicLattice(dims)

      var realTensorField1 = lattice.newTensorField([3, 3]): float64
      var complexTensorField1 = lattice.newTensorField([3, 3]): Complex64

      var localRealTensorField1 = realTensorField1.newLocalTensorField()
      var localComplexTensorField1 = complexTensorField1.newLocalTensorField()

      # transfer local tensor fields to device memory
      var deviceRealTensorView1 = localRealTensorField1.newTensorFieldView(AccRead)
      var deviceComplexTensorView1 = localComplexTensorField1.newTensorFieldView(AccRead)
      var deviceRealTensorView2 = realTensorField1.newTensorFieldView(AccWrite)
      var deviceComplexTensorView2 = complexTensorField1.newTensorFieldView(AccWrite)
    
    # tensors created in block go out of scope here and are destroyed

    finalizeCL()
    finalizeGA()
    finalizeMPI()