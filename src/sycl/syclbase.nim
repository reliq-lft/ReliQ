#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/sycl/syclbase.nim
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

## SYCL Base Module - Native Kernel Edition
##
## Higher-level SYCL utilities for tensor operations.
## This version uses pre-compiled native SYCL kernels, ensuring
## compatibility with all SYCL devices (CPU, GPU, accelerators).
##
## Global State:
## - syclQueues: seq[SyclQueue] - One queue per device
## - syclDeviceType: SyclDeviceType - Device type (CPU/GPU/etc)
##
## Usage:
##   initSycl()                    # Initialize SYCL with default device
##   initSycl(sdtGPU)              # Initialize SYCL with GPU
##   var buf = allocate[float64](queue, 1024)
##   write(queue, data, buf)
##   # ... run kernels via the each macro ...
##   read(queue, result, buf)
##   finalizeSycl()

import syclwrap
# Export types and symbols from syclwrap
export SyclQueue, SyclBuffer, SyclDevice, SyclContext, SyclEvent
export SyclDeviceType, SyclResult, ESycl
export getDeviceCount, createQueue, destroyQueue, wait, getDeviceName
export allocate, deallocate, isCpu, isGpu, getBufferPtr

# Export kernel functions
export kernelCopy, kernelAdd, kernelSub, kernelMul
export kernelScalarMul, kernelScalarAdd
export kernelComplexAdd, kernelComplexScalarMul
export kernelMatMul, kernelComplexMatMul
export kernelMatVec, kernelComplexMatVec
export kernelMatAdd, kernelComplexMatAdd
export kernelVecAdd, kernelComplexVecAdd
export kernelTensorScalarMul, kernelComplexTensorScalarMul, kernelTensorScalarAdd
export kernelSetElement, kernelSetElements

# Compatibility with OpenCL result checking
type TClResult* = enum
  Success = 0

template check*(a: TClResult) =
  ## OpenCL compatibility - SYCL throws exceptions on error
  discard

var syclQueues*: seq[SyclQueue]
var syclDeviceType*: SyclDeviceType = sdtDefault

proc initSycl*(dtype: SyclDeviceType = sdtDefault) =
  ## Initialize SYCL with the specified device type.
  ## Creates one queue per available device.
  syclDeviceType = dtype
  let numDevices = getDeviceCount(dtype)
  if numDevices == 0:
    raise newException(ESycl, "No SYCL devices found for type: " & $dtype)
  
  syclQueues = newSeq[SyclQueue](numDevices)
  for i in 0..<numDevices:
    syclQueues[i] = createQueue(dtype, i)
    let deviceName = getDeviceName(dtype, i)
    let deviceType = if isCpu(dtype, i): "CPU" elif isGpu(dtype, i): "GPU" else: "Accelerator"
    echo "SYCL: Initialized device ", i, " (", deviceType, "): ", deviceName
  
  echo "SYCL: Using native pre-compiled kernels (works on all SYCL devices)"

proc finalizeSycl*() =
  ## Finalize SYCL and release all resources.
  for queue in syclQueues:
    if queue != nil:
      wait(queue)
      destroyQueue(queue)
  syclQueues = @[]

# Buffer management

proc buffer*[T](queue: SyclQueue, count: int): SyclBuffer =
  ## Allocate a typed buffer on the device.
  result = allocate(queue, count * sizeof(T))

proc bufferLike*[T](queue: SyclQueue, data: openArray[T]): SyclBuffer =
  ## Allocate a buffer sized to match the given array.
  result = allocate(queue, data.len * sizeof(T))

proc write*[T](queue: SyclQueue, data: openArray[T], buf: SyclBuffer) =
  ## Write array data to device buffer.
  if data.len > 0:
    syclwrap.write(queue, buf, unsafeAddr data[0], data.len * sizeof(T))

proc read*[T](queue: SyclQueue, data: var openArray[T], buf: SyclBuffer) =
  ## Read device buffer into array.
  if data.len > 0:
    syclwrap.read(queue, addr data[0], buf, data.len * sizeof(T))

proc read*(queue: SyclQueue, dest: pointer, buf: SyclBuffer, size: int) =
  ## Read raw data from device buffer.
  syclwrap.read(queue, dest, buf, size)

proc write*(queue: SyclQueue, src: pointer, buf: SyclBuffer, size: int) =
  ## Write raw data to device buffer.
  syclwrap.write(queue, buf, src, size)

proc finish*(queue: SyclQueue): TClResult =
  ## Wait for all operations on queue to complete.
  ## Returns Success for OpenCL compatibility.
  wait(queue)
  result = TClResult.Success

proc release*(buf: SyclBuffer, queue: SyclQueue) =
  ## Release a buffer with explicit queue.
  deallocate(queue, buf)

proc release*(buf: SyclBuffer) =
  ## Release a buffer using the first global queue.
  ## OpenCL compatibility - in OpenCL, buffers are associated with context not queue.
  if syclQueues.len > 0:
    deallocate(syclQueues[0], buf)

# Compatibility aliases to match clbase API
type
  PMem* = SyclBuffer
  PCommandQueue* = SyclQueue

# These allow code using OpenCL naming to work with SYCL
template clQueues*: seq[SyclQueue] = syclQueues
template clContext*: SyclQueue = syclQueues[0]
template clDevices*: seq[SyclQueue] = syclQueues
