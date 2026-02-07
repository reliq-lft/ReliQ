#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/sycl/syclwrap.nim
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

## SYCL wrapper for Nim - Multi-Type Native Kernel Edition
##
## This module provides low-level SYCL bindings using C++ interop.
## Supports: float32, float64, int32, int64
##
## The SYCL C++ wrapper (sycl_wrapper.cpp) is compiled separately into
## libreliq_sycl.so and loaded at runtime.

# SYCL types - opaque handles to C++ objects
type
  SyclQueue* = pointer
  SyclBuffer* = pointer
  SyclDevice* = pointer
  SyclContext* = pointer
  SyclEvent* = pointer

  SyclDeviceType* = enum
    sdtDefault = 0
    sdtCPU = 1
    sdtGPU = 2
    sdtAccelerator = 3
    sdtAll = 4

  SyclResult* = enum
    srSuccess = 0
    srInvalidDevice = 1
    srInvalidContext = 2
    srInvalidQueue = 3
    srInvalidBuffer = 4
    srInvalidKernel = 5
    srCompilationError = 6
    srRuntimeError = 7

  ## Element type codes for dispatch
  SyclElementType* = enum
    setFloat32
    setFloat64
    setInt32
    setInt64

# ============================================================================
# C wrapper functions (implemented in sycl_wrapper.cpp)
# ============================================================================
{.push cdecl, importc, dynlib: "libreliq_sycl.so".}

# Device and queue management
proc sycl_get_device_count*(dtype: cint): cint
proc sycl_create_queue*(dtype: cint, deviceIdx: cint): SyclQueue
proc sycl_destroy_queue*(queue: SyclQueue)
proc sycl_get_device_name*(dtype: cint, deviceIdx: cint): cstring
proc sycl_device_is_cpu*(dtype: cint, deviceIdx: cint): cint
proc sycl_device_is_gpu*(dtype: cint, deviceIdx: cint): cint

# Memory management
proc sycl_allocate*(queue: SyclQueue, size: csize_t): SyclBuffer
proc sycl_deallocate*(queue: SyclQueue, buf: SyclBuffer)
proc sycl_write*(queue: SyclQueue, buf: SyclBuffer, src: pointer, size: csize_t)
proc sycl_read*(queue: SyclQueue, dest: pointer, buf: SyclBuffer, size: csize_t)
proc sycl_wait*(queue: SyclQueue)
proc sycl_get_buffer_ptr*(buffer: SyclBuffer): pointer

# ============================================================================
# float32 (f32) Kernels
# ============================================================================
proc sycl_kernel_copy_f32*(queue: SyclQueue, bufA, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_add_f32*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_sub_f32*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_mul_f32*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_scalar_mul_f32*(queue: SyclQueue, bufA: SyclBuffer, scalar: cfloat, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_scalar_add_f32*(queue: SyclQueue, bufA: SyclBuffer, scalar: cfloat, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_matmul_f32*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                             numSites: csize_t, rows, cols, inner: cint,
                             vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_matvec_f32*(queue: SyclQueue, bufA, bufX, bufY: SyclBuffer,
                             numSites: csize_t, rows, cols: cint,
                             vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_matadd_f32*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                             numSites: csize_t, rows, cols: cint,
                             vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_vecadd_f32*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                             numSites: csize_t, vecLen: cint,
                             vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_tensor_scalar_mul_f32*(queue: SyclQueue, bufA: SyclBuffer, scalar: cfloat, bufC: SyclBuffer,
                                        numSites: csize_t, elemsPerSite: cint,
                                        vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_tensor_scalar_add_f32*(queue: SyclQueue, bufA: SyclBuffer, scalar: cfloat, bufC: SyclBuffer,
                                        numSites: csize_t, elemsPerSite: cint,
                                        vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_set_element_f32*(queue: SyclQueue, bufC: SyclBuffer,
                                   elementIdx: cint, value: cfloat,
                                   numSites: csize_t, elemsPerSite: cint,
                                   vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_set_elements_f32*(queue: SyclQueue, bufC: SyclBuffer,
                                    elementIndices: ptr cint, values: ptr cfloat, numWrites: cint,
                                    numSites: csize_t, elemsPerSite: cint,
                                    vectorWidth: cint, numVectorGroups: csize_t)
# Complex f32
proc sycl_kernel_complex_add_f32*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer, numComplexElements: csize_t)
proc sycl_kernel_complex_scalar_mul_f32*(queue: SyclQueue, bufA: SyclBuffer, scalar_re, scalar_im: cfloat, bufC: SyclBuffer, numComplexElements: csize_t)
proc sycl_kernel_complex_matmul_f32*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                                     numSites: csize_t, rows, cols, inner: cint,
                                     vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_complex_matvec_f32*(queue: SyclQueue, bufA, bufX, bufY: SyclBuffer,
                                     numSites: csize_t, rows, cols: cint,
                                     vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_complex_matadd_f32*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                                     numSites: csize_t, rows, cols: cint,
                                     vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_complex_vecadd_f32*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                                     numSites: csize_t, vecLen: cint,
                                     vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_complex_tensor_scalar_mul_f32*(queue: SyclQueue, bufA: SyclBuffer, 
                                                scalar_re, scalar_im: cfloat, 
                                                bufC: SyclBuffer, numSites: csize_t, elemsPerSite: cint,
                                                vectorWidth: cint, numVectorGroups: csize_t)

# ============================================================================
# float64 (f64) Kernels
# ============================================================================
proc sycl_kernel_copy_f64*(queue: SyclQueue, bufA, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_add_f64*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_sub_f64*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_mul_f64*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_scalar_mul_f64*(queue: SyclQueue, bufA: SyclBuffer, scalar: cdouble, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_scalar_add_f64*(queue: SyclQueue, bufA: SyclBuffer, scalar: cdouble, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_matmul_f64*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                             numSites: csize_t, rows, cols, inner: cint,
                             vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_matvec_f64*(queue: SyclQueue, bufA, bufX, bufY: SyclBuffer,
                             numSites: csize_t, rows, cols: cint,
                             vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_matadd_f64*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                             numSites: csize_t, rows, cols: cint,
                             vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_vecadd_f64*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                             numSites: csize_t, vecLen: cint,
                             vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_tensor_scalar_mul_f64*(queue: SyclQueue, bufA: SyclBuffer, scalar: cdouble, bufC: SyclBuffer,
                                        numSites: csize_t, elemsPerSite: cint,
                                        vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_tensor_scalar_add_f64*(queue: SyclQueue, bufA: SyclBuffer, scalar: cdouble, bufC: SyclBuffer,
                                        numSites: csize_t, elemsPerSite: cint,
                                        vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_set_element_f64*(queue: SyclQueue, bufC: SyclBuffer,
                                   elementIdx: cint, value: cdouble,
                                   numSites: csize_t, elemsPerSite: cint,
                                   vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_set_elements_f64*(queue: SyclQueue, bufC: SyclBuffer,
                                    elementIndices: ptr cint, values: ptr cdouble, numWrites: cint,
                                    numSites: csize_t, elemsPerSite: cint,
                                    vectorWidth: cint, numVectorGroups: csize_t)
# Complex f64
proc sycl_kernel_complex_add_f64*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer, numComplexElements: csize_t)
proc sycl_kernel_complex_scalar_mul_f64*(queue: SyclQueue, bufA: SyclBuffer, scalar_re, scalar_im: cdouble, bufC: SyclBuffer, numComplexElements: csize_t)
proc sycl_kernel_complex_matmul_f64*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                                     numSites: csize_t, rows, cols, inner: cint,
                                     vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_complex_matvec_f64*(queue: SyclQueue, bufA, bufX, bufY: SyclBuffer,
                                     numSites: csize_t, rows, cols: cint,
                                     vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_complex_matadd_f64*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                                     numSites: csize_t, rows, cols: cint,
                                     vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_complex_vecadd_f64*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                                     numSites: csize_t, vecLen: cint,
                                     vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_complex_tensor_scalar_mul_f64*(queue: SyclQueue, bufA: SyclBuffer, 
                                                scalar_re, scalar_im: cdouble, 
                                                bufC: SyclBuffer, numSites: csize_t, elemsPerSite: cint,
                                                vectorWidth: cint, numVectorGroups: csize_t)

# ============================================================================
# int32 (i32) Kernels
# ============================================================================
proc sycl_kernel_copy_i32*(queue: SyclQueue, bufA, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_add_i32*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_sub_i32*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_mul_i32*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_scalar_mul_i32*(queue: SyclQueue, bufA: SyclBuffer, scalar: cint, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_scalar_add_i32*(queue: SyclQueue, bufA: SyclBuffer, scalar: cint, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_matmul_i32*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                             numSites: csize_t, rows, cols, inner: cint,
                             vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_matvec_i32*(queue: SyclQueue, bufA, bufX, bufY: SyclBuffer,
                             numSites: csize_t, rows, cols: cint,
                             vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_matadd_i32*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                             numSites: csize_t, rows, cols: cint,
                             vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_vecadd_i32*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                             numSites: csize_t, vecLen: cint,
                             vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_tensor_scalar_mul_i32*(queue: SyclQueue, bufA: SyclBuffer, scalar: cint, bufC: SyclBuffer,
                                        numSites: csize_t, elemsPerSite: cint,
                                        vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_tensor_scalar_add_i32*(queue: SyclQueue, bufA: SyclBuffer, scalar: cint, bufC: SyclBuffer,
                                        numSites: csize_t, elemsPerSite: cint,
                                        vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_set_element_i32*(queue: SyclQueue, bufC: SyclBuffer,
                                   elementIdx: cint, value: cint,
                                   numSites: csize_t, elemsPerSite: cint,
                                   vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_set_elements_i32*(queue: SyclQueue, bufC: SyclBuffer,
                                    elementIndices: ptr cint, values: ptr cint, numWrites: cint,
                                    numSites: csize_t, elemsPerSite: cint,
                                    vectorWidth: cint, numVectorGroups: csize_t)

# ============================================================================
# int64 (i64) Kernels
# ============================================================================
proc sycl_kernel_copy_i64*(queue: SyclQueue, bufA, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_add_i64*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_sub_i64*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_mul_i64*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_scalar_mul_i64*(queue: SyclQueue, bufA: SyclBuffer, scalar: clonglong, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_scalar_add_i64*(queue: SyclQueue, bufA: SyclBuffer, scalar: clonglong, bufC: SyclBuffer, numElements: csize_t)
proc sycl_kernel_matmul_i64*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                             numSites: csize_t, rows, cols, inner: cint,
                             vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_matvec_i64*(queue: SyclQueue, bufA, bufX, bufY: SyclBuffer,
                             numSites: csize_t, rows, cols: cint,
                             vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_matadd_i64*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                             numSites: csize_t, rows, cols: cint,
                             vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_vecadd_i64*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                             numSites: csize_t, vecLen: cint,
                             vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_tensor_scalar_mul_i64*(queue: SyclQueue, bufA: SyclBuffer, scalar: clonglong, bufC: SyclBuffer,
                                        numSites: csize_t, elemsPerSite: cint,
                                        vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_tensor_scalar_add_i64*(queue: SyclQueue, bufA: SyclBuffer, scalar: clonglong, bufC: SyclBuffer,
                                        numSites: csize_t, elemsPerSite: cint,
                                        vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_set_element_i64*(queue: SyclQueue, bufC: SyclBuffer,
                                   elementIdx: cint, value: clonglong,
                                   numSites: csize_t, elemsPerSite: cint,
                                   vectorWidth: cint, numVectorGroups: csize_t)
proc sycl_kernel_set_elements_i64*(queue: SyclQueue, bufC: SyclBuffer,
                                    elementIndices: ptr cint, values: ptr clonglong, numWrites: cint,
                                    numSites: csize_t, elemsPerSite: cint,
                                    vectorWidth: cint, numVectorGroups: csize_t)

# ============================================================================
# Stencil Gather Kernels
# ============================================================================
# float32
proc sycl_kernel_stencil_copy_f32*(queue: SyclQueue, bufSrc, bufDst, bufOffsets: SyclBuffer,
                                    pointIdx: cint, nPoints: cint,
                                    numSites: csize_t, elemsPerSite: cint, vectorWidth: cint)
proc sycl_kernel_stencil_scalar_mul_f32*(queue: SyclQueue, bufSrc: SyclBuffer, scalar: cfloat, bufDst, bufOffsets: SyclBuffer,
                                          pointIdx: cint, nPoints: cint,
                                          numSites: csize_t, elemsPerSite: cint, vectorWidth: cint)
proc sycl_kernel_stencil_add_f32*(queue: SyclQueue, bufSrcA, bufSrcB, bufDst, bufOffsets: SyclBuffer,
                                   pointIdx: cint, nPoints: cint,
                                   numSites: csize_t, elemsPerSite: cint, vectorWidth: cint)
# float64
proc sycl_kernel_stencil_copy_f64*(queue: SyclQueue, bufSrc, bufDst, bufOffsets: SyclBuffer,
                                    pointIdx: cint, nPoints: cint,
                                    numSites: csize_t, elemsPerSite: cint, vectorWidth: cint)
proc sycl_kernel_stencil_scalar_mul_f64*(queue: SyclQueue, bufSrc: SyclBuffer, scalar: cdouble, bufDst, bufOffsets: SyclBuffer,
                                          pointIdx: cint, nPoints: cint,
                                          numSites: csize_t, elemsPerSite: cint, vectorWidth: cint)
proc sycl_kernel_stencil_add_f64*(queue: SyclQueue, bufSrcA, bufSrcB, bufDst, bufOffsets: SyclBuffer,
                                   pointIdx: cint, nPoints: cint,
                                   numSites: csize_t, elemsPerSite: cint, vectorWidth: cint)
# int32
proc sycl_kernel_stencil_copy_i32*(queue: SyclQueue, bufSrc, bufDst, bufOffsets: SyclBuffer,
                                    pointIdx: cint, nPoints: cint,
                                    numSites: csize_t, elemsPerSite: cint, vectorWidth: cint)
proc sycl_kernel_stencil_scalar_mul_i32*(queue: SyclQueue, bufSrc: SyclBuffer, scalar: cint, bufDst, bufOffsets: SyclBuffer,
                                          pointIdx: cint, nPoints: cint,
                                          numSites: csize_t, elemsPerSite: cint, vectorWidth: cint)
proc sycl_kernel_stencil_add_i32*(queue: SyclQueue, bufSrcA, bufSrcB, bufDst, bufOffsets: SyclBuffer,
                                   pointIdx: cint, nPoints: cint,
                                   numSites: csize_t, elemsPerSite: cint, vectorWidth: cint)
# int64
proc sycl_kernel_stencil_copy_i64*(queue: SyclQueue, bufSrc, bufDst, bufOffsets: SyclBuffer,
                                    pointIdx: cint, nPoints: cint,
                                    numSites: csize_t, elemsPerSite: cint, vectorWidth: cint)
proc sycl_kernel_stencil_scalar_mul_i64*(queue: SyclQueue, bufSrc: SyclBuffer, scalar: clonglong, bufDst, bufOffsets: SyclBuffer,
                                          pointIdx: cint, nPoints: cint,
                                          numSites: csize_t, elemsPerSite: cint, vectorWidth: cint)
proc sycl_kernel_stencil_add_i64*(queue: SyclQueue, bufSrcA, bufSrcB, bufDst, bufOffsets: SyclBuffer,
                                   pointIdx: cint, nPoints: cint,
                                   numSites: csize_t, elemsPerSite: cint, vectorWidth: cint)

{.pop.}

# ============================================================================
# Nim-friendly wrappers
# ============================================================================

type
  ESycl* = object of CatchableError
    ## SYCL exception type

proc check*(result: SyclResult) =
  ## Check SYCL result and raise exception on error
  if result != srSuccess:
    raise newException(ESycl, "SYCL error: " & $result)

# Device and queue management
proc getDeviceCount*(dtype: SyclDeviceType = sdtDefault): int =
  result = sycl_get_device_count(dtype.cint).int

proc createQueue*(dtype: SyclDeviceType = sdtDefault, deviceIdx: int = 0): SyclQueue =
  result = sycl_create_queue(dtype.cint, deviceIdx.cint)
  if result == nil:
    raise newException(ESycl, "Failed to create SYCL queue")

proc destroyQueue*(queue: SyclQueue) =
  sycl_destroy_queue(queue)

proc getDeviceName*(dtype: SyclDeviceType = sdtDefault, deviceIdx: int = 0): string =
  result = $sycl_get_device_name(dtype.cint, deviceIdx.cint)

proc isCpu*(dtype: SyclDeviceType = sdtDefault, deviceIdx: int = 0): bool =
  result = sycl_device_is_cpu(dtype.cint, deviceIdx.cint) != 0

proc isGpu*(dtype: SyclDeviceType = sdtDefault, deviceIdx: int = 0): bool =
  result = sycl_device_is_gpu(dtype.cint, deviceIdx.cint) != 0

# Memory management
proc allocate*(queue: SyclQueue, size: int): SyclBuffer =
  result = sycl_allocate(queue, size.csize_t)
  if result == nil:
    raise newException(ESycl, "Failed to allocate SYCL buffer")

proc deallocate*(queue: SyclQueue, buf: SyclBuffer) =
  sycl_deallocate(queue, buf)

proc write*[T](queue: SyclQueue, buf: SyclBuffer, data: openArray[T]) =
  if data.len > 0:
    sycl_write(queue, buf, unsafeAddr data[0], (data.len * sizeof(T)).csize_t)

proc write*(queue: SyclQueue, buf: SyclBuffer, src: pointer, size: int) =
  sycl_write(queue, buf, src, size.csize_t)

proc read*[T](queue: SyclQueue, data: var openArray[T], buf: SyclBuffer) =
  if data.len > 0:
    sycl_read(queue, addr data[0], buf, (data.len * sizeof(T)).csize_t)

proc read*(queue: SyclQueue, dest: pointer, buf: SyclBuffer, size: int) =
  sycl_read(queue, dest, buf, size.csize_t)

proc wait*(queue: SyclQueue) =
  sycl_wait(queue)

proc getBufferPtr*(buffer: SyclBuffer): pointer =
  result = sycl_get_buffer_ptr(buffer)

proc finish*(queue: SyclQueue): int {.discardable.} =
  ## Wait for queue to finish and return 0
  sycl_wait(queue)
  result = 0

# ============================================================================
# Type-Generic High-Level Kernel Wrappers
# These use overloading to select the correct typed kernel
# ============================================================================

# Copy kernels
proc kernelCopy*(queue: SyclQueue, bufA, bufC: SyclBuffer, numElements: int) {.inline.} =
  ## Copy buffer A to buffer C (float64 default for backward compatibility)
  sycl_kernel_copy_f64(queue, bufA, bufC, numElements.csize_t)

proc kernelCopy*[T](queue: SyclQueue, bufA, bufC: SyclBuffer, numElements: int, dummy: typedesc[T]) {.inline.} =
  ## Type-specific copy kernel
  when T is float32:
    sycl_kernel_copy_f32(queue, bufA, bufC, numElements.csize_t)
  elif T is float64:
    sycl_kernel_copy_f64(queue, bufA, bufC, numElements.csize_t)
  elif T is int32:
    sycl_kernel_copy_i32(queue, bufA, bufC, numElements.csize_t)
  elif T is int64:
    sycl_kernel_copy_i64(queue, bufA, bufC, numElements.csize_t)
  else:
    {.error: "Unsupported type for kernelCopy".}

# Add kernels
proc kernelAdd*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer, numElements: int) {.inline.} =
  sycl_kernel_add_f64(queue, bufA, bufB, bufC, numElements.csize_t)

proc kernelAdd*[T](queue: SyclQueue, bufA, bufB, bufC: SyclBuffer, numElements: int, dummy: typedesc[T]) {.inline.} =
  when T is float32:
    sycl_kernel_add_f32(queue, bufA, bufB, bufC, numElements.csize_t)
  elif T is float64:
    sycl_kernel_add_f64(queue, bufA, bufB, bufC, numElements.csize_t)
  elif T is int32:
    sycl_kernel_add_i32(queue, bufA, bufB, bufC, numElements.csize_t)
  elif T is int64:
    sycl_kernel_add_i64(queue, bufA, bufB, bufC, numElements.csize_t)
  else:
    {.error: "Unsupported type for kernelAdd".}

# Sub kernels
proc kernelSub*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer, numElements: int) {.inline.} =
  sycl_kernel_sub_f64(queue, bufA, bufB, bufC, numElements.csize_t)

proc kernelSub*[T](queue: SyclQueue, bufA, bufB, bufC: SyclBuffer, numElements: int, dummy: typedesc[T]) {.inline.} =
  when T is float32:
    sycl_kernel_sub_f32(queue, bufA, bufB, bufC, numElements.csize_t)
  elif T is float64:
    sycl_kernel_sub_f64(queue, bufA, bufB, bufC, numElements.csize_t)
  elif T is int32:
    sycl_kernel_sub_i32(queue, bufA, bufB, bufC, numElements.csize_t)
  elif T is int64:
    sycl_kernel_sub_i64(queue, bufA, bufB, bufC, numElements.csize_t)
  else:
    {.error: "Unsupported type for kernelSub".}

# Mul kernels
proc kernelMul*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer, numElements: int) {.inline.} =
  sycl_kernel_mul_f64(queue, bufA, bufB, bufC, numElements.csize_t)

proc kernelMul*[T](queue: SyclQueue, bufA, bufB, bufC: SyclBuffer, numElements: int, dummy: typedesc[T]) {.inline.} =
  when T is float32:
    sycl_kernel_mul_f32(queue, bufA, bufB, bufC, numElements.csize_t)
  elif T is float64:
    sycl_kernel_mul_f64(queue, bufA, bufB, bufC, numElements.csize_t)
  elif T is int32:
    sycl_kernel_mul_i32(queue, bufA, bufB, bufC, numElements.csize_t)
  elif T is int64:
    sycl_kernel_mul_i64(queue, bufA, bufB, bufC, numElements.csize_t)
  else:
    {.error: "Unsupported type for kernelMul".}

# Scalar mul kernels
proc kernelScalarMul*(queue: SyclQueue, bufA: SyclBuffer, scalar: float64, bufC: SyclBuffer, numElements: int) {.inline.} =
  sycl_kernel_scalar_mul_f64(queue, bufA, scalar.cdouble, bufC, numElements.csize_t)

proc kernelScalarMul*[T](queue: SyclQueue, bufA: SyclBuffer, scalar: T, bufC: SyclBuffer, numElements: int, dummy: typedesc[T]) {.inline.} =
  when T is float32:
    sycl_kernel_scalar_mul_f32(queue, bufA, scalar.cfloat, bufC, numElements.csize_t)
  elif T is float64:
    sycl_kernel_scalar_mul_f64(queue, bufA, scalar.cdouble, bufC, numElements.csize_t)
  elif T is int32:
    sycl_kernel_scalar_mul_i32(queue, bufA, scalar.cint, bufC, numElements.csize_t)
  elif T is int64:
    sycl_kernel_scalar_mul_i64(queue, bufA, scalar.clonglong, bufC, numElements.csize_t)
  else:
    {.error: "Unsupported type for kernelScalarMul".}

# Scalar add kernels
proc kernelScalarAdd*(queue: SyclQueue, bufA: SyclBuffer, scalar: float64, bufC: SyclBuffer, numElements: int) {.inline.} =
  sycl_kernel_scalar_add_f64(queue, bufA, scalar.cdouble, bufC, numElements.csize_t)

proc kernelScalarAdd*[T](queue: SyclQueue, bufA: SyclBuffer, scalar: T, bufC: SyclBuffer, numElements: int, dummy: typedesc[T]) {.inline.} =
  when T is float32:
    sycl_kernel_scalar_add_f32(queue, bufA, scalar.cfloat, bufC, numElements.csize_t)
  elif T is float64:
    sycl_kernel_scalar_add_f64(queue, bufA, scalar.cdouble, bufC, numElements.csize_t)
  elif T is int32:
    sycl_kernel_scalar_add_i32(queue, bufA, scalar.cint, bufC, numElements.csize_t)
  elif T is int64:
    sycl_kernel_scalar_add_i64(queue, bufA, scalar.clonglong, bufC, numElements.csize_t)
  else:
    {.error: "Unsupported type for kernelScalarAdd".}

# ============================================================================
# Matrix Operations (float64 default for backward compatibility)
# ============================================================================

proc kernelMatMul*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                   numSites: int, rows, cols, inner: int,
                   vectorWidth: int, numVectorGroups: int) {.inline.} =
  sycl_kernel_matmul_f64(queue, bufA, bufB, bufC,
                         numSites.csize_t, rows.cint, cols.cint, inner.cint,
                         vectorWidth.cint, numVectorGroups.csize_t)

proc kernelMatMul*[T](queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                      numSites: int, rows, cols, inner: int,
                      vectorWidth: int, numVectorGroups: int, dummy: typedesc[T]) {.inline.} =
  when T is float32:
    sycl_kernel_matmul_f32(queue, bufA, bufB, bufC,
                           numSites.csize_t, rows.cint, cols.cint, inner.cint,
                           vectorWidth.cint, numVectorGroups.csize_t)
  elif T is float64:
    sycl_kernel_matmul_f64(queue, bufA, bufB, bufC,
                           numSites.csize_t, rows.cint, cols.cint, inner.cint,
                           vectorWidth.cint, numVectorGroups.csize_t)
  elif T is int32:
    sycl_kernel_matmul_i32(queue, bufA, bufB, bufC,
                           numSites.csize_t, rows.cint, cols.cint, inner.cint,
                           vectorWidth.cint, numVectorGroups.csize_t)
  elif T is int64:
    sycl_kernel_matmul_i64(queue, bufA, bufB, bufC,
                           numSites.csize_t, rows.cint, cols.cint, inner.cint,
                           vectorWidth.cint, numVectorGroups.csize_t)
  else:
    {.error: "Unsupported type for kernelMatMul".}

proc kernelMatVec*(queue: SyclQueue, bufA, bufX, bufY: SyclBuffer,
                   numSites: int, rows, cols: int,
                   vectorWidth: int, numVectorGroups: int) {.inline.} =
  sycl_kernel_matvec_f64(queue, bufA, bufX, bufY,
                         numSites.csize_t, rows.cint, cols.cint,
                         vectorWidth.cint, numVectorGroups.csize_t)

proc kernelMatVec*[T](queue: SyclQueue, bufA, bufX, bufY: SyclBuffer,
                      numSites: int, rows, cols: int,
                      vectorWidth: int, numVectorGroups: int, dummy: typedesc[T]) {.inline.} =
  when T is float32:
    sycl_kernel_matvec_f32(queue, bufA, bufX, bufY,
                           numSites.csize_t, rows.cint, cols.cint,
                           vectorWidth.cint, numVectorGroups.csize_t)
  elif T is float64:
    sycl_kernel_matvec_f64(queue, bufA, bufX, bufY,
                           numSites.csize_t, rows.cint, cols.cint,
                           vectorWidth.cint, numVectorGroups.csize_t)
  elif T is int32:
    sycl_kernel_matvec_i32(queue, bufA, bufX, bufY,
                           numSites.csize_t, rows.cint, cols.cint,
                           vectorWidth.cint, numVectorGroups.csize_t)
  elif T is int64:
    sycl_kernel_matvec_i64(queue, bufA, bufX, bufY,
                           numSites.csize_t, rows.cint, cols.cint,
                           vectorWidth.cint, numVectorGroups.csize_t)
  else:
    {.error: "Unsupported type for kernelMatVec".}

proc kernelMatAdd*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                   numSites: int, rows, cols: int,
                   vectorWidth: int, numVectorGroups: int) {.inline.} =
  sycl_kernel_matadd_f64(queue, bufA, bufB, bufC,
                         numSites.csize_t, rows.cint, cols.cint,
                         vectorWidth.cint, numVectorGroups.csize_t)

proc kernelVecAdd*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                   numSites: int, vecLen: int,
                   vectorWidth: int, numVectorGroups: int) {.inline.} =
  sycl_kernel_vecadd_f64(queue, bufA, bufB, bufC,
                         numSites.csize_t, vecLen.cint,
                         vectorWidth.cint, numVectorGroups.csize_t)

# Complex operations (float64 only for now)
proc kernelComplexAdd*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer, numComplexElements: int) {.inline.} =
  sycl_kernel_complex_add_f64(queue, bufA, bufB, bufC, numComplexElements.csize_t)

proc kernelComplexScalarMul*(queue: SyclQueue, bufA: SyclBuffer, re, im: float64, bufC: SyclBuffer, numComplexElements: int) {.inline.} =
  sycl_kernel_complex_scalar_mul_f64(queue, bufA, re.cdouble, im.cdouble, bufC, numComplexElements.csize_t)

proc kernelComplexMatMul*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                          numSites: int, rows, cols, inner: int,
                          vectorWidth: int, numVectorGroups: int) {.inline.} =
  sycl_kernel_complex_matmul_f64(queue, bufA, bufB, bufC,
                                 numSites.csize_t, rows.cint, cols.cint, inner.cint,
                                 vectorWidth.cint, numVectorGroups.csize_t)

proc kernelComplexMatVec*(queue: SyclQueue, bufA, bufX, bufY: SyclBuffer,
                          numSites: int, rows, cols: int,
                          vectorWidth: int, numVectorGroups: int) {.inline.} =
  sycl_kernel_complex_matvec_f64(queue, bufA, bufX, bufY,
                                 numSites.csize_t, rows.cint, cols.cint,
                                 vectorWidth.cint, numVectorGroups.csize_t)

proc kernelComplexMatAdd*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                          numSites: int, rows, cols: int,
                          vectorWidth: int, numVectorGroups: int) {.inline.} =
  sycl_kernel_complex_matadd_f64(queue, bufA, bufB, bufC,
                                 numSites.csize_t, rows.cint, cols.cint,
                                 vectorWidth.cint, numVectorGroups.csize_t)

proc kernelComplexVecAdd*(queue: SyclQueue, bufA, bufB, bufC: SyclBuffer,
                          numSites: int, vecLen: int,
                          vectorWidth: int, numVectorGroups: int) {.inline.} =
  sycl_kernel_complex_vecadd_f64(queue, bufA, bufB, bufC,
                                 numSites.csize_t, vecLen.cint,
                                 vectorWidth.cint, numVectorGroups.csize_t)

proc kernelTensorScalarMul*(queue: SyclQueue, bufA: SyclBuffer, scalar: float64, bufC: SyclBuffer,
                            numSites: int, elemsPerSite: int,
                            vectorWidth: int, numVectorGroups: int) {.inline.} =
  sycl_kernel_tensor_scalar_mul_f64(queue, bufA, scalar.cdouble, bufC,
                                    numSites.csize_t, elemsPerSite.cint,
                                    vectorWidth.cint, numVectorGroups.csize_t)

proc kernelComplexTensorScalarMul*(queue: SyclQueue, bufA: SyclBuffer, re, im: float64, bufC: SyclBuffer,
                                   numSites: int, elemsPerSite: int,
                                   vectorWidth: int, numVectorGroups: int) {.inline.} =
  sycl_kernel_complex_tensor_scalar_mul_f64(queue, bufA, re.cdouble, im.cdouble, bufC,
                                            numSites.csize_t, elemsPerSite.cint,
                                            vectorWidth.cint, numVectorGroups.csize_t)

proc kernelTensorScalarAdd*(queue: SyclQueue, bufA: SyclBuffer, scalar: float64, bufC: SyclBuffer,
                            numSites: int, elemsPerSite: int,
                            vectorWidth: int, numVectorGroups: int) {.inline.} =
  sycl_kernel_tensor_scalar_add_f64(queue, bufA, scalar.cdouble, bufC,
                                    numSites.csize_t, elemsPerSite.cint,
                                    vectorWidth.cint, numVectorGroups.csize_t)

# Element write kernels (float64 default)
proc kernelSetElement*(queue: SyclQueue, bufC: SyclBuffer,
                       elementIdx: int, value: float64,
                       numSites: int, elemsPerSite: int,
                       vectorWidth: int, numVectorGroups: int) {.inline.} =
  sycl_kernel_set_element_f64(queue, bufC, elementIdx.cint, value.cdouble,
                              numSites.csize_t, elemsPerSite.cint,
                              vectorWidth.cint, numVectorGroups.csize_t)

proc kernelSetElements*(queue: SyclQueue, bufC: SyclBuffer,
                        elementIndices: openArray[int32], values: openArray[float64],
                        numSites: int, elemsPerSite: int,
                        vectorWidth: int, numVectorGroups: int) {.inline.} =
  if elementIndices.len != values.len or elementIndices.len == 0:
    return
  var indices = newSeq[cint](elementIndices.len)
  var vals = newSeq[cdouble](values.len)
  for i in 0..<elementIndices.len:
    indices[i] = elementIndices[i].cint
    vals[i] = values[i].cdouble
  sycl_kernel_set_elements_f64(queue, bufC,
                               addr indices[0], addr vals[0], elementIndices.len.cint,
                               numSites.csize_t, elemsPerSite.cint,
                               vectorWidth.cint, numVectorGroups.csize_t)

# ============================================================================
# Stencil Gather Kernels (type-generic)
# ============================================================================

proc kernelStencilCopy*(queue: SyclQueue, bufSrc, bufDst, bufOffsets: SyclBuffer,
                        pointIdx: int, nPoints: int,
                        numSites: int, elemsPerSite: int, vectorWidth: int) {.inline.} =
  ## Gather copy: dst[n] = src[neighbor(n, pointIdx)]
  sycl_kernel_stencil_copy_f64(queue, bufSrc, bufDst, bufOffsets,
                                pointIdx.cint, nPoints.cint,
                                numSites.csize_t, elemsPerSite.cint, vectorWidth.cint)

proc kernelStencilCopy*[T](queue: SyclQueue, bufSrc, bufDst, bufOffsets: SyclBuffer,
                           pointIdx: int, nPoints: int,
                           numSites: int, elemsPerSite: int, vectorWidth: int,
                           dummy: typedesc[T]) {.inline.} =
  when T is float32:
    sycl_kernel_stencil_copy_f32(queue, bufSrc, bufDst, bufOffsets,
                                  pointIdx.cint, nPoints.cint,
                                  numSites.csize_t, elemsPerSite.cint, vectorWidth.cint)
  elif T is float64:
    sycl_kernel_stencil_copy_f64(queue, bufSrc, bufDst, bufOffsets,
                                  pointIdx.cint, nPoints.cint,
                                  numSites.csize_t, elemsPerSite.cint, vectorWidth.cint)
  elif T is int32:
    sycl_kernel_stencil_copy_i32(queue, bufSrc, bufDst, bufOffsets,
                                  pointIdx.cint, nPoints.cint,
                                  numSites.csize_t, elemsPerSite.cint, vectorWidth.cint)
  elif T is int64:
    sycl_kernel_stencil_copy_i64(queue, bufSrc, bufDst, bufOffsets,
                                  pointIdx.cint, nPoints.cint,
                                  numSites.csize_t, elemsPerSite.cint, vectorWidth.cint)
  else:
    {.error: "Unsupported type for kernelStencilCopy".}

proc kernelStencilScalarMul*(queue: SyclQueue, bufSrc: SyclBuffer, scalar: float64,
                             bufDst, bufOffsets: SyclBuffer,
                             pointIdx: int, nPoints: int,
                             numSites: int, elemsPerSite: int, vectorWidth: int) {.inline.} =
  ## Gather scalar mul: dst[n] = scalar * src[neighbor(n, pointIdx)]
  sycl_kernel_stencil_scalar_mul_f64(queue, bufSrc, scalar.cdouble, bufDst, bufOffsets,
                                      pointIdx.cint, nPoints.cint,
                                      numSites.csize_t, elemsPerSite.cint, vectorWidth.cint)

proc kernelStencilScalarMul*[T](queue: SyclQueue, bufSrc: SyclBuffer, scalar: T,
                                bufDst, bufOffsets: SyclBuffer,
                                pointIdx: int, nPoints: int,
                                numSites: int, elemsPerSite: int, vectorWidth: int,
                                dummy: typedesc[T]) {.inline.} =
  when T is float32:
    sycl_kernel_stencil_scalar_mul_f32(queue, bufSrc, scalar.cfloat, bufDst, bufOffsets,
                                        pointIdx.cint, nPoints.cint,
                                        numSites.csize_t, elemsPerSite.cint, vectorWidth.cint)
  elif T is float64:
    sycl_kernel_stencil_scalar_mul_f64(queue, bufSrc, scalar.cdouble, bufDst, bufOffsets,
                                        pointIdx.cint, nPoints.cint,
                                        numSites.csize_t, elemsPerSite.cint, vectorWidth.cint)
  elif T is int32:
    sycl_kernel_stencil_scalar_mul_i32(queue, bufSrc, scalar.cint, bufDst, bufOffsets,
                                        pointIdx.cint, nPoints.cint,
                                        numSites.csize_t, elemsPerSite.cint, vectorWidth.cint)
  elif T is int64:
    sycl_kernel_stencil_scalar_mul_i64(queue, bufSrc, scalar.clonglong, bufDst, bufOffsets,
                                        pointIdx.cint, nPoints.cint,
                                        numSites.csize_t, elemsPerSite.cint, vectorWidth.cint)
  else:
    {.error: "Unsupported type for kernelStencilScalarMul".}

proc kernelStencilAdd*(queue: SyclQueue, bufSrcA, bufSrcB, bufDst, bufOffsets: SyclBuffer,
                       pointIdx: int, nPoints: int,
                       numSites: int, elemsPerSite: int, vectorWidth: int) {.inline.} =
  ## Gather add: dst[n] = srcA[n] + srcB[neighbor(n, pointIdx)]
  sycl_kernel_stencil_add_f64(queue, bufSrcA, bufSrcB, bufDst, bufOffsets,
                               pointIdx.cint, nPoints.cint,
                               numSites.csize_t, elemsPerSite.cint, vectorWidth.cint)

proc kernelStencilAdd*[T](queue: SyclQueue, bufSrcA, bufSrcB, bufDst, bufOffsets: SyclBuffer,
                          pointIdx: int, nPoints: int,
                          numSites: int, elemsPerSite: int, vectorWidth: int,
                          dummy: typedesc[T]) {.inline.} =
  when T is float32:
    sycl_kernel_stencil_add_f32(queue, bufSrcA, bufSrcB, bufDst, bufOffsets,
                                 pointIdx.cint, nPoints.cint,
                                 numSites.csize_t, elemsPerSite.cint, vectorWidth.cint)
  elif T is float64:
    sycl_kernel_stencil_add_f64(queue, bufSrcA, bufSrcB, bufDst, bufOffsets,
                                 pointIdx.cint, nPoints.cint,
                                 numSites.csize_t, elemsPerSite.cint, vectorWidth.cint)
  elif T is int32:
    sycl_kernel_stencil_add_i32(queue, bufSrcA, bufSrcB, bufDst, bufOffsets,
                                 pointIdx.cint, nPoints.cint,
                                 numSites.csize_t, elemsPerSite.cint, vectorWidth.cint)
  elif T is int64:
    sycl_kernel_stencil_add_i64(queue, bufSrcA, bufSrcB, bufDst, bufOffsets,
                                 pointIdx.cint, nPoints.cint,
                                 numSites.csize_t, elemsPerSite.cint, vectorWidth.cint)
  else:
    {.error: "Unsupported type for kernelStencilAdd".}
