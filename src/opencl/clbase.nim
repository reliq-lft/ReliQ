#[
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/opencl/clbase.nim
  Contact: reliq-lft@proton.me

  Author: Andrea Ferretti
  Modifications: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  Original License:

  Copyright 2016-2017 UniCredit S.p.A.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  ReliQ Modifications License:

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

#[
  Notes:

  Resources: 
    - https://www.compilersutra.com/docs/gpu/opencl/basic/running_first_opencl_code/

  OpenCL language:
    - Compute unit = SIMD core or GPU core
    - Processing element = SIMD lane or ALU
    - Work-item = executes thread instance
    - Work-group = group of work-items executed together
  
  OpenCL initialization API:
    - clGetPlatformIDs: get available OpenCL platforms (vendors - Intel, AMD, NVIDIA)
    - clGetDeviceIDs: enumrate CPUs, GPUs, etc
    - clGetDeviceInfo: check device type, memory size, max compute units, etc
    - clCreateContext: establish link between host and selected devices

  OpenCL memory API:
    - clCreateBuffer: allocate memory buffer in device memory
    - clEnqueueWriteBuffer: copy data from host to device
    - clEnqueueReadBuffer: copy data from device to host
    - clEnqueueMapBuffer: map buffer to host address space for access
    - clSetKernelArg: bind memory buffer to kernel

  TODOs:
    - Add multi-platform initialization
]#

import std/[macros]

import clwrap
export clwrap

type
  PlatformNotFound = object of CatchableError
  DeviceNotFound = object of CatchableError

type CLResource = PCommandQueue | PKernel | PProgram | PMem | PContext

proc newPlatformNotFound(): ref PlatformNotFound =
  new result
  result.msg = "PlatformNotFound"

proc newDeviceNotFound(): ref DeviceNotFound =
  new result
  result.msg = "DeviceNotFound"

proc name*(id: PPlatformId): string =
  var size = 0
  check getPlatformInfo(id, PLATFORM_NAME, 0, nil, addr size)
  result = newString(size)
  check getPlatformInfo(id, PLATFORM_NAME, size, addr result[0], nil)

proc name*(id: PDeviceId): string =
  var size = 0
  check getDeviceInfo(id, DEVICE_NAME, 0, nil, addr size)
  result = newString(size)
  check getDeviceInfo(id, DEVICE_NAME, size, addr result[0], nil)

proc maxWorkGroups*(id: PDeviceId): int =
  check getDeviceInfo(id, DEVICE_MAX_WORK_GROUP_SIZE, sizeof(int), addr result, nil)

proc localMemory*(id: PDeviceId): uint64 =
  check getDeviceInfo(id, DEVICE_LOCAL_MEM_SIZE, sizeof(int), addr result, nil)

proc globalMemory*(id: PDeviceId): uint64 =
  check getDeviceInfo(id, DEVICE_GLOBAL_MEM_SIZE, sizeof(int), addr result, nil)

proc maxWorkItems*(id: PDeviceId): seq[int] =
  var dims: int
  check getDeviceInfo(id, DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(int), addr dims, nil)
  result = newSeq[int](dims)
  check getDeviceInfo(id, DEVICE_MAX_WORK_ITEM_SIZES, dims * sizeof(int), addr result[0], nil)

proc version*(id: PPlatformId): string =
  var size = 0
  check getPlatformInfo(id, PLATFORM_VERSION, 0, nil, addr size)
  result = newString(size)
  check getPlatformInfo(id, PLATFORM_VERSION, size, addr result[0], nil)

proc getPlatformByName*(platformName: string): PPlatformId =
  var numPlatforms: uint32
  check getPlatformIDs(0, nil, addr numPlatforms)
  var platforms = newSeq[PPlatformId](numPlatforms)
  check getPlatformIDs(numPlatforms, addr platforms[0], nil)

  for platform in platforms:
    if platform.name.substr(0, platformName.high) == platformName:
      return platform

  raise newPlatformNotFound()

proc firstPlatform*(): PPlatformId =
  var numPlatforms: uint32
  check getPlatformIDs(0, nil, addr numPlatforms)
  if numPlatforms == 0:
    raise newPlatformNotFound()
  var platforms = newSeq[PPlatformId](numPlatforms)
  check getPlatformIDs(numPlatforms, addr platforms[0], nil)
  return platforms[0]

proc getDevices*(platform: PPlatformId): seq[PDeviceId] =
  var numDevices: uint32
  check getDeviceIDs(platform, DEVICE_TYPE_ALL, 0, nil, addr numDevices)
  if numDevices == 0:
    raise newDeviceNotFound()

  var devices = newSeq[PDeviceId](numDevices)
  check getDeviceIDs(platform, DEVICE_TYPE_ALL, numDevices, addr devices[0], nil)
  devices

proc createContext*(devices: seq[PDeviceId]): PContext =
  var status: TClResult
  var devs = devices
  result = createContext(nil, devs.len.uint32, cast[ptr PDeviceId](addr devs[0]), nil, nil, addr status)
  check status

proc commandQueueFor*(context: PContext, device: PDeviceId): PCommandQueue =
  var status: TClResult
  result = createCommandQueue(context, device, 0, addr status)
  check status

proc openclDefaults*(): tuple[devices: seq[PDeviceId], context: PContext] =
  let
    platform = firstPlatform()
    devices = platform.getDevices
    context = devices.createContext
  return (devices, context)

proc singleDeviceDefaults*(): tuple[device: PDeviceId, context: PContext, queue: PCommandQueue] =
  let
    platform = firstPlatform()
    device = platform.getDevices[0]
    context = @[device].createContext
    queue = context.commandQueueFor(device)
  return (device, context, queue)

proc multipleDeviceDefaults*(): tuple[devices: seq[PDeviceId], context: PContext, queues: seq[PCommandQueue]] =
  let
    platform = firstPlatform()
    devices = platform.getDevices()
    context = devices.createContext()
  var queues = newSeq[PCommandQueue](devices.len)
  for i in 0..<devices.len: queues[i] = context.commandQueueFor(devices[i])
  return (devices, context, queues)

proc createProgram*(context: PContext, body: string): PProgram =
  var status: TClResult
  var lines = [cstring(body)]
  result = createProgramWithSource(context, 1, cast[cstringArray](addr lines), nil, addr status)
  check status

proc createProgramBinary*(context: PContext, device: PDeviceId, body: string): PProgram =
  var status: TClResult
  var binaryStatus: int32
  var dev = device
  var lines = [cstring(body)]
  var L = body.len
  result = createProgramWithBinary(context, 1, addr dev, addr L, cast[ptr ptr uint8](addr lines), addr binaryStatus, addr status)
  check status

proc buildOn*(program: PProgram, devices: seq[PDeviceId]) =
  var devs = devices
  check buildProgram(program, devs.len.uint32, cast[ptr PDeviceId](addr devs[0]), nil, nil, nil)

proc buildOn*(program: PProgram, device: PDeviceId) = program.buildOn(@[device])

proc createAndBuild*(context: PContext, body: string, devices: seq[PDeviceId]): PProgram =
  result = createProgram(context, body)
  result.buildOn(devices)

proc createAndBuild*(context: PContext, body: string, device: PDeviceId): PProgram =
  result = createProgram(context, body)
  result.buildOn(device)

proc createAndBuildBinary*(context: PContext, body: string, device: PDeviceId): PProgram =
  result = createProgramBinary(context, device, body)
  result.buildOn(device)

proc buffer*[A](context: PContext, size: int, flags: Tmem_flags = MEM_READ_WRITE): PMem =
  var status: TClResult
  result = createBuffer(context, flags, size * sizeof(A), nil, addr status)
  check status

proc bufferLike*[A](context: PContext, xs: openArray[A], flags: Tmem_flags = MEM_READ_WRITE): PMem =
  buffer[A](context, xs.len, flags)

# Typed GPU buffer wrapper - preserves element type information
type GpuBuffer*[T] = object
  mem*: PMem
  len*: int

proc gpuBuffer*[T](context: PContext, size: int, flags: Tmem_flags = MEM_READ_WRITE): GpuBuffer[T] =
  ## Creates a typed GPU buffer that preserves element type information
  var status: TClResult
  result.mem = createBuffer(context, flags, size * sizeof(T), nil, addr status)
  result.len = size
  check status

proc gpuBufferLike*[T](context: PContext, xs: openArray[T], flags: Tmem_flags = MEM_READ_WRITE): GpuBuffer[T] =
  ## Creates a typed GPU buffer with the same size as the input array
  gpuBuffer[T](context, xs.len, flags)

proc write*[T](queue: PCommandQueue, src: var seq[T], dest: GpuBuffer[T]) =
  ## Writes data from a seq to a typed GPU buffer
  check enqueueWriteBuffer(queue, dest.mem, CL_FALSE, 0, src.len * sizeof(T), addr src[0], 0, nil, nil)

proc read*[T](queue: PCommandQueue, dest: var seq[T], src: GpuBuffer[T]) =
  ## Reads data from a typed GPU buffer to a seq
  check enqueueReadBuffer(queue, src.mem, CL_TRUE, 0, dest.len * sizeof(T), addr dest[0], 0, nil, nil)

template release*[T](buffer: GpuBuffer[T]) = check releaseMemObject(buffer.mem)

# Phantom [] operator for GpuBuffer - provides type info for the typed macro
# These are never called at runtime; the body becomes OpenCL code
# The implementation raises an error if somehow called
proc `[]`*[T](buffer: GpuBuffer[T], index: int): T = 
  raise newException(Defect, "GpuBuffer[] should not be called - this is a phantom operator for type inference")
proc `[]=`*[T](buffer: GpuBuffer[T], index: int, value: T) = 
  raise newException(Defect, "GpuBuffer[]= should not be called - this is a phantom operator for type inference")

# Phantom [] operator for PMem - defaults to float64 (type-erased fallback)
# These are never called at runtime; the body becomes OpenCL code
proc `[]`*(buffer: PMem, index: int): float64 = 
  raise newException(Defect, "PMem[] should not be called - this is a phantom operator for type inference")
proc `[]=`*(buffer: PMem, index: int, value: float64) = 
  raise newException(Defect, "PMem[]= should not be called - this is a phantom operator for type inference")

proc buildErrors*(program: PProgram, devices: seq[PDeviceId]): string =
  var logSize: int
  check getProgramBuildInfo(program, devices[0], PROGRAM_BUILD_LOG, 0, nil, addr logSize)
  result = newString(logSize + 1)
  check getProgramBuildInfo(program, devices[0], PROGRAM_BUILD_LOG, logSize, addr result[0], nil)

proc createKernel*(program: PProgram, name: string): PKernel =
  var status: TClResult
  result = createKernel(program, name, addr status)
  check status

type
  LocalBuffer*[A] = distinct int
  anyInt = int or int32 or int64

template setArg*(kernel: PKernel, item: PMem, index: int) =
  var x = item
  check setKernelArg(kernel, index.uint32, sizeof(Pmem), addr x)

template setArg*[T](kernel: PKernel, item: GpuBuffer[T], index: int) =
  var x = item.mem
  check setKernelArg(kernel, index.uint32, sizeof(Pmem), addr x)

template setArg*[A](kernel: PKernel, item: var A, index: int) =
  check setKernelArg(kernel, index.uint32, sizeof(A), addr item)

template setArg*[A](kernel: PKernel, item: LocalBuffer[A], index: int) =
  check setKernelArg(kernel, index.uint32, int(item) * sizeof(A), nil)

template setArg*(kernel: PKernel, item: anyInt, index: int) =
  var x = item
  check setKernelArg(kernel, index.uint32, sizeof(type(item)), addr x)

macro args*(kernel: Pkernel, args: varargs[untyped]): untyped =
  result = newStmtList()

  var i = 0 # no pairs for macro for loop
  for arg in items(args):
    let s = quote do:
      `kernel`.setArg(`arg`, `i`)
    result.add(s)
    inc i

proc run*(queue: PCommandQueue, kernel: PKernel, totalWork: int) =
  var globalWorkSize = [totalWork, 0, 0]
  check enqueueNDRangeKernel(queue, kernel, 1, nil,  cast[ptr int](addr globalWorkSize), nil, 0, nil, nil)

proc run*(queue: PCommandQueue, kernel: PKernel, totalWork, localWork: int) =
  var
    globalWorkSize = [totalWork, 0, 0]
    localWorkSize = [localWork, 0, 0]
  check enqueueNDRangeKernel(queue, kernel, 1, nil,  cast[ptr int](addr globalWorkSize), cast[ptr int](addr localWorkSize), 0, nil, nil)

proc run2d*(queue: PCommandQueue, kernel: PKernel, totalWork: (int, int)) =
  let (a, b) = totalWork
  var globalWorkSize = [a, b, 0]
  check enqueueNDRangeKernel(queue, kernel, 2, nil,  cast[ptr int](addr globalWorkSize), nil, 0, nil, nil)

proc run2d*(queue: PCommandQueue, kernel: PKernel, totalWork, localWork: (int, int)) =
  let
    (a, b) = totalWork
    (c, d) = localWork
  var
    globalWorkSize = [a, b, 0]
    localWorkSize = [c, d, 0]
  check enqueueNDRangeKernel(queue, kernel, 2, nil,  cast[ptr int](addr globalWorkSize), cast[ptr int](addr localWorkSize), 0, nil, nil)

proc run3d*(queue: PCommandQueue, kernel: PKernel, totalWork: (int, int, int)) =
  let (a, b, c) = totalWork
  var globalWorkSize = [a, b, c]
  check enqueueNDRangeKernel(queue, kernel, 3, nil,  cast[ptr int](addr globalWorkSize), nil, 0, nil, nil)

proc run3d*(queue: PCommandQueue, kernel: PKernel, totalWork, localWork: (int, int, int)) =
  let
    (a, b, c) = totalWork
    (d, e, f) = localWork
  var
    globalWorkSize = [a, b, c]
    localWorkSize = [d, e, f]
  check enqueueNDRangeKernel(queue, kernel, 3, nil,  cast[ptr int](addr globalWorkSize), cast[ptr int](addr localWorkSize), 0, nil, nil)

proc write*(queue: PCommandQueue, src: pointer, dest: PMem, size: int) =
  check enqueueWriteBuffer(queue, dest, CL_FALSE, 0, size, src, 0, nil, nil)

proc write*[A](queue: PCommandQueue, src: var seq[A], dest: PMem) =
  write(queue, addr src[0], dest, src.len * sizeof(A))

proc read*(queue: PCommandQueue, dest: pointer, src: PMem, size: int) =
  check enqueueReadBuffer(queue, src, CL_TRUE, 0, size, dest, 0, nil, nil)

proc read*[A](queue: PCommandQueue, dest: var seq[A], src: PMem) =
  read(queue, addr dest[0], src, dest.len * sizeof(A))

template release*(queue: PCommandQueue) = check releaseCommandQueue(queue)
template release*(kernel: PKernel) = check releaseKernel(kernel)
template release*(program: PProgram) = check releaseProgram(program)
template release*(buffer: PMem) = check releaseMemObject(buffer)
template release*(context: PContext) = check releaseContext(context)

# OpenCL initialization and finalization

template initCL*: untyped =
  let (
    clDevices {.inject.}, 
    clContext {.inject.}, 
    clQueues  {.inject.}
  ) = multipleDeviceDefaults()

template finalizeCL*: untyped =
  for queue in clQueues: release(queue)
  release(clContext)

when isMainModule:
  # OpenCL initialization: assuming single platform (TODO: handle multiple platforms),
  # detect platform devices and create both context and commend queue from the 
  # available devices
  initCL()

  for device in clDevices:
    echo "Device: ", device.name
    echo "  Max work groups: ", device.maxWorkGroups
    echo "  Max work items: ", device.maxWorkItems
    echo "  Local memory: ", device.localMemory div 1024, " KB"
    echo "  Global memory: ", device.globalMemory div (1024*1024), " MB"
  
  let
    program = clContext.createAndBuild("""
    __kernel void vadd(
      __global const float* a,
      __global const float* b,
      __global float* c
    ) { int gid = get_global_id(0); c[gid] = a[gid] + b[gid]; }
    """, clDevices)
    size = 1_000_000
  
  var kernel = program.createKernel("vadd")

  var
    a = newSeq[float32](size)
    b = newSeq[float32](size)
    c = newSeq[float32](size)
  
  var
    gpuA = clContext.bufferLike(a)
    gpuB = clContext.bufferLike(b)
    gpuC = clContext.bufferLike(c)
  
  kernel.args(gpuA, gpuB, gpuC)

  for i in 0..<size:
    a[i] = float(i)
    b[i] = float(2*i)
  
  clQueues[0].write(a, gpuA)
  clQueues[0].write(b, gpuB)
  clQueues[0].run(kernel, size)
  clQueues[0].read(c, gpuC)

  for i in 0..<size:
    assert c[i] == a[i] + b[i], "Error at index " & $i
  
  release(kernel)
  release(program)
  release(gpuA)
  release(gpuB)
  release(gpuC)

  finalizeCL()

#[ ============================================================================
   OpenCL DSL Pragmas and Stub Functions
   ============================================================================
   These are used by the compile-time Nim-to-OpenCL compiler.
   The stub implementations allow typed Nim code to compile on the host,
   while the names are preserved in the generated OpenCL C code.
]#

# Pragmas for marking kernels and memory qualifiers
template kernel*() {.pragma.}     ## Mark proc as __kernel
template device*() {.pragma.}     ## Helper function (no qualifier in OpenCL)
template global*() {.pragma.}     ## __global memory qualifier
template local*() {.pragma.}      ## __local memory qualifier
template constant*() {.pragma.}   ## __constant memory qualifier
template oclName*(s: string): untyped {.pragma.}  ## Custom OpenCL name

# Work-item functions
type Dim* = cint

proc get_global_id_impl(dim: Dim): Dim = 0
proc get_local_id_impl(dim: Dim): Dim = 0
proc get_group_id_impl(dim: Dim): Dim = 0
proc get_global_size_impl(dim: Dim): Dim = 0
proc get_local_size_impl(dim: Dim): Dim = 0
proc get_num_groups_impl(dim: Dim): Dim = 0
proc get_work_dim_impl(): Dim = 0
proc get_global_offset_impl(dim: Dim): Dim = 0

template get_global_id*(dim: Dim): Dim = get_global_id_impl(dim)
template get_local_id*(dim: Dim): Dim = get_local_id_impl(dim)
template get_group_id*(dim: Dim): Dim = get_group_id_impl(dim)
template get_global_size*(dim: Dim): Dim = get_global_size_impl(dim)
template get_local_size*(dim: Dim): Dim = get_local_size_impl(dim)
template get_num_groups*(dim: Dim): Dim = get_num_groups_impl(dim)
template get_work_dim*(): Dim = get_work_dim_impl()
template get_global_offset*(dim: Dim): Dim = get_global_offset_impl(dim)

# Synchronization functions
proc barrier_impl*(flags: cint) = discard
template barrier*(flags: cint) = barrier_impl(flags)

proc mem_fence_impl*(flags: cint) = discard
template mem_fence*(flags: cint) = mem_fence_impl(flags)

proc read_mem_fence_impl*(flags: cint) = discard
template read_mem_fence*(flags: cint) = read_mem_fence_impl(flags)

proc write_mem_fence_impl*(flags: cint) = discard
template write_mem_fence*(flags: cint) = write_mem_fence_impl(flags)

const CLK_LOCAL_MEM_FENCE* = 1.cint
const CLK_GLOBAL_MEM_FENCE* = 2.cint

# Atomic functions (int32)
proc atomic_add_impl*(p: ptr int32, val: int32): int32 = discard
proc atomic_sub_impl*(p: ptr int32, val: int32): int32 = discard
proc atomic_xchg_impl*(p: ptr int32, val: int32): int32 = discard
proc atomic_inc_impl*(p: ptr int32): int32 = discard
proc atomic_dec_impl*(p: ptr int32): int32 = discard
proc atomic_cmpxchg_impl*(p: ptr int32, cmp, val: int32): int32 = discard
proc atomic_min_impl*(p: ptr int32, val: int32): int32 = discard
proc atomic_max_impl*(p: ptr int32, val: int32): int32 = discard
proc atomic_and_impl*(p: ptr int32, val: int32): int32 = discard
proc atomic_or_impl*(p: ptr int32, val: int32): int32 = discard
proc atomic_xor_impl*(p: ptr int32, val: int32): int32 = discard

template atomic_add*(p: ptr int32, val: int32): int32 = atomic_add_impl(p, val)
template atomic_sub*(p: ptr int32, val: int32): int32 = atomic_sub_impl(p, val)
template atomic_xchg*(p: ptr int32, val: int32): int32 = atomic_xchg_impl(p, val)
template atomic_inc*(p: ptr int32): int32 = atomic_inc_impl(p)
template atomic_dec*(p: ptr int32): int32 = atomic_dec_impl(p)
template atomic_cmpxchg*(p: ptr int32, cmp, val: int32): int32 = atomic_cmpxchg_impl(p, cmp, val)
template atomic_min*(p: ptr int32, val: int32): int32 = atomic_min_impl(p, val)
template atomic_max*(p: ptr int32, val: int32): int32 = atomic_max_impl(p, val)
template atomic_and*(p: ptr int32, val: int32): int32 = atomic_and_impl(p, val)
template atomic_or*(p: ptr int32, val: int32): int32 = atomic_or_impl(p, val)
template atomic_xor*(p: ptr int32, val: int32): int32 = atomic_xor_impl(p, val)

# Math functions (float32)
proc sin*(x: float32): float32 {.importc: "sinf", nodecl.}
proc cos*(x: float32): float32 {.importc: "cosf", nodecl.}
proc tan*(x: float32): float32 {.importc: "tanf", nodecl.}
proc asin*(x: float32): float32 {.importc: "asinf", nodecl.}
proc acos*(x: float32): float32 {.importc: "acosf", nodecl.}
proc atan*(x: float32): float32 {.importc: "atanf", nodecl.}
proc atan2*(y, x: float32): float32 {.importc: "atan2f", nodecl.}
proc sinh*(x: float32): float32 {.importc: "sinhf", nodecl.}
proc cosh*(x: float32): float32 {.importc: "coshf", nodecl.}
proc tanh*(x: float32): float32 {.importc: "tanhf", nodecl.}
proc exp*(x: float32): float32 {.importc: "expf", nodecl.}
proc exp2*(x: float32): float32 {.importc: "exp2f", nodecl.}
proc log*(x: float32): float32 {.importc: "logf", nodecl.}
proc log2*(x: float32): float32 {.importc: "log2f", nodecl.}
proc log10*(x: float32): float32 {.importc: "log10f", nodecl.}
proc pow*(x, y: float32): float32 {.importc: "powf", nodecl.}
proc sqrt*(x: float32): float32 {.importc: "sqrtf", nodecl.}
proc rsqrt_impl*(x: float32): float32 = 1.0f / sqrt(x)
template rsqrt*(x: float32): float32 = rsqrt_impl(x)
proc fabs*(x: float32): float32 {.importc: "fabsf", nodecl.}
proc floor*(x: float32): float32 {.importc: "floorf", nodecl.}
proc ceil*(x: float32): float32 {.importc: "ceilf", nodecl.}
proc round*(x: float32): float32 {.importc: "roundf", nodecl.}
proc trunc*(x: float32): float32 {.importc: "truncf", nodecl.}
proc fmod*(x, y: float32): float32 {.importc: "fmodf", nodecl.}
proc fmin*(x, y: float32): float32 {.importc: "fminf", nodecl.}
proc fmax*(x, y: float32): float32 {.importc: "fmaxf", nodecl.}
proc fma*(a, b, c: float32): float32 {.importc: "fmaf", nodecl.}

proc clamp_impl*(x, minVal, maxVal: float32): float32 = fmin(fmax(x, minVal), maxVal)
template clamp*(x, minVal, maxVal: float32): float32 = clamp_impl(x, minVal, maxVal)

# Native (fast) math functions - stubs that call regular versions
proc native_sin_impl*(x: float32): float32 = sin(x)
proc native_cos_impl*(x: float32): float32 = cos(x)
proc native_tan_impl*(x: float32): float32 = tan(x)
proc native_exp_impl*(x: float32): float32 = exp(x)
proc native_exp2_impl*(x: float32): float32 = exp2(x)
proc native_log_impl*(x: float32): float32 = log(x)
proc native_log2_impl*(x: float32): float32 = log2(x)
proc native_log10_impl*(x: float32): float32 = log10(x)
proc native_sqrt_impl*(x: float32): float32 = sqrt(x)
proc native_rsqrt_impl*(x: float32): float32 = rsqrt(x)
proc native_recip_impl*(x: float32): float32 = 1.0f / x
proc native_powr_impl*(x, y: float32): float32 = pow(x, y)

template native_sin*(x: float32): float32 = native_sin_impl(x)
template native_cos*(x: float32): float32 = native_cos_impl(x)
template native_tan*(x: float32): float32 = native_tan_impl(x)
template native_exp*(x: float32): float32 = native_exp_impl(x)
template native_exp2*(x: float32): float32 = native_exp2_impl(x)
template native_log*(x: float32): float32 = native_log_impl(x)
template native_log2*(x: float32): float32 = native_log2_impl(x)
template native_log10*(x: float32): float32 = native_log10_impl(x)
template native_sqrt*(x: float32): float32 = native_sqrt_impl(x)
template native_rsqrt*(x: float32): float32 = native_rsqrt_impl(x)
template native_recip*(x: float32): float32 = native_recip_impl(x)
template native_powr*(x, y: float32): float32 = native_powr_impl(x, y)