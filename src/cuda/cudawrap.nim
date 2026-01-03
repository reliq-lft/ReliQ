#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/hip/hipwrap.nim
  Contact: reliq-lft@proton.me

  Author: Andrew Brower <monofuel@japura.net>
  Modifications: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  Original Software License:

  MIT License
  
  Copyright (c) 2024 Author
  
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

  ReliQ Modifications License:

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

# CUDA runtime C++ FFI
import std/strformat

type
  size_t* = uint64
  uint8_t* = uint8
  uint16_t* = uint16
  uint32_t* = uint32
  uint64_t* = uint64
  cudaStream_t* = pointer
  cudaError_t* {.importcpp: "cudaError_t", header: "cuda_runtime.h".} = cint

type
  cudaMemcpyKind* {.size: sizeof(cint), header: "cuda_runtime.h", importcpp: "cudaMemcpyKind".} = enum
    cudaMemcpyHostToHost = 0,    ## < Host-to-Host Copy
    cudaMemcpyHostToDevice = 1,  ## < Host-to-Device Copy
    cudaMemcpyDeviceToHost = 2,  ## < Device-to-Host Copy
    cudaMemcpyDeviceToDevice = 3, ## < Device-to-Device Copy
    cudaMemcpyDefault = 4        ## < Runtime will automatically determine copy-kind based on virtual addresses.

type
  Dim3* {.importcpp: "dim3", header: "cuda_runtime.h", bycopy.} = object
    x* {.importc: "x".}: uint32_t ## < x
    y* {.importc: "y".}: uint32_t ## < y
    z* {.importc: "z".}: uint32_t ## < z
  BlockDim* {.importcpp: "const __cuda_Coordinates<__cuda_BlockDim>", header: "cuda_runtime.h".} = object
    x* {.importc: "x".}: uint32_t ## < x
    y* {.importc: "y".}: uint32_t ## < y
    z* {.importc: "z".}: uint32_t ## < z
  BlockIdx* {.importcpp: "const __cuda_Coordinates<__cuda_BlockIdx>", header: "cuda_runtime.h".} = object
    x* {.importc: "x".}: uint32_t ## < x
    y* {.importc: "y".}: uint32_t ## < y
    z* {.importc: "z".}: uint32_t ## < z
  GridDim* {.importcpp: "const __cuda_Coordinates<__cuda_GridDim>", header: "cuda_runtime.h".} = object
    x* {.importc: "x".}: uint32_t ## < x
    y* {.importc: "y".}: uint32_t ## < y
    z* {.importc: "z".}: uint32_t ## < z
  ThreadIdx* {.importcpp: "const __cuda_Coordinates<__cuda_ThreadIdx>", header: "cuda_runtime.h".} = object
    x* {.importc: "x".}: uint32_t ## < x
    y* {.importc: "y".}: uint32_t ## < y
    z* {.importc: "z".}: uint32_t ## < z

proc newDim3*(x: uint32_t = 1; y: uint32_t = 1; z: uint32_t = 1): Dim3 =
  result.x = x
  result.y = y
  result.z = z

# CUDA built-in variables (only available in device code)
var blockDim* {.importcpp: "blockDim", header: "cuda_runtime.h", nodecl.}: BlockDim
var blockIdx* {.importcpp: "blockIdx", header: "cuda_runtime.h", nodecl.}: BlockIdx
var gridDim* {.importcpp: "gridDim", header: "cuda_runtime.h", nodecl.}: GridDim
var threadIdx* {.importcpp: "threadIdx", header: "cuda_runtime.h", nodecl.}: ThreadIdx

# CUDA kernel pragmas - use these for device functions
template cudaGlobal* {.pragma.}
template cudaDevice* {.pragma.}
template cudaHost* {.pragma.}
template cudaShared* {.pragma.}

proc cudaMalloc*(`ptr`: ptr pointer; size: cint): cudaError_t {.header: "cuda_runtime.h",importcpp: "cudaMalloc(@)".}
proc cudaMemcpy*(dst: pointer; src: pointer; size: cint; kind: cudaMemcpyKind): cudaError_t {.header: "cuda_runtime.h",importcpp: "cudaMemcpy(@)".}
proc cudaFree*(`ptr`: pointer): cudaError_t {.header: "cuda_runtime.h",importcpp: "cudaFree(@)".}

proc cudaLaunchKernel*(function_address: pointer; numBlocks: Dim3; dimBlocks: Dim3;
                     args: ptr pointer): cudaError_t {.
    importcpp: "cudaLaunchKernel(@)", header: "cuda_runtime.h".}
proc cudaLaunchKernel*(function_address: pointer; numBlocks: Dim3; dimBlocks: Dim3;
                     args: ptr pointer, sharedMemBytes: uint32_t, stream: cudaStream_t): cudaError_t {.
    importcpp: "cudaLaunchKernel(@)", header: "cuda_runtime.h".}
proc cudaDeviceSynchronize*(): cudaError_t {.header: "cuda_runtime.h",importcpp: "cudaDeviceSynchronize(@)".}
proc cudaSyncthreads*() {.importcpp: "__syncthreads()", header: "cuda_runtime.h".}
proc hippoSyncthreads*() {.importcpp: "__syncthreads()", header: "cuda_runtime.h".}

proc cudaLaunchKernelGGL*(
  function_address: proc;
  numBlocks: Dim3;
  dimBlocks: Dim3;
  sharedMemBytes: uint32_t;
  stream: cudaStream_t;
  ) {.
    importcpp: "cudaLaunchKernelGGL(@)", header: "cuda_runtime.h", varargs.}


type ConstCString* {.importc: "const char*".} = object
converter toCString*(self: ConstCString): cstring {.importc: "(char*)", noconv, nodecl.}
converter toConstCString*(self: cstring): ConstCString {.importc: "(const char*)", noconv, nodecl.}
proc `$`*(self: ConstCString): string = $(self.toCString())
proc cudaGetErrorString*(err: cudaError_t): ConstCString {.header: "cuda_runtime.h",importcpp: "cudaGetErrorString(@)".}
proc cudaGetLastError*(): cudaError_t {.header: "cuda_runtime.h",importcpp: "cudaGetLastError()".}

# Error Helpers
proc handleError*(err: cudaError_t) =
  if err != 0:
    var cstr = cudaGetErrorString(err).toCString
    raise newException(Exception, &"CUDA Error: " & $cstr)

# CUDA Math Functions
# Single-precision floating-point math functions available in device code
proc expf*(x: cfloat): cfloat {.header: "cuda_runtime.h", importcpp: "expf(@)".}
proc logf*(x: cfloat): cfloat {.header: "cuda_runtime.h", importcpp: "logf(@)".}
proc sinf*(x: cfloat): cfloat {.header: "cuda_runtime.h", importcpp: "sinf(@)".}
proc cosf*(x: cfloat): cfloat {.header: "cuda_runtime.h", importcpp: "cosf(@)".}
proc sqrtf*(x: cfloat): cfloat {.header: "cuda_runtime.h", importcpp: "sqrtf(@)".}
proc powf*(base: cfloat, exp: cfloat): cfloat {.header: "cuda_runtime.h", importcpp: "powf(@)".}

# Double-precision floating-point math functions
proc exp*(x: cdouble): cdouble {.header: "cuda_runtime.h", importcpp: "exp(@)".}
proc log*(x: cdouble): cdouble {.header: "cuda_runtime.h", importcpp: "log(@)".}
proc sin*(x: cdouble): cdouble {.header: "cuda_runtime.h", importcpp: "sin(@)".}
proc cos*(x: cdouble): cdouble {.header: "cuda_runtime.h", importcpp: "cos(@)".}
proc sqrt*(x: cdouble): cdouble {.header: "cuda_runtime.h", importcpp: "sqrt(@)".}
proc pow*(base: cdouble, exp: cdouble): cdouble {.header: "cuda_runtime.h", importcpp: "pow(@)".}

#[ CUDA device code and tests ]#

when isMainModule:
  # Inline CUDA kernel using cudaGlobal pragma (hippo-style)
  # Must disable Nim runtime checks for device code
  {.push stackTrace:off, lineTrace:off, checks:off.}
  proc vectorAdd(a: ptr cfloat, b: ptr cfloat, c: ptr cfloat, n: cint) {.
    cudaGlobal,
    codegenDecl: "__global__ $# $#$#".} =
    let idx = cint(blockIdx.x * blockDim.x + threadIdx.x)
    if idx < n:
      let a_arr = cast[ptr UncheckedArray[cfloat]](a)
      let b_arr = cast[ptr UncheckedArray[cfloat]](b)
      let c_arr = cast[ptr UncheckedArray[cfloat]](c)
      c_arr[idx] = a_arr[idx] + b_arr[idx]
  {.pop.}
  
  echo "CUDA Test: Vector Addition"
  
  # Test parameters
  const N = 1024
  let size = cint(N * sizeof(cfloat))
  
  # Allocate host memory
  var h_a = newSeq[cfloat](N)
  var h_b = newSeq[cfloat](N)
  var h_c = newSeq[cfloat](N)
  
  # Initialize host arrays
  for i in 0..<N:
    h_a[i] = cfloat(i)
    h_b[i] = cfloat(i * 2)
  
  # Allocate device memory
  var d_a, d_b, d_c: pointer
  handleError(cudaMalloc(addr d_a, size))
  handleError(cudaMalloc(addr d_b, size))
  handleError(cudaMalloc(addr d_c, size))
  
  echo "Memory allocated on device"
  
  # Copy data to device
  handleError(cudaMemcpy(d_a, addr h_a[0], size, cudaMemcpyHostToDevice))
  handleError(cudaMemcpy(d_b, addr h_b[0], size, cudaMemcpyHostToDevice))
  
  echo "Data copied to device"
  
  # Launch kernel
  let threadsPerBlock = 256
  let blocksPerGrid = (N + threadsPerBlock - 1) div threadsPerBlock
  let grid = newDim3(uint32(blocksPerGrid))
  let blk = newDim3(uint32(threadsPerBlock))
  
  var n_val = cint(N)
  var kernel_args = [cast[pointer](d_a), cast[pointer](d_b), cast[pointer](d_c), cast[pointer](addr n_val)]
  handleError(cudaLaunchKernel(cast[pointer](vectorAdd), grid, blk, addr kernel_args[0]))
  handleError(cudaDeviceSynchronize())
  
  echo "Kernel executed"
  
  # Copy result back to host
  handleError(cudaMemcpy(addr h_c[0], d_c, size, cudaMemcpyDeviceToHost))
  
  echo "Result copied back to host"
  
  # Verify results
  var success = true
  for i in 0..<N:
    let expected = h_a[i] + h_b[i]
    if abs(h_c[i] - expected) > 1e-5:
      echo &"Mismatch at index {i}: {h_c[i]} != {expected}"
      success = false
      break
  
  # Cleanup
  handleError(cudaFree(d_a))
  handleError(cudaFree(d_b))
  handleError(cudaFree(d_c))
  
  if success:
    echo "CUDA test PASSED - Vector addition successful!"
  else:
    echo "CUDA test FAILED"
    quit(1)