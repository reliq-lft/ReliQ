# Compute Backends

ReliQ supports three compute backends, each implementing the same ``each``
macro interface.  User code is backend-agnostic — the same ``for n in each``
loop works across all three backends.

## Backend Comparison

| Feature | OpenCL | SYCL | OpenMP |
|---------|--------|------|--------|
| Compilation | JIT (runtime) | Pre-compiled (C++) | Source-level (Nim→C) |
| Target | GPUs, FPGAs, CPUs | Intel/AMD GPUs, CPUs | CPUs only |
| Requirements | OpenCL runtime | Intel oneAPI (icpx) | GCC/Clang with OpenMP |
| Compile flag | *(default)* | ``BACKEND=sycl`` | ``BACKEND=openmp`` |
| SIMD | GPU warps | GPU subgroups | Explicit intrinsics |
| Memory | OpenCL buffers | SYCL USM/buffers | Shared memory (AoSoA) |
| Test count | 245 tests | 245 tests | 295 tests |

## OpenCL Backend (Default)

The OpenCL backend generates kernel source code at compile time and JIT
compiles it at runtime.  This is the default backend.

### How It Works

The ``each`` macro in ``opencl/cldisp`` receives the loop body as an AST and:

1. **Classifies** each assignment expression into a dispatch kind
   (copy, add, matmul, scalar-mul, etc.)
2. **Gathers** all ``TensorFieldView`` symbols referenced in the loop
3. **Detects** stencil neighbor accesses
4. **Generates** an OpenCL C kernel source string with:
   - Buffer parameters for each view
   - Element type declarations (``float``/``double``/``int``/``long``)
   - Site indexing with ``get_global_id(0)``
   - Inlined arithmetic matching the expression pattern
5. **JIT-compiles** the kernel using ``clCreateProgramWithSource``
   and ``clBuildProgram``
6. **Dispatches** with ``clEnqueueNDRangeKernel``

### Expression Kinds

| Kind | Example | Kernel pattern |
|------|---------|---------------|
| Copy | ``vC[n] = vA[n]`` | ``C[i] = A[i]`` |
| Add | ``vC[n] = vA[n] + vB[n]`` | Element-wise sum |
| Sub | ``vC[n] = vA[n] - vB[n]`` | Element-wise diff |
| MatMul | ``vC[n] = vA[n] * vB[n]`` | ``C[i,j] = Σ A[i,k]*B[k,j]`` |
| MatVec | ``vC[n] = vA[n] * vB[n]`` | ``C[i] = Σ A[i,k]*B[k]`` |
| ScalarMul | ``vC[n] = 3.0 * vA[n]`` | ``C[i] = s * A[i]`` |
| ScalarAdd | ``vC[n] = vA[n] + 1.0`` | ``C[i] = A[i] + s`` |
| StencilCopy | ``vC[n] = vA[fwd]`` | Offset index lookup |

### Debug Kernels

Compile with ``-d:DebugKernels`` to print generated OpenCL kernel source to
stdout at compile time.

### Modules

| Module | Description |
|--------|-------------|
| [opencl/cldisp](opencl/cldisp.html) | ``each`` macro → OpenCL kernel generation |
| [opencl/clbase](opencl/clbase.html) | Platform, device, context, buffer management |

### OpenCL Base Layer

``opencl/clbase`` provides the OpenCL platform and device management:

```nim
# Initialization (called automatically by TensorFieldView constructors)
initCL()
finalizeCL()

# Manual platform selection
let platform = firstPlatform()
let devices = getDevices(platform)
echo platform.name  # e.g., "NVIDIA CUDA"

# Device properties
echo devices[0].globalMemory
echo devices[0].maxWorkItems
```

## SYCL Backend

The SYCL backend dispatches to pre-compiled C++ kernel templates in a shared
library (``libreliq_sycl.so``), avoiding JIT compilation overhead.

### How It Works

The ``each`` macro in ``sycl/sycldisp``:

1. **Analyzes** the AST (same expression classification as OpenCL)
2. **Builds an execution plan** for complex expressions involving
   temporaries
3. **Dispatches** to typed C++ template functions via FFI:
   - ``sycl_kernel_copy_f64(queue, bufA, bufC, nSites, elems)``
   - ``sycl_kernel_matmul_f64(queue, bufA, bufB, bufC, nSites, rows, cols)``
   - etc.
4. Each kernel function is a ``sycl::handler::parallel_for`` with
   type-specialized inner loops

### Building the SYCL Wrapper

```
# Build libreliq_sycl.so (requires Intel oneAPI or hipSYCL)
make sycl-lib

# Then build/test with SYCL backend
make tensorview BACKEND=sycl
make test-sycl
```

### Type-Specialized Kernels

The SYCL wrapper provides kernels for each element type:

- ``float32`` (``sycl_kernel_*_f32``)
- ``float64`` (``sycl_kernel_*_f64``)
- ``int32`` (``sycl_kernel_*_i32``)
- ``int64`` (``sycl_kernel_*_i64``)

Plus complex-number variants for ``Complex64`` fields.

### Stencil Kernels

Pre-compiled stencil kernels accept an offset buffer and perform neighbor
lookups on-device:

| Kernel | Description |
|--------|-------------|
| ``kernelStencilCopy`` | Copy from neighbor site |
| ``kernelStencilScalarMul`` | Scalar × neighbor value |
| ``kernelStencilAdd`` | Sum of site and neighbor |

### Modules

| Module | Description |
|--------|-------------|
| [sycl/sycldisp](sycl/sycldisp.html) | ``each`` macro → native SYCL dispatch |
| [sycl/syclbase](sycl/syclbase.html) | Queue/buffer management |
| [sycl/syclwrap](sycl/syclwrap.html) | Low-level C++ FFI (60+ kernel functions) |

## OpenMP Backend

The OpenMP backend generates SIMD-vectorized C code with explicit intrinsic
calls, targeting CPU architectures.

### How It Works

The ``each`` macro in ``openmp/ompdisp``:

1. **Analyzes** the AST (same expression classification)
2. **Determines** if SIMD vectorization is applicable
3. **Generates** C code with:
   - ``#pragma omp parallel for`` for outer loop parallelism
   - SIMD intrinsics (AVX2/AVX-512) for inner loop vectorization
   - AoSoA memory access pattern matching the ``SimdLatticeLayout``

### SIMD-Vectorized Views

When using the OpenMP backend, ``TensorFieldView`` can use a SIMD-aware
AoSoA layout:

```nim
# Default SIMD grid (auto-distributed based on VectorWidth)
var vA = localA.newTensorFieldView(iokRead)

# Explicit SIMD grid
var vB = localB.newTensorFieldView(iokRead, [1, 1, 1, 8])
```

The SIMD grid controls how lattice sites are grouped into SIMD lanes.
With ``VectorWidth=8`` (AVX-512), 8 consecutive sites along the innermost
SIMD dimension are processed simultaneously.

### Loop Patterns

The SIMD backend uses an outer/inner loop pattern:

```nim
# Outer loop: iterates over SIMD groups (OpenMP parallel)
# Inner loop: iterates over SIMD lanes (vectorized)
for outer in 0..<nSitesOuter:
  for lane in 0..<VectorWidth:
    # Each lane processes one site within the SIMD group
```

The ``eachOuter`` macro in ``openmp/ompsimd`` provides direct access to this
pattern for low-level SIMD programming.

### SIMD Intrinsics

The OpenMP backend can use hardware-specific SIMD intrinsics:

```nim
# SIMD vector operations (compile with -d:AVX2 or -d:AVX512)
import reliq

var a = SimdF64x4(data: [1.0, 2.0, 3.0, 4.0])
var b = SimdF64x4(data: [5.0, 6.0, 7.0, 8.0])
var c = a + b              # [6.0, 8.0, 10.0, 12.0]
let s = a.sum()            # 10.0
```

### Modules

| Module | Description |
|--------|-------------|
| [openmp/ompdisp](openmp/ompdisp.html) | ``each`` macro → SIMD-vectorized code |
| [openmp/ompbase](openmp/ompbase.html) | OpenMP initialization, thread management |
| [openmp/ompsimd](openmp/ompsimd.html) | SIMD-aware dispatch, ``eachOuter`` macro |

## Building and Testing

```
# Build with specific backend
make tensorview                    # OpenCL (default)
make tensorview BACKEND=openmp     # OpenMP
make tensorview BACKEND=sycl       # SYCL

# Run all backend tests
make test          # All backends (1,660 tests)
make test-core     # Core tests (875)
make test-opencl   # OpenCL tests (245)
make test-openmp   # OpenMP tests (295)
make test-sycl     # SYCL tests (245)
```
