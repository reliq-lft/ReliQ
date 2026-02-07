# ReliQ Lattice Field Theory Framework

> *"Documentation is a love letter that you write to your future self."* — Damian Conway

## Overview

**ReliQ** is an experimental lattice field theory (LFT) framework written in
[Nim](https://nim-lang.org/), designed for user-friendliness, performance,
reliability, and portability across current and future heterogeneous
architectures.

### Key Features

- **Multi-backend dispatch**: OpenCL (GPU JIT), SYCL (pre-compiled GPU/CPU),
  OpenMP (SIMD-vectorized CPU) — all sharing the same user-facing API
- **AoSoA memory layout**: Array of Structures of Arrays for optimal
  SIMD and GPU memory coalescing
- **Distributed memory**: MPI-based parallelism via
  [Global Arrays](https://globalarrays.github.io/) with automatic ghost
  region management and distributed transport (`GlobalShifter`,
  `discreteLaplacian`)
- **Compile-time tensor types**: `Vec[N,T]` and `Mat[N,M,T]` with
  dimensions known at compile time for zero-overhead abstraction
- **Unified stencil operations**: A single `LatticeStencil[D]` type that
  works identically across all backends, handling periodic boundary
  conditions and ghost regions automatically
- **Macro-based dispatch**: The `each` macro analyzes loop bodies at
  compile time and generates optimized backend-specific code

## Quick Start

```nim
import reliq

# Create a 4D lattice (8×8×8×16)
let lat = newSimpleCubicLattice([8, 8, 8, 16])

# Create a scalar field on the lattice
var phi = newTensorField[4, 1, 1, float64](lat)
var psi = newTensorField[4, 1, 1, float64](lat)

# Create views for GPU/SIMD dispatch
var vPhi = newTensorFieldView(phi)
var vPsi = newTensorFieldView(psi)

# Initialize psi to 1.0 on all sites
each vPsi, n:
  vPsi[n] = 1.0

# Nearest-neighbor stencil for computing Laplacian
let stencil = newLatticeStencil(nearestNeighborStencil(lat), lat)

# Access neighbor data in each loops
each vPhi, vPsi, stencil, n:
  let nbrIdx = stencil.neighbor(n, 0)  # Forward x-neighbor
  vPhi[n] = vPsi[nbrIdx]
```

## Architecture

ReliQ is organized into several layers:

```
┌──────────────────────────────────────────────┐
│                User Code                     │
│          import reliq; each v, n: ...        │
├──────────────────────────────────────────────┤
│             Tensor Layer                     │
│  TensorFieldView · Stencil · Transporter     │
│  GlobalShifter  · Discrete Laplacian         │
├──────────────────────────────────────────────┤
│           Backend Dispatch                   │
│    OpenCL    │    SYCL    │    OpenMP         │
│   (cldisp)   │  (sycldisp) │  (ompdisp)      │
├──────────────────────────────────────────────┤
│         Memory & Communication               │
│   Global Arrays · MPI · AoSoA Layout         │
└──────────────────────────────────────────────┘
```

## Module Reference

### Core Modules

- [reliq](reliq.html) — Top-level entry point; re-exports all public modules
- [lattice](lattice.html) — Lattice types and concepts
- [parallel](parallel.html) — Backend-agnostic parallel dispatch

### Lattice

- [lattice/latticeconcept](lattice/latticeconcept.html) — The `Lattice[D]`
  concept that all lattice types must satisfy
- [lattice/simplecubiclattice](lattice/simplecubiclattice.html) —
  `SimpleCubicLattice[D]` implementation with MPI decomposition
- [lattice/stencil](lattice/stencil.html) — Unified stencil operations:
  `LatticeStencil[D]`, `StencilPattern[D]`, neighbor access, shift API,
  direction types
- [lattice/indexing](lattice/indexing.html) — Coordinate ↔ flat index
  conversion utilities

### Tensor Fields

- [tensor/tensor](tensor/tensor.html) — Tensor module aggregation;
  re-exports all tensor submodules
- [tensor/globaltensor](tensor/globaltensor.html) — `TensorField[D,R,L,T]`:
  distributed tensor field with ghost region support, coordinate utilities
  (C row-major GA layout), `GlobalShifter` for distributed transport,
  `discreteLaplacian`, and `LatticeStencil` integration
- [tensor/localtensor](tensor/localtensor.html) —
  `LocalTensorField[D,R,L,T]`: rank-local tensor data in host memory,
  element/site accessors, arithmetic, and norms
- [tensor/tensorview](tensor/tensorview.html) — `TensorFieldView[L,T]`:
  the primary type for GPU/SIMD dispatch via the `each` macro; handles
  AoSoA layout transformation across OpenCL, SYCL, and OpenMP backends
- [tensor/sitetensor](tensor/sitetensor.html) — `Vec[N,T]` and
  `Mat[N,M,T]`: compile-time dimensioned site-level tensor types
- [tensor/transporter](tensor/transporter.html) — `Shifter[D,T]` and
  `Transporter[D,U,F]`: single-rank parallel transport with MPI halo exchange

### Backend: OpenCL

- [opencl/cldisp](opencl/cldisp.html) — `each` macro: compile-time
  expression analysis → OpenCL kernel generation (JIT)
- [opencl/clbase](opencl/clbase.html) — OpenCL platform/device/buffer
  management and math intrinsics

### Backend: SYCL

- [sycl/sycldisp](sycl/sycldisp.html) — `each` macro: compile-time
  expression analysis → pre-compiled C++ SYCL kernel dispatch
- [sycl/syclbase](sycl/syclbase.html) — SYCL queue/buffer management
- [sycl/syclwrap](sycl/syclwrap.html) — Low-level FFI bindings to
  `libreliq_sycl.so`

### Backend: OpenMP

- [openmp/ompdisp](openmp/ompdisp.html) — `each` macro: compile-time
  expression analysis → SIMD-vectorized OpenMP code generation
- [openmp/ompbase](openmp/ompbase.html) — OpenMP initialization and
  thread management
- [openmp/ompsimd](openmp/ompsimd.html) — SIMD-aware dispatch with
  outer/inner loop pattern

### SIMD Infrastructure

- [simd/simdtypes](simd/simdtypes.html) — `SimdVec[N,T]` and
  `SimdVecDyn[T]`: generic SIMD vector types abstracting SSE/AVX2/AVX-512
- [simd/simdlayout](simd/simdlayout.html) — `SimdLatticeLayout`:
  AoSoA memory layout computation for SIMD-vectorized lattice traversal

### Distributed Memory

- [globalarrays/gatypes](globalarrays/gatypes.html) — `GlobalArray[D,T]`:
  distributed array with ghost regions via Global Arrays
- [globalarrays/gabase](globalarrays/gabase.html) — Global Arrays
  initialization and finalization
- [globalarrays/gawrap](globalarrays/gawrap.html) — Low-level C FFI
  bindings to the Global Arrays library
- [globalarrays/gampi](globalarrays/gampi.html) — MPI initialization
  and communication primitives

### I/O

- [io/io](io/io.html) — Unified I/O for lattice data: LIME containers,
  SciDAC format with XML metadata, ILDG gauge configurations, QIO parallel I/O

### Utilities

- [utils/complex](utils/complex.html) — Complex number type predicates
- [utils/private](utils/private.html) — Internal utility functions

## Backend Selection

ReliQ supports three compute backends, selected at compile time:

| Backend | Flag | Requirements | Best For |
|---------|------|-------------|----------|
| OpenCL | *(default)* | OpenCL runtime (pocl, vendor SDK) | GPUs, FPGAs |
| SYCL | `BACKEND=sycl` | Intel oneAPI (icpx) or hipSYCL | Intel GPUs, modern CPUs |
| OpenMP | `BACKEND=openmp` | GCC/Clang with OpenMP | CPU-only, SIMD vectorization |

### Building

```
# OpenCL (default)
make tensorview

# OpenMP
make tensorview BACKEND=openmp

# SYCL (requires building the wrapper library first)
make sycl-lib
make tensorview BACKEND=sycl
```

### Running Tests

```
# Run all tests across all backends
make test

# Run tests for a specific backend
make test-core       # Backend-agnostic core tests
make test-opencl     # OpenCL backend tests
make test-openmp     # OpenMP backend tests
make test-sycl       # SYCL backend tests
```

## The `each` Macro

The `each` macro is the primary mechanism for expressing computations on
lattice fields. It analyzes the loop body at compile time and generates
optimized backend-specific code.

### Supported Patterns

```nim
# Vector/scalar copy
each vDst, vSrc, n:
  vDst[n] = vSrc[n]

# Scalar multiplication
each vDst, vSrc, n:
  vDst[n] = 3.0 * vSrc[n]

# Matrix multiplication
each vDst, vA, vB, n:
  vDst[n] = vA[n] * vB[n]

# Matrix-vector multiplication
each vDst, vA, vB, n:
  vDst[n] = vA[n] * vB[n]  # Mat * Vec

# Addition / subtraction
each vDst, vA, vB, n:
  vDst[n] = vA[n] + vB[n]

# Stencil neighbor access
each vDst, vSrc, stencil, n:
  let nbrIdx = stencil.neighbor(n, 0)
  vDst[n] = vSrc[nbrIdx]
```

### What Happens Under the Hood

1. **Parse**: The macro receives the loop body as an AST
2. **Analyze**: Expression trees are classified (copy, add, matmul, etc.)
3. **Detect stencils**: `let nbrIdx = stencil.neighbor(n, k)` patterns
   are detected and tracked
4. **Generate**: Backend-specific code is emitted:
   - **OpenCL**: An OpenCL C kernel string is generated, compiled via JIT,
     and dispatched with `clEnqueueNDRangeKernel`
   - **SYCL**: Calls to pre-compiled C++ template kernels in
     `libreliq_sycl.so` are emitted
   - **OpenMP**: SIMD-vectorized C code with `#pragma omp parallel for`
     and explicit SIMD intrinsic calls is generated

## AoSoA Memory Layout

ReliQ uses an Array of Structures of Arrays (AoSoA) memory layout for
optimal SIMD and GPU performance:

```
Traditional AoS:  [s0e0, s0e1, s1e0, s1e1, s2e0, s2e1, ...]
ReliQ AoSoA (VW=4): [s0e0, s1e0, s2e0, s3e0,  ← element 0, group 0
                      s0e1, s1e1, s2e1, s3e1,  ← element 1, group 0
                      s4e0, s5e0, s6e0, s7e0,  ← element 0, group 1
                      s4e1, s5e1, s6e1, s7e1]  ← element 1, group 1
```

**Index formula**: For site `s` with element index `i`:
```
group = s / VW
lane  = s mod VW
index = group * (elemsPerSite * VW) + i * VW + lane
```

Where `VW` (VectorWidth) is typically 8 for CPU (AVX2) or 32 for GPU.

## Stencil Operations

The unified stencil system provides neighbor access that works identically
across all backends:

```nim
# Create a lattice
let lat = newSimpleCubicLattice([8, 8, 8, 16])

# Create a nearest-neighbor stencil (2*D points for D dimensions)
let pattern = nearestNeighborStencil(lat)
let stencil = newLatticeStencil(pattern, lat)

# Direction API
let fwdX = stencil.fwd(0)  # Forward in x
let bwdT = stencil.bwd(3)  # Backward in t

# Neighbor access in each loops
each vDst, vSrc, stencil, n:
  let nbrFwd = stencil.neighbor(n, stencil.fwd(0).idx)
  let nbrBwd = stencil.neighbor(n, stencil.bwd(0).idx)
  # Discrete Laplacian: ψ(x+μ) + ψ(x-μ) - 2ψ(x)
  vDst[n] = vSrc[nbrFwd] + vSrc[nbrBwd] - 2.0 * vSrc[n]
```

## Distributed Transport (GlobalShifter)

For operations that span MPI boundaries at the `TensorField` level (before
down-casting to views), `GlobalShifter` performs distributed nearest-neighbour
shifts using Global Arrays ghost exchange.

```nim
import reliq

parallel:
  let lat = newSimpleCubicLattice([8, 8, 8, 16], [1, 1, 1, 4], [1, 1, 1, 1])
  var src  = lat.newTensorField([1, 1]): float64
  var dest = lat.newTensorField([1, 1]): float64

  # Fill src ...

  # Shift forward in t-dimension (crosses MPI boundaries)
  let shifter = newGlobalShifter(src, dim = 3, len = 1)
  shifter.apply(src, dest)   # dest[x] = src[x + e_t]

  # Create shifters for all dimensions at once
  let fwd = newGlobalShifters(src, len = 1)         # forward in all D dims
  let bwd = newGlobalBackwardShifters(src, len = 1)  # backward in all D dims

  # Discrete Laplacian: sum_mu (f[x+mu] + f[x-mu]) - 2D * f[x]
  var lap = lat.newTensorField([1, 1]): float64
  var tmp = lat.newTensorField([1, 1]): float64
  discreteLaplacian(src, lap, tmp)
```

### How it works

1. `updateAllGhosts()` calls `GA_Update_ghost_dir` in each lattice
   dimension to fill ghost cells with neighbor data from adjacent ranks
2. The source field is read via the *ghost pointer* (`accessGhosts`),
   which sees both local and ghost data
3. The destination is written via the *local pointer* (`accessLocal`),
   which starts at the centre of the inner padded block
4. For dimensions with only one MPI rank (not distributed), coordinates
   are wrapped locally since ghost exchange has no remote data to fetch

### Two transport layers

| Layer | Type | Where | Communication |
|-------|------|-------|---------------|
| `GlobalShifter` | `TensorField` | `globaltensor.nim` | GA ghost exchange (MPI) |
| `Shifter` / `Transporter` | `TensorFieldView` | `transporter.nim` | Halo buffers for device dispatch |

Use `GlobalShifter` when working directly with distributed tensor fields
(e.g. setup, I/O, measurement code).  Use `Shifter` / `Transporter` when
the data is already on-device inside an `each` loop.

## License

MIT License — Copyright (c) 2025 reliq-lft

See [LICENSE](https://github.com/reliq-lft/ReliQ/blob/main/LICENSE) for details.
