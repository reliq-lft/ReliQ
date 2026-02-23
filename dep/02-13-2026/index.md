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
  region management and distributed transport (``GlobalShifter``,
  ``discreteLaplacian``)
- **Layered tensor fields**: ``TensorField`` (distributed GA) →
  ``LocalTensorField`` (direct pointer into rank-local GA memory with siteOffsets lookup) → ``TensorFieldView``
  (device-side AoSoA views)
- **Unified stencil operations**: A single ``LatticeStencil[D]`` type that
  works identically across all backends, handling periodic boundary
  conditions and ghost regions automatically
- **Macro-based dispatch**: The ``each`` macro analyzes loop bodies at
  compile time and generates optimized backend-specific code

## Quick Start

```nim
import reliq

parallel:
  # Create a 4D lattice (8×8×8×16) with MPI decomposition
  let lat = newSimpleCubicLattice([8, 8, 8, 16])

  block:
    # Create tensor fields on the lattice
    var fieldA = lat.newTensorField([3, 3]): float64
    var fieldB = lat.newTensorField([3, 3]): float64
    var fieldC = lat.newTensorField([3, 3]): float64

    # Get local host-memory views
    var localA = fieldA.newLocalTensorField()
    var localB = fieldB.newLocalTensorField()
    var localC = fieldC.newLocalTensorField()

    # Host-side initialization with "for all" loop
    for n in all 0..<localA.numSites():
      var siteA = localA[n]
      siteA[0, 0] = 1.0  # Set matrix element (0,0)

    # Create device views for backend dispatch
    var vA = localA.newTensorFieldView(iokRead)
    var vB = localB.newTensorFieldView(iokRead)
    var vC = localC.newTensorFieldView(iokWrite)

    # Device-side computation with "each" loop
    for n in each 0..<vA.numSites():
      vC[n] = vA[n] + vB[n]      # Matrix addition
      vC[n] = vA[n] * vB[n]      # Matrix multiplication
      vC[n] = 3.0 * vA[n]        # Scalar multiplication
```

## Architecture

ReliQ is organized into several layers.  Each pillar has its own detailed
guide page:

- [Lattice Infrastructure](lattice_guide.html) — Geometry, coordinates, stencils
- [Tensor Fields](tensor_guide.html) — TensorField, LocalTensorField, TensorFieldView, transport
- [Compute Backends](backends_guide.html) — OpenCL, SYCL, OpenMP dispatch
- [SIMD Infrastructure](simd_guide.html) — SimdVec types, AoSoA layout computation
- [Distributed Memory](distributed_guide.html) — Global Arrays, MPI, ghost exchange
- [I/O Module](io_guide.html) — LIME, SciDAC, ILDG file formats

```
┌──────────────────────────────────────────────────────────┐
│                       User Code                          │
│           import reliq; for n in each ...: ...           │
├──────────────────────────────────────────────────────────┤
│                    Tensor Layer                          │
│   TensorField ─► LocalTensorField ─► TensorFieldView    │
│   (GA/MPI)        (direct GA ptr)      (device buffers)   │
├──────────────────────────────────────────────────────────┤
│        GlobalShifter · LatticeStencil · Transporter      │
│        discreteLaplacian · applyStencilShift             │
├──────────────────────────────────────────────────────────┤
│                 Backend Dispatch                         │
│     OpenCL (JIT)  │  SYCL (pre-compiled)  │  OpenMP     │
│      (cldisp)     │    (sycldisp)         │  (ompdisp)  │
├──────────────────────────────────────────────────────────┤
│              Memory & Communication                      │
│   Global Arrays · MPI · AoSoA Layout · SIMD Intrinsics   │
└──────────────────────────────────────────────────────────┘
```

### Data Flow

1. **``TensorField[D,R,L,T]``** — Distributed tensor field stored as a
   Global Array with ghost regions.  Created with
   ``lat.newTensorField(shape): T``.

2. **``LocalTensorField[D,R,L,T]``** — A direct pointer into the
   rank-local GA memory (obtained via ``NGA_Access``) with a precomputed
   ``siteOffsets: seq[int]`` lookup table that maps lexicographic site →
   flat offset in padded GA memory.  Elements are accessed as
   ``data[siteOffsets[site] + e]``.  Created with
   ``field.newLocalTensorField()``.  Writes go directly to the GA —
   no manual flush is needed.

3. **``TensorFieldView[L,T]``** — Device-side view with AoSoA layout
   for SIMD or GPU buffers.  Created with
   ``local.newTensorFieldView(iokRead)`` (or ``iokWrite``/``iokReadWrite``).
   This is the type the ``each`` macro operates on.

## Module Reference

### Core Modules

- [reliq](reliq.html) — Top-level entry point; re-exports all public modules
- [lattice](lattice.html) — Lattice types, stencils, and concepts
- [parallel](parallel.html) — Backend-agnostic parallel dispatch

### Lattice

- [lattice/latticeconcept](lattice/latticeconcept.html) — The ``Lattice[D]``
  concept that all lattice types must satisfy
- [lattice/simplecubiclattice](lattice/simplecubiclattice.html) —
  ``SimpleCubicLattice[D]`` with MPI decomposition and ghost grids
- [lattice/stencil](lattice/stencil.html) — Unified stencil operations:
  ``LatticeStencil[D]``, ``StencilPattern[D]``, neighbor access, shift API
- [lattice/indexing](lattice/indexing.html) — Coordinate ↔ flat index
  conversion utilities

### Tensor Fields

- [tensor/tensor](tensor/tensor.html) — Tensor module aggregation;
  re-exports all tensor submodules
- [tensor/globaltensor](tensor/globaltensor.html) — ``TensorField[D,R,L,T]``:
  distributed tensor field with ghost-padded GA memory layout,
  ``GlobalShifter`` for distributed transport, ``discreteLaplacian``,
  coordinate utilities, and ``LatticeStencil`` integration
- [tensor/localtensor](tensor/localtensor.html) —
  ``LocalTensorField[D,R,L,T]``: direct GA pointer with ``siteOffsets``
  lookup for padded-stride navigation, site proxies for ``all`` loops, arithmetic operators
- [tensor/tensorview](tensor/tensorview.html) — ``TensorFieldView[L,T]``:
  device-side views for the ``each`` macro; handles AoSoA layout
  transformation across OpenCL, SYCL, and OpenMP backends
- [tensor/sitetensor](tensor/sitetensor.html) — ``Vec[N,T]`` and
  ``Mat[N,M,T]``: compile-time dimensioned site-level tensor types
- [tensor/transporter](tensor/transporter.html) — ``Shifter[D,T]`` and
  ``Transporter[D,U,F]``: single-rank parallel transport with halo exchange

### Backend: OpenCL

- [opencl/cldisp](opencl/cldisp.html) — ``each`` macro: compile-time
  expression analysis → OpenCL kernel generation (JIT)
- [opencl/clbase](opencl/clbase.html) — OpenCL platform/device/buffer
  management and math intrinsics

### Backend: SYCL

- [sycl/sycldisp](sycl/sycldisp.html) — ``each`` macro: compile-time
  expression analysis → pre-compiled C++ SYCL kernel dispatch
- [sycl/syclbase](sycl/syclbase.html) — SYCL queue/buffer management
- [sycl/syclwrap](sycl/syclwrap.html) — Low-level FFI bindings to
  ``libreliq_sycl.so``

### Backend: OpenMP

- [openmp/ompdisp](openmp/ompdisp.html) — ``each`` macro: compile-time
  expression analysis → SIMD-vectorized OpenMP code generation
- [openmp/ompbase](openmp/ompbase.html) — OpenMP initialization and
  thread management
- [openmp/ompsimd](openmp/ompsimd.html) — SIMD-aware dispatch with
  outer/inner loop pattern

### SIMD Infrastructure

- [simd/simdtypes](simd/simdtypes.html) — ``SimdVec[N,T]`` and
  ``SimdVecDyn[T]``: generic SIMD vector types abstracting SSE/AVX2/AVX-512
- [simd/simdlayout](simd/simdlayout.html) — ``SimdLatticeLayout``:
  AoSoA memory layout computation for SIMD-vectorized lattice traversal

### Distributed Memory

- [globalarrays/gatypes](globalarrays/gatypes.html) — ``GlobalArray[D,T]``:
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
| SYCL | ``BACKEND=sycl`` | Intel oneAPI (icpx) or hipSYCL | Intel GPUs, modern CPUs |
| OpenMP | ``BACKEND=openmp`` | GCC/Clang with OpenMP | CPU-only, SIMD vectorization |

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

## The ``each`` Macro

The ``each`` macro is the primary mechanism for expressing computations on
lattice fields. It analyzes the loop body at compile time and generates
optimized backend-specific code.

### Creating Tensor Fields and Views

```nim
import reliq

parallel:
  let lat = newSimpleCubicLattice([8, 8, 8, 16])

  block:
    # Create distributed tensor fields
    var fieldA = lat.newTensorField([3, 3]): float64    # 3x3 matrix field
    var fieldB = lat.newTensorField([3, 3]): float64
    var fieldC = lat.newTensorField([3, 3]): float64

    # Get local host-memory views
    var localA = fieldA.newLocalTensorField()
    var localB = fieldB.newLocalTensorField()
    var localC = fieldC.newLocalTensorField()

    # Create device-side views
    var vA = localA.newTensorFieldView(iokRead)      # read-only
    var vB = localB.newTensorFieldView(iokRead)      # read-only
    var vC = localC.newTensorFieldView(iokWrite)     # write-only
```

### Supported Operations

```nim
    # Vector/matrix copy
    for n in each 0..<vC.numSites():
      vC[n] = vA[n]

    # Scalar multiplication
    for n in each 0..<vC.numSites():
      vC[n] = 3.0 * vA[n]

    # Matrix multiplication
    for n in each 0..<vC.numSites():
      vC[n] = vA[n] * vB[n]

    # Addition / subtraction
    for n in each 0..<vC.numSites():
      vC[n] = vA[n] + vB[n]

    # Combined expressions
    for n in each 0..<vC.numSites():
      vC[n] = vA[n] * vB[n] + vC[n]
```

### Stencil Neighbor Access

```nim
    let stencil = newLatticeStencil(lat)

    for n in each 0..<vC.numSites():
      let fwd = stencil.fwd(n, 0)     # Forward x-neighbor
      let bwd = stencil.bwd(n, 0)     # Backward x-neighbor
      vC[n] = vA[fwd] + vA[bwd] - 2.0 * vA[n]
```

### What Happens Under the Hood

1. **Parse**: The macro receives the loop body as an AST
2. **Analyze**: Expression trees are classified (copy, add, matmul, etc.)
3. **Detect stencils**: Stencil neighbor patterns are detected and tracked
4. **Generate**: Backend-specific code is emitted:
   - **OpenCL**: An OpenCL C kernel string is JIT-compiled and dispatched
     with ``clEnqueueNDRangeKernel``
   - **SYCL**: Calls to pre-compiled C++ template kernels in
     ``libreliq_sycl.so`` are emitted
   - **OpenMP**: SIMD-vectorized C code with ``#pragma omp parallel for``
     and explicit SIMD intrinsic calls is generated

## The ``all`` Loop

The ``all`` loop operates on ``LocalTensorField`` objects for host-side
site-level operations using ``LocalSiteProxy``:

```nim
    # Initialize via site proxy
    for n in all 0..<localA.numSites():
      var site = localA[n]
      site[0, 0] = 1.0   # Set matrix element (row, col)

    # Arithmetic operations
    for n in all 0..<localC.numSites():
      localC[n] = localA[n] + localB[n]   # add
      localC[n] = localA[n] * localB[n]   # multiply
      localC[n] = 2.5 * localA[n]                 # scale

```

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

**Index formula**: For site ``s`` with element index ``i``:
```
group = s / VW
lane  = s mod VW
index = group * (elemsPerSite * VW) + i * VW + lane
```

Where ``VW`` (VectorWidth) is typically 8 for CPU (AVX-512) or configurable
via ``-d:VectorWidth=N``.

## Distributed Transport (``GlobalShifter``)

For operations that span MPI boundaries at the ``TensorField`` level (before
down-casting to views), ``GlobalShifter`` performs distributed nearest-neighbour
shifts using Global Arrays ghost exchange.

```nim
parallel:
  let lat = newSimpleCubicLattice([8, 8, 8, 16], [1, 1, 1, 4], [1, 1, 1, 1])

  block:
    var src  = lat.newTensorField([1, 1]): float64
    var dest = lat.newTensorField([1, 1]): float64

    # Fill src ...

    # Shift forward in t-dimension (crosses MPI boundaries)
    let shifter = newGlobalShifter(src, dim=3, len=1)
    shifter.apply(src, dest)   # dest[x] = src[x + e_t]

    # Create shifters for all dimensions at once
    let fwd = newGlobalShifters(src, len=1)
    let bwd = newGlobalBackwardShifters(src, len=1)

    # Discrete Laplacian: sum_mu (f[x+mu] + f[x-mu]) - 2D * f[x]
    var lap = lat.newTensorField([1, 1]): float64
    var scratch = lat.newTensorField([1, 1]): float64
    discreteLaplacian(src, lap, scratch)
```

### Two Transport Layers

| Layer | Type | Communication |
|-------|------|---------------|
| ``GlobalShifter`` | ``TensorField`` | GA ghost exchange (MPI) |
| ``Shifter`` / ``Transporter`` | ``TensorFieldView`` | Device-side halo buffers |

Use ``GlobalShifter`` when working directly with distributed tensor fields
(e.g. setup, I/O, measurement code).  Use ``Shifter`` / ``Transporter`` when
the data is already on-device inside an ``each`` loop.

## I/O

ReliQ supports standard lattice QCD file formats through the ``io`` module:

- **LIME containers** — Low-level record reading/writing
- **SciDAC/QIO** — XML metadata, checksum validation, precision handling
- **ILDG** — Standard gauge configuration format

```nim
parallel:
  let lat = newSimpleCubicLattice([8, 8, 8, 16])

  block:
    # Read an ILDG gauge configuration
    var g0 = lat.newTensorField([3, 3]): Complex64
    var g1 = lat.newTensorField([3, 3]): Complex64
    var g2 = lat.newTensorField([3, 3]): Complex64
    var g3 = lat.newTensorField([3, 3]): Complex64
    var gaugeField = [g0, g1, g2, g3]
    readGaugeField(gaugeField, "config.ildg")

    # Write a tensor field to LIME/SciDAC format
    var field = lat.newTensorField([3, 3]): float64
    writeTensorField(field, "output.lime")
```

## License

MIT License — Copyright (c) 2025 reliq-lft

See [LICENSE](https://github.com/reliq-lft/ReliQ/blob/main/LICENSE) for details.