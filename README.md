# ReliQ — Lattice Field Theory Framework

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/reliq-lft/ReliQ/blob/main/LICENSE)
[![Nim](https://img.shields.io/badge/Nim-2.2.4-orange.svg)](https://nim-lang.org/)
[![GA](https://img.shields.io/badge/GlobalArrays-5.8-green.svg)](https://globalarrays.github.io/)

> *"We all make choices. But in the end our choices make us."* — Andrew Ryan (BioShock)

![](https://github.com/reliq-lft/ReliQ/blob/main/reliq/reliq.png)

**ReliQ** is an experimental lattice field theory framework written in [Nim](https://nim-lang.org/), designed for user-friendliness, performance, reliability, and portability across heterogeneous architectures. Distributed memory is handled through a [partitioned global address space](https://en.wikipedia.org/wiki/Partitioned_global_address_space) model backed by [Global Arrays (GA)](https://globalarrays.github.io/), while device-level parallelism dispatches across three backends — OpenCL, SYCL, and OpenMP — through a single user-facing API.

> **Early Development** — ReliQ is under active development and is not yet production-ready. Contributions are welcome; contact us at [reliq-lft@proton.me](mailto:reliq-lft@proton.me) or follow us on our [organization page](https://github.com/reliq-lft).

---

## Architecture

ReliQ is organized into layered abstractions, each narrowing scope from global distributed data to device-specific kernel execution:

```
┌──────────────────────────────────────────────────────────┐
│                       User Code                          │
│            import reliq; each v, n: v[n] = ...           │
├──────────────────────────────────────────────────────────┤
│                    Tensor Layer                          │
│   TensorField ─► LocalTensorField ─► TensorFieldView    │
│   (GA/MPI)        (host buffer)       (device buffers)   │
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

1. **`TensorField[D,R,L,T]`** — A distributed tensor field stored as a Global Array with ghost (halo) regions for boundary communication across MPI ranks.
2. **`LocalTensorField[D,R,L,T]`** — A contiguous host-memory copy of the rank-local partition. Created via `newLocalTensorField()`; data flows back to the GA on `releaseLocalTensorField()`.
3. **`TensorFieldView[L,T]`** — A device-side view optimized for the active backend (AoSoA layout for SIMD, GPU buffers for OpenCL/SYCL). This is the type the `each` macro operates on.

### Three Compute Backends

| Backend | Flag | Best For | Mechanism |
|---------|------|----------|-----------|
| **OpenCL** | *(default)* | GPUs, FPGAs | JIT kernel compilation at runtime |
| **SYCL** | `BACKEND=sycl` | Intel GPUs, oneAPI | Pre-compiled C++ template kernels |
| **OpenMP** | `BACKEND=openmp` | CPU-only | SIMD-vectorized loops (SSE/AVX2/AVX-512) |

All three backends share the same user-facing API — the `each` macro analyzes loop bodies at compile time and generates the appropriate backend code.

---

## Quick Start

### Prerequisites

- Python 3.10+ (for the bootstrap/configure scripts and launcher)
- A C/C++ compiler (GCC, Clang, or icpx)
- MPI implementation (OpenMPI, MPICH, etc.)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/reliq-lft/ReliQ.git
cd ReliQ

# 2. Create a build directory
mkdir -p /path/to/build && cd /path/to/build

# 3. Bootstrap dependencies (installs Nim, Global Arrays, Kokkos via Spack)
/path/to/ReliQ/bootstrap

# 4. Configure
/path/to/ReliQ/configure
```

The bootstrap script performs a local [Spack](https://spack.io/) installation and uses it to install Nim 2.2.4, Global Arrays 5.8.2, and Kokkos 4.6.01. All dependencies are installed under `<build>/external/`.

### Building and Running

```bash
# Compile a module
make tensor

# Run tests with the parallel launcher
./reliq -e tensor -n 1       # 1 MPI rank
./reliq -e tensor -n 4       # 4 MPI ranks

# Run the full test suite (core + all backends)
make test
```

---

## The `each` Macro

The `each` macro is the primary mechanism for expressing computations on lattice fields. It works on `TensorFieldView` objects and generates optimized backend-specific code at compile time.

```nim
import reliq

parallel:
  let lat = newSimpleCubicLattice([8, 8, 8, 16])

  block:
    var fieldA = lat.newTensorField([3, 3]): float64
    var fieldB = lat.newTensorField([3, 3]): float64
    var fieldC = lat.newTensorField([3, 3]): float64

    var localA = fieldA.newLocalTensorField()
    var localB = fieldB.newLocalTensorField()
    var localC = fieldC.newLocalTensorField()

    # Create device views
    var vA = localA.newTensorFieldView(iokRead)
    var vB = localB.newTensorFieldView(iokRead)
    var vC = localC.newTensorFieldView(iokWrite)

    # Dispatch computation across all backend devices
    for n in each 0..<vA.numSites():
      vC[n] = vA[n] + vB[n]          # Element-wise addition
      vC[n] = vA[n] * vB[n]          # Matrix multiplication
      vC[n] = 3.0 * vA[n]            # Scalar multiplication
```

### Stencil Operations in `each` Loops

```nim
let stencil = newLatticeStencil(nearestNeighborStencil[4](), lat)

for n in each 0..<vDst.numSites():
  let fwd = stencil.fwd(n, 0)     # Forward x-neighbor
  let bwd = stencil.bwd(n, 0)     # Backward x-neighbor
  vDst[n] = vSrc[fwd] + vSrc[bwd] - 2.0 * vSrc[n]
```

---

## The `all` Loop (Host-Side)

The `all` loop operates on `LocalTensorField` objects for host-side site-level operations using `LocalSiteProxy`:

```nim
var localA = fieldA.newLocalTensorField()
var localB = fieldB.newLocalTensorField()
var localC = fieldC.newLocalTensorField()

for n in all 0..<localC.numSites():
  localC[n] = localA.getSite(n) + localB.getSite(n)
  localC[n] = localA.getSite(n) * localB.getSite(n)
  localC[n] = 2.5 * localA.getSite(n)

# Write changes back to the distributed Global Array
localC.releaseLocalTensorField()
```

---

## Distributed Transport

### GlobalShifter — MPI-Level Transport

For operations that cross MPI partition boundaries at the `TensorField` level:

```nim
parallel:
  let lat = newSimpleCubicLattice([8, 8, 8, 16], [1, 1, 1, 4], [1, 1, 1, 1])

  block:
    var src  = lat.newTensorField([1, 1]): float64
    var dest = lat.newTensorField([1, 1]): float64

    # Shift forward in the t-dimension (crosses MPI boundaries)
    let shifter = newGlobalShifter(src, dim=3, len=1)
    shifter.apply(src, dest)   # dest[x] = src[x + e_t]

    # Discrete Laplacian: sum_mu (f[x+mu] + f[x-mu]) - 2D * f[x]
    var lap = lat.newTensorField([1, 1]): float64
    var scratch = lat.newTensorField([1, 1]): float64
    discreteLaplacian(src, lap, scratch)
```

### Two Transport Layers

| Layer | Type | Communication |
|-------|------|---------------|
| `GlobalShifter` | `TensorField` | GA ghost exchange (MPI) |
| `Shifter` / `Transporter` | `TensorFieldView` | Device-side halo buffers |

Use `GlobalShifter` when working with distributed tensor fields directly (setup, I/O, measurements). Use `Shifter` when data is already on-device inside `each` loops.

---

## I/O

ReliQ supports standard lattice QCD file formats:

```nim
parallel:
  let lat = newSimpleCubicLattice([8, 8, 8, 16])

  block:
    # Read an ILDG gauge configuration
    var gaugeField: array[4, TensorField[4, 2, typeof(lat), Complex64]]
    for mu in 0..<4:
      gaugeField[mu] = lat.newTensorField([3, 3]): Complex64
    readGaugeField(gaugeField, "config.ildg")

    # Write a tensor field
    var field = lat.newTensorField([3, 3]): float64
    writeTensorField(field, "output.lime")
```

**Supported formats**: LIME containers, SciDAC/QIO with XML metadata and checksums, ILDG gauge configurations.

---

## Test Suite

ReliQ has a comprehensive test suite organized into four categories:

```bash
make test-core     # Backend-agnostic (lattice, stencil, tensor, transport, I/O)
make test-opencl   # OpenCL backend
make test-openmp   # OpenMP backend with SIMD
make test-sycl     # SYCL backend
make test          # All of the above
```

Each test module runs at both 1 and 4 MPI ranks. The current suite contains **1,660 tests** across all backends with zero failures.

| Suite | Tests |
|-------|-------|
| Core (backend-agnostic) | 875 |
| OpenCL | 245 |
| OpenMP | 295 |
| SYCL | 245 |
| **Total** | **1,660** |

---

## Documentation

API documentation is available at [reliq-lft.github.io/ReliQ](https://reliq-lft.github.io/ReliQ/). Generate documentation locally from the build directory:

```bash
./document
```

---

## Module Overview

| Module | Description |
|--------|-------------|
| `lattice` | `SimpleCubicLattice[D]`, `LatticeStencil[D]`, indexing utilities |
| `tensor` | `TensorField`, `LocalTensorField`, `TensorFieldView`, `GlobalShifter` |
| `parallel` | Backend-agnostic parallel dispatch (`parallel:` template, `each` macro) |
| `io` | LIME/QIO/SciDAC/ILDG file I/O with checksum validation |
| `globalarrays` | Global Arrays FFI bindings, distributed array types, MPI wrappers |
| `opencl` | OpenCL JIT kernel generation and dispatch |
| `sycl` | SYCL pre-compiled kernel dispatch via `libreliq_sycl.so` |
| `openmp` | OpenMP SIMD-vectorized CPU dispatch |
| `simd` | `SimdVec[N,T]`, `SimdLatticeLayout`, AoSoA memory layout |
| `utils` | Complex number predicates, command-line parsing |

---

## Parallel Launcher

The `reliq` launcher script wraps `mpirun` and auto-configures threading:

```bash
./reliq -e <program> -n <ntasks> [-t <nthreads>]
```

| Flag | Description |
|------|-------------|
| `-e` / `--executable` | Program name (from `bin/`) |
| `-n` / `--ntasks` | Number of MPI ranks |
| `-t` / `--nthreads` | Threads per rank (auto-detected if omitted) |

---

## License

MIT License — Copyright (c) 2025 reliq-lft

See [LICENSE](LICENSE) for details.