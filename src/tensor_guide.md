# Tensor Fields

The tensor layer is the core data abstraction in ReliQ.  It provides three
levels of tensor field representation — distributed, local, and device-side —
connected by a well-defined data flow.

## Data Flow

```
TensorField[D,R,L,T]           ← Distributed (GA + MPI)
    │ newLocalTensorField()
    ▼
LocalTensorField[D,R,L,T]      ← Direct GA pointer + siteOffsets
    │ newTensorFieldView()
    ▼
TensorFieldView[L,T]           ← Device-side AoSoA buffers
    │ each macro
    ▼
Backend dispatch (OpenCL / SYCL / OpenMP)
```

## TensorField (Distributed)

``TensorField[D, R, L, T]`` wraps a Global Array with automatic ghost regions
and MPI decomposition.  Here ``D`` is the lattice dimension, ``R`` is the
tensor rank (number of index dimensions), ``L`` is the lattice type, and ``T``
is the element type (``float32``, ``float64``, ``Complex64``, etc.).

### Creation

```nim
import reliq

parallel:
  let lat = newSimpleCubicLattice([8, 8, 8, 16], [1, 1, 1, 4], [1, 1, 1, 1])

  block:
    # Scalar field (rank 0, but stored as [1,1] for GA compatibility)
    var scalar = lat.newTensorField([1, 1]): float64

    # 3×3 matrix field
    var gauge = lat.newTensorField([3, 3]): float64

    # Complex 3×3 matrix field (e.g., SU(3) gauge links)
    var su3 = lat.newTensorField([3, 3]): Complex64

    # 4-vector field
    var vec = lat.newTensorField([4, 1]): float64
```

The ``newTensorField`` macro accepts a shape array and element type.  The
shape ``[3, 3]`` means a 3×3 matrix at each lattice site; ``[4, 1]`` means a
length-4 vector.

### GA Memory Layout

Internally, a ``TensorField`` creates a ``(D + R + 1)``-dimensional Global
Array:

- **Dimensions 0..D-1** — Lattice spatial dimensions
- **Dimensions D..D+R-1** — Tensor index dimensions (shape)
- **Dimension D+R** — Complex/real dimension (size 2 for complex, 1 for real)

Ghost regions of width 1 are added to **all** dimensions (a requirement of
Global Arrays 5.8.2 — ``GA_Update_ghost_dir`` crashes if any dimension has
ghost width 0).  This means the inner (tensor+complex) block has padded
strides:

```
Inner block size (real elements) = shape[0] * shape[1] * ... * complexFactor
Padded block size (with ghosts) = (shape[0]+2) * (shape[1]+2) * ... * (complexFactor+2)
```

The ``innerBlockSize`` and ``innerPaddedBlockSize`` procs compute these values.

### Ghost Exchange

```nim
# Update ghosts in one dimension/direction
tensor.updateGhosts(dim=3, direction=1)

# Update all ghost regions
tensor.updateAllGhosts()
```

### Direct Local Access

For advanced use, you can access the raw GA pointer:

```nim
let rawPtr = tensor.accessLocal()
# ... use rawPtr with innerPaddedBlockSize strides ...
tensor.releaseLocal()
```

However, the recommended approach is to use ``LocalTensorField`` which
provides a ``siteOffsets`` lookup table to navigate the padded memory directly.

## LocalTensorField (Direct GA Pointer)

``LocalTensorField[D, R, L, T]`` holds a direct pointer into the rank-local
GA memory (obtained via ``NGA_Access``) together with a precomputed
``siteOffsets: seq[int]`` lookup table that maps each lexicographic site
index to its flat offset inside the padded GA memory.  No contiguous buffer
is allocated and no copy is performed — reads and writes go directly to
the Global Array.

```
data[siteOffsets[site] + element]
```

### Creation

```nim
# Create from a distributed TensorField
var local = field.newLocalTensorField()

# Access data — writes go directly to the GA
for n in all 0..<local.numSites():
  var site = local[n]
  site[0, 0] = 1.0

# No manual flush needed — data is already in the GA
```

Since ``LocalTensorField.data`` **is** the GA pointer, all modifications
are visible to the distributed array immediately.  There is no
``releaseLocalTensorField()`` step.

### Properties

| Proc | Returns | Description |
|------|---------|-------------|
| ``numSites()`` | ``int`` | Number of local lattice sites |
| ``numGlobalSites()`` | ``int`` | Total sites across all ranks |
| ``numElements()`` | ``int`` | Total elements in local buffer |
| ``tensorElementsPerSite()`` | ``int`` | Elements per lattice site |

### Element Access

```nim
# Direct element access (by flat index)
local.setElement(idx, 3.14)   # Write element at flat index
let x = local.getElement(idx) # Read element at flat index

# Site proxy access (preferred for the "all" loop)
var site = local[n]
site[i, j] = 1.0        # Matrix element (row, col)
site[i] = 2.0           # Vector element (index)
let v = site[i, j]      # Read matrix element
```

### The ``all`` Loop

The ``all`` loop provides parallelized iteration over local sites:

```nim
# Element initialization
for n in all 0..<local.numSites():
  var site = local[n]
  for i in 0..<3:
    for j in 0..<3:
      site[i, j] = if i == j: 1.0 else: 0.0

# Arithmetic via LocalSiteProxy
for n in all 0..<localC.numSites():
  localC[n] = localA[n] + localB[n]
  localC[n] = localA[n] * localB[n]
  localC[n] = 2.5 * localA[n]
```

Supported proxy operations:

| Operation | Syntax |
|-----------|--------|
| Addition | ``localC[n] = a[n] + b[n]`` |
| Subtraction | ``localC[n] = a[n] - b[n]`` |
| Multiplication | ``localC[n] = a[n] * b[n]`` |
| Scalar multiply | ``localC[n] = 3.0 * a[n]`` |
| Scalar add | ``localC[n] = a[n] + 3.0`` |
| Chaining | ``localC[n] = a[n] * b[n] + c[n]`` |

## TensorFieldView (Device-Side)

``TensorFieldView[L, T]`` wraps a ``LocalTensorField`` into device-side
buffers with AoSoA memory layout.  This is the type that the ``each`` macro
operates on.

### Creation

```nim
# Read-only view (data uploaded to device)
var vA = localA.newTensorFieldView(iokRead)

# Write-only view (results downloaded from device after each loop)
var vC = localC.newTensorFieldView(iokWrite)

# Read-write view
var vRW = localRW.newTensorFieldView(iokReadWrite)
```

The ``IOKind`` enum controls data transfer:

| Kind | Upload | Download | Use case |
|------|--------|----------|----------|
| ``iokRead`` | Yes | No | Input fields |
| ``iokWrite`` | No | Yes | Output fields |
| ``iokReadWrite`` | Yes | Yes | In-place updates |

### The ``each`` Macro

The ``each`` macro is the primary computation dispatch mechanism:

```nim
for n in each 0..<vC.numSites():
  vC[n] = vA[n] + vB[n]
```

This single line of code works across all three backends.  The macro:

1. **Parses** the loop body AST at compile time
2. **Classifies** each expression (copy, add, matmul, scalar-mul, etc.)
3. **Detects** stencil neighbor patterns
4. **Generates** backend-specific code:
   - **OpenCL**: JIT-compiled OpenCL C kernel string
   - **SYCL**: Calls to pre-compiled C++ kernel templates
   - **OpenMP**: SIMD-vectorized C code with intrinsics

### Supported ``each`` Operations

```nim
# Copy
for n in each 0..<vC.numSites():
  vC[n] = vA[n]

# Addition / subtraction
for n in each 0..<vC.numSites():
  vC[n] = vA[n] + vB[n]
  vC[n] = vA[n] - vB[n]

# Matrix multiplication (R=2 fields)
for n in each 0..<vC.numSites():
  vC[n] = vA[n] * vB[n]

# Matrix-vector multiplication
for n in each 0..<vC.numSites():
  vOut[n] = vMat[n] * vVec[n]

# Scalar multiplication / addition
for n in each 0..<vC.numSites():
  vC[n] = 3.0 * vA[n]
  vC[n] = vA[n] + 1.0

# Chained expressions
for n in each 0..<vC.numSites():
  vC[n] = vA[n] * vB[n] + vC[n]

# Stencil neighbor access
for n in each 0..<vC.numSites():
  let f = stencil.fwd(n, 0)
  let b = stencil.bwd(n, 0)
  vC[n] = vA[f] + vA[b]

# Element-level write
for n in each 0..<vC.numSites():
  vC[n][0, 0] = 1.0
```

### AoSoA Memory Layout

Device-side data uses Array of Structures of Arrays layout for optimal
SIMD/GPU memory access:

```
AoS:   [s0e0, s0e1, s1e0, s1e1, s2e0, s2e1, ...]

AoSoA (VW=4):
  Group 0: [s0e0, s1e0, s2e0, s3e0,   ← element 0
            s0e1, s1e1, s2e1, s3e1]   ← element 1
  Group 1: [s4e0, s5e0, s6e0, s7e0,
            s4e1, s5e1, s6e1, s7e1]
```

**Index formula:**
```
group = site / VectorWidth
lane  = site mod VectorWidth
index = group * (elemsPerSite * VectorWidth) + element * VectorWidth + lane
```

``VectorWidth`` defaults to 8 (AVX-512) and can be set with ``-d:VectorWidth=N``.

## Site Tensor Types

``Vec[N, T]`` and ``Mat[N, M, T]`` represent tensors at a single lattice site
with compile-time dimensions.  These are the types that appear inside ``each``
loop bodies and are translated to backend-specific code by the macro.

### Type Aliases

| Alias | Full type |
|-------|-----------|
| ``Vec2f`` | ``Vec[2, float32]`` |
| ``Vec3d`` | ``Vec[3, float64]`` |
| ``Mat3d`` | ``Mat[3, 3, float64]`` |
| ``Mat4f`` | ``Mat[4, 4, float32]`` |

### Operations

```nim
# Vec operations
let v = Vec3d(data: [1.0, 2.0, 3.0])
let w = Vec3d(data: [4.0, 5.0, 6.0])
let sum = v + w
let scaled = 2.0 * v
let d = dot(v, w)

# Mat operations
let m = identity[3, float64]()
let mt = m.transpose()
let tr = m.trace()
let prod = m * m
let mv = m * v
```

## Transport Infrastructure

### Shifter (Single-Rank)

``Shifter[D, T]`` performs field shifts within device-side views, using halo
buffers for boundary communication:

```nim
let localGeom = [8, 8, 8, 4]  # Local lattice after MPI decomposition
var shifter = newShifter[4, float64](localGeom, dim=0, len=1)

# Create shifters for all dimensions
var fwd = newShifters[4, float64](localGeom, len=1)
var bwd = newBackwardShifters[4, float64](localGeom, len=1)
```

### Transporter (Gauge Link Multiplication)

``Transporter[D, U, F]`` performs parallel transport with gauge link
multiplication (covariant shifts):

```nim
var transporter = newTransporter[4, GaugeType, FermionType](
  localGeom, gaugeField, dim=0, len=1
)
```

### GlobalShifter (MPI-Level)

``GlobalShifter[D, R, L, T]`` performs distributed shifts using GA ghost
exchange across MPI boundaries:

```nim
parallel:
  let lat = newSimpleCubicLattice([8, 8, 8, 16], [1, 1, 1, 4], [1, 1, 1, 1])

  block:
    var src  = lat.newTensorField([1, 1]): float64
    var dest = lat.newTensorField([1, 1]): float64

    # Forward shift in t-dimension
    let shifter = newGlobalShifter(src, dim=3, len=1)
    shifter.apply(src, dest)   # dest[x] = src[x + e_t]

    # Discrete Laplacian
    var lap = lat.newTensorField([1, 1]): float64
    var scratch = lat.newTensorField([1, 1]): float64
    discreteLaplacian(src, lap, scratch)
```

### Two Transport Layers

| Layer | Operates On | Communication |
|-------|------------|---------------|
| ``GlobalShifter`` | ``TensorField`` | GA ghost exchange (MPI) |
| ``Shifter`` / ``Transporter`` | ``TensorFieldView`` | Device-side halo buffers |

Use ``GlobalShifter`` for setup, I/O, and measurement code.  Use
``Shifter``/``Transporter`` for on-device computation inside ``each`` loops.

## Module Reference

| Module | Description |
|--------|-------------|
| [tensor/globaltensor](tensor/globaltensor.html) | Distributed tensor fields and GlobalShifter |
| [tensor/localtensor](tensor/localtensor.html) | Direct GA pointer with ``siteOffsets`` lookup |
| [tensor/tensorview](tensor/tensorview.html) | Device-side views and ``each`` macro dispatch |
| [tensor/sitetensor](tensor/sitetensor.html) | ``Vec[N,T]`` and ``Mat[N,M,T]`` site types |
| [tensor/transporter](tensor/transporter.html) | Single-rank Shifter and Transporter |
