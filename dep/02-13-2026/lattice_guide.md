# Lattice Infrastructure

The lattice layer defines the spatial geometry of the simulation and provides
all coordinate, indexing, and stencil infrastructure used by the tensor and
backend layers.

## Lattice Concept

Every lattice type must satisfy the ``Lattice[D]`` concept, parameterized by
the number of spatial dimensions ``D`` (a compile-time ``static[int]``).
A conforming type exposes three arrays:

| Field | Type | Description |
|-------|------|-------------|
| ``globalGrid`` | ``array[D, int]`` | Full extent per dimension |
| ``mpiGrid`` | ``array[D, int]`` | MPI process grid (``-1`` for auto) |
| ``ghostGrid`` | ``array[D, int]`` | Ghost (halo) width per dimension |

The concept lives in its own module (``lattice/latticeconcept``) to break
circular imports between ``lattice.nim`` and ``lattice/stencil.nim``.

## SimpleCubicLattice

The only concrete lattice type currently provided is
``SimpleCubicLattice[D]``, which satisfies ``Lattice[D]``.

### Construction

```nim
import reliq

# Minimal: auto-detect MPI decomposition, no ghosts
let lat = newSimpleCubicLattice([8, 8, 8, 16])

# Explicit MPI grid (4 ranks along t-axis)
let lat2 = newSimpleCubicLattice([8, 8, 8, 16], [1, 1, 1, 4])

# Full control: MPI grid + ghost widths
let lat3 = newSimpleCubicLattice([8, 8, 8, 16], [1, 1, 1, 4], [1, 1, 1, 1])
```

- **``globalGrid``** — The total lattice extent.  All dimensions must be
  evenly divisible by their MPI grid factors.
- **``mpiGrid``** — How MPI ranks are distributed.  Use ``-1`` for
  auto-detection via Global Arrays (requires GA to be initialized first).
- **``ghostGrid``** — Width of the ghost/halo region per dimension.  Used by
  stencils and ghost exchange.  Default is ``0`` (no ghosts).

### Properties

```nim
lat.globalGrid   # [8, 8, 8, 16]
lat.mpiGrid      # [1, 1, 1, 4]
lat.ghostGrid    # [1, 1, 1, 1]
```

## Coordinate Indexing

The ``lattice/indexing`` module provides pure-function coordinate conversion
utilities.  All use **little-endian** flat-index ordering: dimension 0 varies
fastest.

| Proc | Signature | Description |
|------|-----------|-------------|
| ``flatToCoords`` | ``(idx: int, dims: array[D, int]): array[D, int]`` | Flat index → coordinates |
| ``coordsToFlat`` | ``(coords, dims: array[D, int]): int`` | Coordinates → flat index |
| ``localToGlobalCoords`` | ``(localCoords, lo: array[D, int]): array[D, int]`` | Local → global offset |
| ``globalToLocalCoords`` | ``(globalCoords, lo: array[D, int]): array[D, int]`` | Global → local offset |
| ``localFlatToGlobalFlat`` | ``(localIdx: int, localDims, globalDims, lo): int`` | Flat local → flat global |
| ``globalFlatToLocalFlat`` | ``(globalIdx: int, localDims, globalDims, lo): int`` | Flat global → flat local |

**Example:**

```nim
import reliq

let dims = [4, 4]
let coords = flatToCoords(5, dims)   # [1, 1]  (5 = 1 + 1*4)
let flat = coordsToFlat(coords, dims) # 5
```

## Stencil Operations

The unified stencil system lives in ``lattice/stencil`` and provides a single
``LatticeStencil[D]`` type that works identically across all backends.

### Stencil Patterns

A ``StencilPattern[D]`` defines the shape of a stencil as a set of offsets
relative to the center site:

```nim
import reliq

let lat = newSimpleCubicLattice([8, 8, 8, 16])

# Built-in nearest-neighbor stencil (infers D from lattice)
let nn = nearestNeighborStencil(lat)
echo nn.name      # "nearest_neighbor"
echo nn.nPoints   # 8  (±1 in each of 4 dims)

# Built-in Laplacian stencil
let lap = laplacianStencil(lat)

# Custom patterns
let lat2D = newSimpleCubicLattice([8, 8])
var custom = newStencilPattern(lat2D, "custom")
custom.addPoint([2, 0])   # Two steps in x
custom.addPoint([0, 2])   # Two steps in y
```

### LatticeStencil

A ``LatticeStencil[D]`` binds a pattern to a specific lattice, pre-computing
all neighbor offsets (including ghost region handling):

```nim
import reliq

let lat = newSimpleCubicLattice([8, 8, 8, 16], [1, 1, 1, 1], [1, 1, 1, 1])

# Create stencil from pattern and lattice
let stencil = newLatticeStencil(nearestNeighborStencil(lat), lat)

# Direct construction from lattice (uses nearest-neighbor by default)
let stencil2 = newLatticeStencil(lat)
```

### Neighbor Access

Inside an ``each`` loop, use the **shift API**:

```nim
for n in each 0..<vC.numSites():
  let fwd_x = stencil.fwd(n, 0)   # Forward neighbor in x
  let bwd_x = stencil.bwd(n, 0)   # Backward neighbor in x
  let fwd_t = stencil.fwd(n, 3)   # Forward neighbor in t
  vC[n] = vA[fwd_x] + vA[bwd_x] - 2.0 * vA[n]
```

- ``stencil.fwd(n, dim)`` — Returns a ``StencilShift`` for the forward
  neighbor of site ``n`` in dimension ``dim``.
- ``stencil.bwd(n, dim)`` — Backward neighbor.
- ``stencil.shift(n, point)`` — Access an arbitrary stencil point by index.

### Path-Based Stencils

For Wilson loops and transport paths:

```nim
# Plaquette path: fwd(mu), fwd(nu), bwd(mu), bwd(nu)
let path = plaquettePath(0, 1)  # xy-plaquette

# Rectangle path
let rect = rectanglePath(0, 1, 2, 1)  # 2×1 rectangle in xy-plane

# Convert path to stencil pattern
let pattern = pathToStencil(path, lat)
```

### Direction Types

Type-safe direction indexing:

```nim
let d = Direction(0)
echo d    # 0
let sd = forward(d)    # SignedDirection(dir: 0, sign: +1)
let bd = backward(d)   # SignedDirection(dir: 0, sign: -1)

# Named constants
echo X   # Direction(0)
echo Y   # Direction(1)
echo Z   # Direction(2)
echo T   # Direction(3)

# Iterators
for d in directions(4):
  echo d  # 0, 1, 2, 3

for sd in allDirections(4):
  echo sd  # ±0, ±1, ±2, ±3
```

### Backend Detection

The stencil system auto-detects the active backend at construction time:

| Backend | ``StencilBackend`` | Offset format |
|---------|-------------------|---------------|
| OpenCL | ``sbOpenCL`` | Flat offsets for GPU buffers |
| SYCL | ``sbSycl`` | Flat offsets for GPU buffers |
| OpenMP (SIMD) | ``sbSimd`` | Outer/lane offset pairs |
| Scalar | ``sbScalar`` | Flat coordinate offsets |

## Module Reference

| Module | Description |
|--------|-------------|
| [lattice/latticeconcept](lattice/latticeconcept.html) | ``Lattice[D]`` concept definition |
| [lattice/simplecubiclattice](lattice/simplecubiclattice.html) | ``SimpleCubicLattice[D]`` implementation |
| [lattice/stencil](lattice/stencil.html) | Unified stencil operations |
| [lattice/indexing](lattice/indexing.html) | Coordinate conversion utilities |
