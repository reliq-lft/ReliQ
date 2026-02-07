# SIMD Infrastructure

ReliQ's SIMD layer provides generic vector types and AoSoA memory layout
computation for CPU-vectorized lattice traversal.  This infrastructure is used
primarily by the OpenMP backend but is available to all backends.

## SIMD Vector Types

The ``simd/simdtypes`` module provides ``SimdVec[N, T]``, a generic SIMD
vector wrapper that abstracts over different hardware SIMD widths.

### Static-Width Vectors

```nim
import reliq

# Generic construction
var v = SimdVec[4, float64]()
v[0] = 1.0
v[1] = 2.0

# Broadcast
let ones = splat[4, float64](1.0)    # [1.0, 1.0, 1.0, 1.0]
let z = zero[4, float64]()           # [0.0, 0.0, 0.0, 0.0]
```

### Pre-Defined Type Aliases

| Type | Width | Element | Hardware |
|------|-------|---------|----------|
| ``SimdF32x4`` | 4 | ``float32`` | SSE |
| ``SimdF32x8`` | 8 | ``float32`` | AVX2 |
| ``SimdF32x16`` | 16 | ``float32`` | AVX-512 |
| ``SimdF64x2`` | 2 | ``float64`` | SSE |
| ``SimdF64x4`` | 4 | ``float64`` | AVX2 |
| ``SimdF64x8`` | 8 | ``float64`` | AVX-512 |
| ``SimdI32x4/8/16`` | 4/8/16 | ``int32`` | SSE/AVX2/AVX-512 |
| ``SimdI64x2/4/8`` | 2/4/8 | ``int64`` | SSE/AVX2/AVX-512 |

### Arithmetic Operations

```nim
var a = SimdF64x4(data: [1.0, 2.0, 3.0, 4.0])
var b = SimdF64x4(data: [5.0, 6.0, 7.0, 8.0])

# Element-wise operations
let c = a + b        # [6.0, 8.0, 10.0, 12.0]
let d = a * b        # [5.0, 12.0, 21.0, 32.0]
let e = a - b        # [-4.0, -4.0, -4.0, -4.0]
let f = a / b        # [0.2, 0.333..., 0.428..., 0.5]

# Scalar operations
let g = 2.0 * a      # [2.0, 4.0, 6.0, 8.0]
let h = a + 1.0      # [2.0, 3.0, 4.0, 5.0]

# In-place operations
a += b
a -= b
a *= 2.0

# Reductions
let s = a.sum()      # 10.0
let p = a.product()  # 24.0
let mn = a.min()     # 1.0
let mx = a.max()     # 4.0
```

### Memory Operations

```nim
var buf: array[8, float64]

# Load from contiguous memory
let v = load[4, float64](addr buf[0])

# Store to contiguous memory
store(v, addr buf[0])

# Strided load/store (e.g., for gather/scatter)
let vs = loadStrided[4, float64](addr buf[0], stride=2)
storeStrided(vs, addr buf[0], stride=2)
```

### Hardware Intrinsics

When compiled with ``-d:AVX2`` or ``-d:AVX512``, the generic loop-based
implementations are replaced with hardware-specific intrinsics:

| Compile flag | Procs accelerated | Instructions |
|-------------|-------------------|--------------|
| ``-d:AVX2`` | ``SimdF32x8``, ``SimdF64x4`` add/sub/mul/madd | ``_mm256_*`` |
| ``-d:AVX512`` | ``SimdF32x16``, ``SimdF64x8`` add/sub/mul/madd | ``_mm512_*`` |

The ``madd`` (fused multiply-add) proc computes ``a * b + c`` in a single
instruction when available.

### Dynamic-Width Vectors

``SimdVecDyn[T]`` supports runtime-configurable SIMD width:

```nim
let dyn = newSimdVecDyn[float64](4, 1.0)  # 4 lanes, filled with 1.0
echo dyn.width  # 4
let sum = dyn.sum()  # 4.0
```

## SIMD Lattice Layout

The ``simd/simdlayout`` module computes AoSoA memory layouts for
SIMD-vectorized lattice traversal.

### Key Concepts

The layout splits each lattice dimension into:

- **innerGeom** — Sites within a SIMD group (lane indices)
- **outerGeom** — SIMD groups (outer loop indices)

```
localGeom = [8, 8, 8, 16]
simdGrid  = [1, 1, 1, 8]     ← 8 lanes along t-axis
innerGeom = [1, 1, 1, 8]     ← each SIMD group is 8 sites in t
outerGeom = [8, 8, 8, 2]     ← 2 outer groups in t (16/8)
```

### Construction

```nim
import reliq

# Explicit SIMD grid
let layout = newSimdLatticeLayout([8, 8, 8, 16], [1, 1, 1, 8])

# Auto-distributed (fills fastest-varying dims first)
let layout2 = newSimdLatticeLayout([8, 8, 8, 16], simdWidth=8)
```

### Properties

| Property | Description |
|----------|-------------|
| ``nSitesInner`` | Sites per SIMD group (= product of innerGeom) |
| ``nSitesOuter`` | Number of SIMD groups |
| ``nSites`` | Total local sites |
| ``simdLanes(layout)`` | Alias for ``nSitesInner`` |
| ``vectorGroups(layout)`` | Alias for ``nSitesOuter`` |

### Index Conversions

```nim
# Local site ↔ (outer, inner)
let (outer, inner) = localToOuterInner(localIdx, layout)
let local = outerInnerToLocal(outer, inner, layout)

# AoSoA index (for data buffers)
let idx = aosoaIndex(outer, inner, elemIdx, elemsPerSite, nSitesInner)

# Coordinate lookup table
let table = generateCoordTable(layout)
# table[outer][lane] → local site index
```

### AoSoA Index Formula

For a site at ``(outerIdx, innerIdx)`` with element ``elemIdx``:

$$\text{index} = \text{outerIdx} \times (\text{elemsPerSite} \times \text{nSitesInner}) + \text{elemIdx} \times \text{nSitesInner} + \text{innerIdx}$$

This layout ensures that consecutive SIMD lanes access consecutive memory
addresses for the same element — optimal for SIMD ``load``/``store``
instructions.

### Validation

```nim
let (valid, msg) = validateSimdGrid([8, 8, 8, 16], [1, 1, 1, 4], [1, 1, 1, 8])
if not valid:
  echo msg  # Error: SIMD grid exceeds local geometry
```

## Module Reference

| Module | Description |
|--------|-------------|
| [simd/simdtypes](simd/simdtypes.html) | ``SimdVec[N,T]`` and hardware intrinsic wrappers |
| [simd/simdlayout](simd/simdlayout.html) | ``SimdLatticeLayout`` and AoSoA index computation |
