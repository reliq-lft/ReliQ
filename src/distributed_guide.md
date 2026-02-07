# Distributed Memory

ReliQ uses [Global Arrays](https://globalarrays.github.io/) (GA) for
distributed memory management and MPI for inter-process communication.
The GA layer provides one-sided (PGAS) access to distributed arrays with
automatic ghost region management.

## Architecture

```
┌─────────────────────────────────────┐
│       TensorField / GlobalShifter   │  ← User-facing
├─────────────────────────────────────┤
│        GlobalArray[D, T]            │  ← Nim wrapper
├─────────────────────────────────────┤
│     GA C API (ga.h / macdecls.h)    │  ← FFI bindings
├─────────────────────────────────────┤
│    MPI (message passing)            │  ← Communication
└─────────────────────────────────────┘
```

## GlobalArray Type

``GlobalArray[D, T]`` wraps a single GA handle with Nim-level resource
management (RAII via ``=destroy``):

```nim
import reliq

# Usually created indirectly through TensorField:
var field = lat.newTensorField([3, 3]): float64
# field.data is a GlobalArray[D+R+1, T]

# Direct construction (advanced use):
var ga = newGlobalArray[4](
  globalGrid = [8, 8, 8, 16],
  mpiGrid    = [1, 1, 1, 4],
  ghostGrid  = [1, 1, 1, 1],
  T = float64
)
```

### Properties

| Proc | Returns | Description |
|------|---------|-------------|
| ``numSites()`` | ``int`` | Total sites in the global array |
| ``getGlobalGrid()`` | ``array[D, int]`` | Full grid dimensions |
| ``getLocalGrid()`` | ``array[D, int]`` | This rank's local partition |
| ``getMPIGrid()`` | ``array[D, int]`` | MPI process grid |
| ``getGhostGrid()`` | ``array[D, int]`` | Ghost widths per dimension |
| ``getBounds()`` | ``(lo, hi)`` | This rank's index bounds |

### Local Data Access

```nim
# Get a pointer to this rank's local data (including ghost region)
let ptr = ga.accessLocal()

# ... use the pointer ...

# Release the local access
ga.releaseLocal()

# Access ghost data
let ghostPtr = ga.accessGhosts()
```

### Ghost Exchange

```nim
# Update ghost regions in one direction
ga.updateGhostDirection(dim=3, direction=1)

# Update all ghost regions
ga.updateGhosts()
```

**GA 5.8.2 Limitation:** All dimensions must have ghost width ≥ 1 for
``GA_Update_ghost_dir`` to work correctly.  Ghost width 0 on any dimension
causes a crash ("cannot locate region" with invalid bounds).

## MPI Integration

The ``globalarrays/gampi`` module provides MPI initialization and collective
operations:

### Initialization

```nim
import reliq

# The parallel: template handles init/finalize automatically:
parallel:
  echo "Rank ", GA_Nodeid(), " of ", GA_Nnodes()
  # ...

# Equivalent to:
initMPI()
initGA()
# ... user code ...
finalizeGA()
finalizeMPI()
```

### Collective Operations

```nim
# Barrier synchronization (GA level)
GA_Sync()

# Typed all-reduce operations
var localSum = 42.0
allReduceFloat64(addr localSum, 1)  # In-place sum across ranks

var localInt: int32 = 10
allReduceInt32(addr localInt, 1)

# GA broadcast (rank 0 sends to all)
var data: float64 = 3.14
GA_Brdcst(addr data, sizeof(float64).cint, 0.cint)
```

## The ``parallel`` Template

The ``parallel:`` block template sets up MPI + GA and handles cleanup:

```nim
import reliq

parallel:
  # MPI and GA are initialized here
  echo "Running on ", GA_Nnodes(), " ranks"

  block:
    # Create distributed objects inside a block:
    var field = lat.newTensorField([3, 3]): float64
    # ... computation ...
  # GA objects destroyed at block exit (before finalizeGA)

# MPI and GA finalized automatically
```

**Important:** Use ``block:`` scoping for GA-backed objects.  They must be
destroyed before ``finalizeGA()`` is called.

## Running with MPI

ReliQ provides a launcher script (``reliq``) that dispatches to ``mpirun``:

```
# Run on 4 MPI ranks
./reliq -e tensor -n 4

# Run on 1 rank (default)
./reliq -e tensor
```

## Data Transfer Between GA and Host

The ``LocalTensorField`` contiguous buffer approach handles the padded GA
memory layout automatically:

```nim
parallel:
  let lat = newSimpleCubicLattice([8, 8, 8, 16])

  block:
    var field = lat.newTensorField([3, 3]): float64

    # Copies real data from padded GA → contiguous host buffer
    var local = field.newLocalTensorField()

    # Work with local data...
    for n in all 0..<local.numSites():
      var site = local.getSite(n)
      site[0, 0] = 1.0

    # Copy contiguous buffer back → padded GA
    local.releaseLocalTensorField()

    # Ghost exchange after modification
    field.updateAllGhosts()
```

## Module Reference

| Module | Description |
|--------|-------------|
| [globalarrays/gatypes](globalarrays/gatypes.html) | ``GlobalArray[D,T]`` distributed array wrapper |
| [globalarrays/gabase](globalarrays/gabase.html) | GA initialization and finalization |
| [globalarrays/gawrap](globalarrays/gawrap.html) | Low-level C FFI to Global Arrays |
| [globalarrays/gampi](globalarrays/gampi.html) | MPI initialization and collective operations |
