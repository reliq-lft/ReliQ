#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/lattice/stencil.nim
  Contact: reliq-lft@proton.me

  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 reliq-lft
]#

## Unified Stencil Operations for Lattice Field Theory
## ====================================================
##
## This module provides a single, unified stencil type that works identically
## across all backends (OpenMP, OpenCL, SYCL) and handles distributed lattices
## with ghost regions automatically.
##
## Key Design Principles:
## - **One Type**: `LatticeStencil[D]` is the only stencil type you need
## - **Always Distributed**: Lattices are always distributed, so stencils always
##   understand ghost regions and padded layouts
## - **Backend Agnostic**: Same API inside `each` loops for all backends
## - **Clean Operators**: Use `view[stencil.shift(n, dir)]` for neighbor access
##
## Usage Example:
## ```nim
## # Create stencil from lattice - D is inferred automatically
## let stencil = newLatticeStencil(nearestNeighborStencil(lat), lat)
## # Or even simpler (defaults to nearest-neighbor):
## let stencil = newLatticeStencil(lat)
##
## # Inside any backend's each loop:
## for n in each 0..<view.numSites:
##   for dir in 0..<4:
##     # Forward neighbor
##     let fwd = view[stencil.shift(n, +1, dir)]
##     # Backward neighbor  
##     let bwd = view[stencil.shift(n, -1, dir)]
##     result[n] = fwd + bwd - 2.0 * view[n]
## ```
##
## The stencil handles:
## - Periodic boundary conditions within local domain
## - Ghost region detection and padded coordinate mapping
## - SIMD-aware indexing for vectorized backends
## - Device buffer offsets for GPU backends

import std/[tables]

#[ ============================================================================
   Direction Types - Type-safe direction handling
   ============================================================================ ]#

type
  Direction* = distinct int
    ## Type-safe direction index (0 = x, 1 = y, 2 = z, 3 = t for 4D)

  SignedDirection* = object
    ## Direction with sign for forward (+1) or backward (-1) shifts
    dir*: Direction
    sign*: int  # +1 or -1

proc `$`*(d: Direction): string = "dir" & $d.int
proc `$`*(sd: SignedDirection): string =
  (if sd.sign > 0: "+" else: "-") & $sd.dir

proc `==`*(a, b: Direction): bool {.borrow.}
proc hash*(d: Direction): int = d.int

proc forward*(d: Direction): SignedDirection {.inline.} =
  SignedDirection(dir: d, sign: +1)

proc backward*(d: Direction): SignedDirection {.inline.} =
  SignedDirection(dir: d, sign: -1)

template X*: Direction = Direction(0)
template Y*: Direction = Direction(1)
template Z*: Direction = Direction(2)
template T*: Direction = Direction(3)

iterator directions*(nDim: int): Direction =
  for d in 0..<nDim:
    yield Direction(d)

iterator allDirections*(nDim: int): SignedDirection =
  ## Iterate over all directions (forward and backward)
  for d in 0..<nDim:
    yield forward(Direction(d))
    yield backward(Direction(d))

#[ ============================================================================
   Stencil Point - A single offset in a stencil pattern
   ============================================================================ ]#

type
  StencilPoint*[D: static int] = object
    ## A single point in a stencil pattern (offset from center)
    offset*: array[D, int]
    id*: int

proc newStencilPoint*[D: static int](offset: array[D, int], id: int = 0): StencilPoint[D] =
  StencilPoint[D](offset: offset, id: id)

proc newStencilPoint*[D: static int](sd: SignedDirection): StencilPoint[D] =
  var offset: array[D, int]
  for i in 0..<D: offset[i] = 0
  offset[sd.dir.int] = sd.sign
  StencilPoint[D](offset: offset, id: 0)

proc `$`*[D: static int](p: StencilPoint[D]): string =
  result = "["
  for i, o in p.offset:
    if i > 0: result &= ", "
    result &= $o
  result &= "]"

#[ ============================================================================
   Stencil Pattern - Collection of stencil points
   ============================================================================ ]#

type
  StencilPattern*[D: static int] = object
    ## A stencil pattern defining neighbor offsets
    points*: seq[StencilPoint[D]]
    name*: string

proc newStencilPattern*[D: static int](name: string = ""): StencilPattern[D] =
  StencilPattern[D](points: @[], name: name)

proc addPoint*[D: static int](s: var StencilPattern[D], offset: array[D, int]) =
  s.points.add(StencilPoint[D](offset: offset, id: s.points.len))

proc addPoint*[D: static int](s: var StencilPattern[D], sd: SignedDirection) =
  var offset: array[D, int]
  for i in 0..<D: offset[i] = 0
  offset[sd.dir.int] = sd.sign
  s.points.add(StencilPoint[D](offset: offset, id: s.points.len))

proc addPoint*[D: static int](s: var StencilPattern[D], sign: int, dir: int) =
  var offset: array[D, int]
  for i in 0..<D: offset[i] = 0
  offset[dir] = sign
  s.points.add(StencilPoint[D](offset: offset, id: s.points.len))

proc nPoints*[D: static int](s: StencilPattern[D]): int {.inline.} = s.points.len

proc `$`*[D: static int](s: StencilPattern[D]): string =
  result = "StencilPattern"
  if s.name.len > 0: result &= "[" & s.name & "]"
  result &= "(" & $s.nPoints & " points)"

# Backward compatibility alias
type Stencil*[D: static int] = StencilPattern[D]
proc newStencil*[D: static int](name: string = ""): Stencil[D] = newStencilPattern[D](name)

#[ ============================================================================
   Common Stencil Patterns
   ============================================================================ ]#

proc nearestNeighborStencil*[D: static int](): StencilPattern[D] =
  ## Create a nearest-neighbor stencil with 2D points (forward and backward in each dimension).
  result = newStencilPattern[D]("nearest-neighbor")
  for d in 0..<D:
    result.addPoint(+1, d)  # Forward
    result.addPoint(-1, d)  # Backward

proc laplacianStencil*[D: static int](): StencilPattern[D] =
  result = nearestNeighborStencil[D]()
  result.name = "laplacian"

proc forwardStencil*[D: static int](): StencilPattern[D] =
  ## Forward-only stencil (+x, +y, +z, +t)
  result = newStencilPattern[D]("forward")
  for d in 0..<D:
    result.addPoint(+1, d)

proc backwardStencil*[D: static int](): StencilPattern[D] =
  ## Backward-only stencil (-x, -y, -z, -t)
  result = newStencilPattern[D]("backward")
  for d in 0..<D:
    result.addPoint(-1, d)

#[ ============================================================================
   Path-based Stencil Construction (for Wilson loops, etc.)
   ============================================================================ ]#

type
  PathStep* = object
    dir*: int
    sign*: int

proc step*(dir: int, forward: bool = true): PathStep =
  PathStep(dir: dir, sign: if forward: +1 else: -1)

proc fwd*(dir: int): PathStep = step(dir, true)
proc bwd*(dir: int): PathStep = step(dir, false)

proc pathToStencil*[D: static int](path: openArray[PathStep]): StencilPattern[D] =
  result = newStencilPattern[D]("path")
  var currentOffset: array[D, int]
  for i in 0..<D: currentOffset[i] = 0
  result.addPoint(currentOffset)  # Origin
  for step in path:
    currentOffset[step.dir] += step.sign
    result.addPoint(currentOffset)

proc rectanglePath*(mu, nu: int, a, b: int): seq[PathStep] =
  result = @[]
  for i in 0..<a: result.add(fwd(mu))
  for i in 0..<b: result.add(fwd(nu))
  for i in 0..<a: result.add(bwd(mu))
  for i in 0..<b: result.add(bwd(nu))

proc plaquettePath*(mu, nu: int): seq[PathStep] =
  rectanglePath(mu, nu, 1, 1)

#[ ============================================================================
   Stencil Backend Configuration
   ============================================================================ ]#

const
  UseOpenCL* {.booldefine.} = false
  UseSYCL* {.booldefine.} = false
  UseOpenMP* {.booldefine.} = false
  VectorWidth* {.intdefine.} = 1

type
  StencilBackend* = enum
    sbScalar     ## Scalar CPU (no SIMD)
    sbSimd       ## SIMD-vectorized CPU (OpenMP)
    sbOpenCL     ## OpenCL GPU
    sbSycl       ## SYCL GPU

proc detectBackend(): StencilBackend {.compileTime.} =
  when UseOpenCL:
    sbOpenCL
  elif UseSYCL:
    sbSycl
  elif UseOpenMP and VectorWidth > 1:
    sbSimd
  else:
    sbScalar

#[ ============================================================================
   Coordinate/Index Utilities
   ============================================================================ ]#

proc computeStrides[D: static int](geom: array[D, int]): array[D, int] =
  result[0] = 1
  for d in 1..<D:
    result[d] = result[d-1] * geom[d-1]

proc lexToCoords[D: static int](idx: int, geom: array[D, int], strides: array[D, int]): array[D, int] =
  var remaining = idx
  for d in countdown(D-1, 0):
    result[d] = remaining div strides[d]
    remaining = remaining mod strides[d]

proc coordsToLex[D: static int](coords: array[D, int], strides: array[D, int]): int =
  result = 0
  for d in 0..<D:
    result += coords[d] * strides[d]

#[ ============================================================================
   LatticeStencil - THE Unified Stencil Type
   ============================================================================
   
   This is the ONLY stencil type you need. It handles:
   - Local lattice geometry
   - Ghost regions for distributed layouts
   - Padded coordinate mapping
   - All backends (OpenMP, OpenCL, SYCL)
   
   The stencil always operates on the LOCAL portion of a distributed lattice,
   with ghost regions available for neighbor access across MPI boundaries.
]#

type
  LatticeStencil*[D: static int] = object
    ## Unified stencil for all backends and distributed lattices
    ##
    ## Provides a single, clean API for neighbor access that works
    ## identically whether you're using OpenMP, OpenCL, or SYCL.
    pattern*: StencilPattern[D]
    localGeom*: array[D, int]     # Local lattice size (no ghosts)
    ghostWidth*: array[D, int]    # Ghost width per dimension
    paddedGeom*: array[D, int]    # Total size including ghosts
    nLocalSites*: int             # Sites in local region
    nPaddedSites*: int            # Sites including ghosts
    nPoints*: int                 # Number of stencil points
    backend*: StencilBackend
    
    # Pre-computed lookup tables
    offsets*: seq[int32]          # neighbor offset for each (site, point)
    isGhost*: seq[bool]           # whether neighbor is in ghost region
    
    # SIMD support
    simdGrid*: array[D, int]
    nSitesInner*: int             # SIMD lanes
    nSitesOuter*: int             # Outer loop iterations
    nbrOuter*: seq[int32]         # SIMD: neighbor outer index
    nbrLane*: seq[int32]          # SIMD: neighbor lane index

# Alias for number of local sites (main iteration count)
proc nSites*[D: static int](s: LatticeStencil[D]): int {.inline.} = s.nLocalSites

#[ ============================================================================
   LatticeStencil Construction
   ============================================================================ ]#

proc defaultSimdGrid[D: static int](localGeom: array[D, int]): array[D, int] =
  for d in 0..<D: result[d] = 1
  var remaining = VectorWidth
  var d = 0
  while remaining > 1 and d < D:
    var factor = 1
    while factor * 2 <= remaining and 
          factor * 2 <= localGeom[d] and
          (localGeom[d] mod (factor * 2)) == 0:
      factor *= 2
    result[d] = factor
    remaining = remaining div factor
    d.inc

proc simdGridIsValid[D: static int](localGeom, simdGrid: array[D, int]): bool =
  for d in 0..<D:
    if localGeom[d] mod simdGrid[d] != 0 or simdGrid[d] <= 0:
      return false
  return true

proc buildLookupTables[D: static int](s: var LatticeStencil[D]) =
  ## Build pre-computed offset tables for fast neighbor access
  let localStrides = computeStrides(s.localGeom)
  let paddedStrides = computeStrides(s.paddedGeom)
  
  s.offsets = newSeq[int32](s.nLocalSites * s.nPoints)
  s.isGhost = newSeq[bool](s.nLocalSites * s.nPoints)
  
  for localSite in 0..<s.nLocalSites:
    let localCoords = lexToCoords(localSite, s.localGeom, localStrides)
    
    # Convert to padded coordinates (shift by ghost width)
    var paddedCoords: array[D, int]
    for d in 0..<D:
      paddedCoords[d] = localCoords[d] + s.ghostWidth[d]
    
    for pointIdx, point in s.pattern.points:
      var nbrPaddedCoords: array[D, int]
      var ghostFlag = false
      
      for d in 0..<D:
        nbrPaddedCoords[d] = paddedCoords[d] + point.offset[d]
        
        # Check if in ghost region
        if nbrPaddedCoords[d] < s.ghostWidth[d]:
          ghostFlag = true
        elif nbrPaddedCoords[d] >= s.ghostWidth[d] + s.localGeom[d]:
          ghostFlag = true
        
        # Apply periodic wrapping within padded domain
        if nbrPaddedCoords[d] < 0:
          nbrPaddedCoords[d] += s.paddedGeom[d]
        elif nbrPaddedCoords[d] >= s.paddedGeom[d]:
          nbrPaddedCoords[d] -= s.paddedGeom[d]
      
      let entryIdx = localSite * s.nPoints + pointIdx
      s.offsets[entryIdx] = coordsToLex(nbrPaddedCoords, paddedStrides).int32
      s.isGhost[entryIdx] = ghostFlag

proc buildSimdTables[D: static int](s: var LatticeStencil[D]) =
  ## Build SIMD-aware lookup tables
  var innerGeom, outerGeom: array[D, int]
  for d in 0..<D:
    innerGeom[d] = s.simdGrid[d]
    outerGeom[d] = s.localGeom[d] div s.simdGrid[d]
  
  s.nSitesInner = 1
  s.nSitesOuter = 1
  for d in 0..<D:
    s.nSitesInner *= innerGeom[d]
    s.nSitesOuter *= outerGeom[d]
  
  let totalEntries = s.nSitesOuter * s.nSitesInner * s.nPoints
  s.nbrOuter = newSeq[int32](totalEntries)
  s.nbrLane = newSeq[int32](totalEntries)
  
  let innerStrides = computeStrides(innerGeom)
  let outerStrides = computeStrides(outerGeom)
  let paddedStrides = computeStrides(s.paddedGeom)
  
  for outer in 0..<s.nSitesOuter:
    let outerCoords = lexToCoords(outer, outerGeom, outerStrides)
    
    for lane in 0..<s.nSitesInner:
      let innerCoords = lexToCoords(lane, innerGeom, innerStrides)
      
      # Compute local coordinates
      var localCoords: array[D, int]
      for d in 0..<D:
        localCoords[d] = outerCoords[d] * innerGeom[d] + innerCoords[d]
      
      # Compute padded coordinates
      var paddedCoords: array[D, int]
      for d in 0..<D:
        paddedCoords[d] = localCoords[d] + s.ghostWidth[d]
      
      for pointIdx, point in s.pattern.points:
        var nbrPaddedCoords: array[D, int]
        for d in 0..<D:
          nbrPaddedCoords[d] = paddedCoords[d] + point.offset[d]
          # Periodic wrapping in padded domain
          if nbrPaddedCoords[d] < 0:
            nbrPaddedCoords[d] += s.paddedGeom[d]
          elif nbrPaddedCoords[d] >= s.paddedGeom[d]:
            nbrPaddedCoords[d] -= s.paddedGeom[d]
        
        # Convert neighbor padded coords back to (outer, lane)
        # First subtract ghost to get local coords
        var nbrLocalCoords: array[D, int]
        for d in 0..<D:
          nbrLocalCoords[d] = nbrPaddedCoords[d] - s.ghostWidth[d]
          # Wrap to local domain if needed
          if nbrLocalCoords[d] < 0:
            nbrLocalCoords[d] += s.localGeom[d]
          elif nbrLocalCoords[d] >= s.localGeom[d]:
            nbrLocalCoords[d] -= s.localGeom[d]
        
        var nbrOuterCoords, nbrInnerCoords: array[D, int]
        for d in 0..<D:
          nbrOuterCoords[d] = nbrLocalCoords[d] div innerGeom[d]
          nbrInnerCoords[d] = nbrLocalCoords[d] mod innerGeom[d]
        
        let nbrOuterIdx = coordsToLex(nbrOuterCoords, outerStrides)
        let nbrLaneIdx = coordsToLex(nbrInnerCoords, innerStrides)
        
        let entryIdx = (outer * s.nSitesInner + lane) * s.nPoints + pointIdx
        s.nbrOuter[entryIdx] = nbrOuterIdx.int32
        s.nbrLane[entryIdx] = nbrLaneIdx.int32

proc newLatticeStencil*[D: static int](
  pattern: StencilPattern[D],
  localGeom: array[D, int],
  ghostWidth: array[D, int],
  simdGrid: array[D, int]
): LatticeStencil[D] =
  ## Create a unified lattice stencil with explicit parameters
  ##
  ## Parameters:
  ##   pattern: Stencil pattern (e.g., nearestNeighborStencil(lat))
  ##   localGeom: Local lattice dimensions (without ghosts)
  ##   ghostWidth: Ghost width in each dimension
  ##   simdGrid: SIMD grid for vectorization (use [1,1,1,1] for no SIMD)
  result.pattern = pattern
  result.localGeom = localGeom
  result.ghostWidth = ghostWidth
  result.nPoints = pattern.nPoints
  result.backend = detectBackend()
  
  # Compute padded geometry
  for d in 0..<D:
    result.paddedGeom[d] = localGeom[d] + 2 * ghostWidth[d]
  
  # Compute site counts
  result.nLocalSites = 1
  result.nPaddedSites = 1
  for d in 0..<D:
    result.nLocalSites *= localGeom[d]
    result.nPaddedSites *= result.paddedGeom[d]
  
  # Build main lookup tables
  result.buildLookupTables()
  
  # Handle SIMD
  result.simdGrid = simdGrid
  if result.backend == sbSimd and simdGridIsValid(localGeom, simdGrid):
    result.buildSimdTables()
  else:
    # Fallback: no SIMD vectorization
    result.nSitesInner = 1
    result.nSitesOuter = result.nLocalSites

proc newLatticeStencil*[D: static int](
  pattern: StencilPattern[D],
  localGeom: array[D, int],
  ghostWidth: array[D, int]
): LatticeStencil[D] =
  ## Create stencil with default SIMD grid
  var simdGrid: array[D, int]
  when UseOpenMP and VectorWidth > 1:
    simdGrid = defaultSimdGrid[D](localGeom)
  else:
    for d in 0..<D: simdGrid[d] = 1
  newLatticeStencil(pattern, localGeom, ghostWidth, simdGrid)

proc newLatticeStencil*[D: static int](
  pattern: StencilPattern[D],
  localGeom: array[D, int]
): LatticeStencil[D] =
  ## Create stencil with zero ghost width (local-only periodic BC)
  var ghostWidth: array[D, int]
  for d in 0..<D: ghostWidth[d] = 0
  newLatticeStencil(pattern, localGeom, ghostWidth)

# Import Lattice concept for lattice-based constructors
import latticeconcept

#[ ============================================================================
   Lattice-based Stencil Pattern Constructors
   ============================================================================
   
   These overloads infer D from the Lattice type, so users never need to
   write nearestNeighborStencil[4]() — just nearestNeighborStencil(lat).
]#

proc nearestNeighborStencil*[D: static int, L: Lattice[D]](lat: L): StencilPattern[D] =
  ## Create a nearest-neighbor stencil, inferring D from the lattice
  ##
  ## Example:
  ##   let lat = newSimpleCubicLattice([8, 8, 8, 16])
  ##   let pattern = nearestNeighborStencil(lat)  # D=4 inferred
  nearestNeighborStencil[D]()

proc laplacianStencil*[D: static int, L: Lattice[D]](lat: L): StencilPattern[D] =
  ## Create a laplacian stencil, inferring D from the lattice
  laplacianStencil[D]()

proc forwardStencil*[D: static int, L: Lattice[D]](lat: L): StencilPattern[D] =
  ## Create a forward-only stencil, inferring D from the lattice
  forwardStencil[D]()

proc backwardStencil*[D: static int, L: Lattice[D]](lat: L): StencilPattern[D] =
  ## Create a backward-only stencil, inferring D from the lattice
  backwardStencil[D]()

proc newStencilPattern*[D: static int, L: Lattice[D]](lat: L, name: string = ""): StencilPattern[D] =
  ## Create an empty stencil pattern, inferring D from the lattice
  ##
  ## Example:
  ##   let lat = newSimpleCubicLattice([8, 8, 8, 16])
  ##   var custom = newStencilPattern(lat, "custom")
  ##   custom.addPoint([2, 0, 0, 0])
  newStencilPattern[D](name)

proc pathToStencil*[D: static int, L: Lattice[D]](path: openArray[PathStep], lat: L): StencilPattern[D] =
  ## Convert a path to a stencil pattern, inferring D from the lattice
  ##
  ## Example:
  ##   let lat = newSimpleCubicLattice([8, 8, 8, 16])
  ##   let pattern = pathToStencil(plaquettePath(0, 1), lat)
  pathToStencil[D](path)

import globalarrays/[gatypes]

proc newLatticeStencil*[D: static int, L: Lattice[D]](
  pattern: StencilPattern[D],
  lat: L
): LatticeStencil[D] =
  ## Create stencil from a Lattice type (preferred constructor)
  ##
  ## Automatically extracts local geometry and ghost width from the lattice.
  ## When `mpiGrid` uses auto-detect sentinels (values ≤ 0), queries
  ## GlobalArrays for the actual MPI decomposition.
  ##
  ## Example:
  ## ```nim
  ## let lat = newSimpleCubicLattice([16, 16, 16, 32], [2, 2, 2, 4], [1, 1, 1, 1])
  ## let stencil = newLatticeStencil(nearestNeighborStencil(lat), lat)
  ## ```
  var localGeom: array[D, int]
  var needsQuery = false
  for d in 0..<D:
    if lat.mpiGrid[d] <= 0:
      needsQuery = true
      break
  
  if needsQuery:
    # mpiGrid has auto-detect sentinels — query GlobalArrays for actual decomposition
    let tmpGA = newGlobalArray[D](lat.globalGrid, lat.mpiGrid, lat.ghostGrid, float32)
    localGeom = tmpGA.getLocalGrid()
  else:
    for d in 0..<D:
      localGeom[d] = lat.globalGrid[d] div lat.mpiGrid[d]
  
  newLatticeStencil(pattern, localGeom, lat.ghostGrid)

proc newLatticeStencil*[D: static int, L: Lattice[D]](
  lat: L
): LatticeStencil[D] =
  ## Create a nearest-neighbor stencil from a Lattice (most convenient constructor)
  ##
  ## Infers D and uses nearestNeighborStencil automatically.
  ##
  ## Example:
  ## ```nim
  ## let lat = newSimpleCubicLattice([8, 8, 8, 16])
  ## let stencil = newLatticeStencil(lat)  # 4D nearest-neighbor stencil
  ## ```
  newLatticeStencil(nearestNeighborStencil[D](), lat)

#[ ============================================================================
   Neighbor Access - The Core API
   ============================================================================ ]#

proc neighbor*[D: static int](s: LatticeStencil[D], site, point: int): int {.inline.} =
  ## Get neighbor index in PADDED layout
  ##
  ## This is the primary API for reading neighbors. The returned index
  ## is valid for accessing a padded tensor that includes ghost regions.
  s.offsets[site * s.nPoints + point].int

proc isGhostNeighbor*[D: static int](s: LatticeStencil[D], site, point: int): bool {.inline.} =
  ## Check if neighbor is in a ghost region
  s.isGhost[site * s.nPoints + point]

proc neighborSimd*[D: static int](
  s: LatticeStencil[D], 
  outer, lane, point: int
): tuple[outer, lane: int] {.inline.} =
  ## SIMD-aware neighbor lookup for vectorized backends
  if s.nbrOuter.len > 0:
    let idx = (outer * s.nSitesInner + lane) * s.nPoints + point
    (s.nbrOuter[idx].int, s.nbrLane[idx].int)
  else:
    let site = outer * s.nSitesInner + lane
    let nbr = s.neighbor(site, point)
    (nbr, 0)

#[ ============================================================================
   Shift API - Clean syntax for direction-based access
   ============================================================================ ]#

type
  StencilShift*[D: static int] = object
    ## Result of stencil.shift() - used with view[] for neighbor access
    stencil*: ptr LatticeStencil[D]
    site*: int
    point*: int
    neighborIdx*: int

proc shift*[D: static int](
  s: LatticeStencil[D], 
  site: int, 
  sign: int, 
  dir: int
): StencilShift[D] {.inline.} =
  ## Get a stencil shift for neighbor access
  ##
  ## Usage: view[stencil.shift(n, +1, 0)]  # Forward neighbor in x
  ##        view[stencil.shift(n, -1, 3)]  # Backward neighbor in t
  ##
  ## The point index is computed from (sign, dir) using the pattern:
  ##   Forward stencil points are at even indices (0, 2, 4, ...)
  ##   Backward stencil points are at odd indices (1, 3, 5, ...)
  let point = if sign > 0: 2 * dir else: 2 * dir + 1
  result.stencil = unsafeAddr s
  result.site = site
  result.point = point
  result.neighborIdx = s.neighbor(site, point)

proc shift*[D: static int](
  s: LatticeStencil[D], 
  site: int, 
  sd: SignedDirection
): StencilShift[D] {.inline.} =
  ## Get a stencil shift using SignedDirection
  s.shift(site, sd.sign, sd.dir.int)

proc fwd*[D: static int](s: LatticeStencil[D], site: int, dir: int): StencilShift[D] {.inline.} =
  ## Shorthand for forward shift
  s.shift(site, +1, dir)

proc bwd*[D: static int](s: LatticeStencil[D], site: int, dir: int): StencilShift[D] {.inline.} =
  ## Shorthand for backward shift
  s.shift(site, -1, dir)

# Get the neighbor index from a shift (for direct use)
proc idx*(sh: StencilShift): int {.inline.} = sh.neighborIdx

#[ ============================================================================
   Iteration Helpers
   ============================================================================ ]#

iterator sites*[D: static int](s: LatticeStencil[D]): int {.inline.} =
  for i in 0..<s.nLocalSites:
    yield i

iterator points*[D: static int](s: LatticeStencil[D]): int {.inline.} =
  for p in 0..<s.nPoints:
    yield p

iterator neighbors*[D: static int](s: LatticeStencil[D], site: int): int {.inline.} =
  for p in 0..<s.nPoints:
    yield s.neighbor(site, p)

template forEachNeighbor*[D: static int](
  s: LatticeStencil[D], 
  site: int, 
  pointVar, nbrVar: untyped, 
  body: untyped
) =
  for pointVar in 0..<s.nPoints:
    let nbrVar = s.neighbor(site, pointVar)
    body

proc nLanes*[D: static int](s: LatticeStencil[D]): int {.inline.} = s.nSitesInner
proc nOuter*[D: static int](s: LatticeStencil[D]): int {.inline.} = s.nSitesOuter

#[ ============================================================================
   Coordinate Conversion Utilities
   ============================================================================ ]#

proc localToPadded*[D: static int](s: LatticeStencil[D], localSite: int): int =
  ## Convert local site index to padded site index
  let localStrides = computeStrides(s.localGeom)
  let paddedStrides = computeStrides(s.paddedGeom)
  let localCoords = lexToCoords(localSite, s.localGeom, localStrides)
  var paddedCoords: array[D, int]
  for d in 0..<D:
    paddedCoords[d] = localCoords[d] + s.ghostWidth[d]
  coordsToLex(paddedCoords, paddedStrides)

proc paddedToLocal*[D: static int](s: LatticeStencil[D], paddedSite: int): int =
  ## Convert padded site index to local site index
  ## Returns -1 if site is in ghost region
  let paddedStrides = computeStrides(s.paddedGeom)
  let localStrides = computeStrides(s.localGeom)
  let paddedCoords = lexToCoords(paddedSite, s.paddedGeom, paddedStrides)
  var localCoords: array[D, int]
  for d in 0..<D:
    localCoords[d] = paddedCoords[d] - s.ghostWidth[d]
    if localCoords[d] < 0 or localCoords[d] >= s.localGeom[d]:
      return -1
  coordsToLex(localCoords, localStrides)

# Standalone versions for when you don't have a stencil object
proc localToPadded*[D: static int](
  localSite: int,
  localGeom: array[D, int],
  ghostWidth: array[D, int]
): int =
  let localStrides = computeStrides(localGeom)
  let localCoords = lexToCoords(localSite, localGeom, localStrides)
  var paddedGeom: array[D, int]
  for d in 0..<D:
    paddedGeom[d] = localGeom[d] + 2 * ghostWidth[d]
  var paddedCoords: array[D, int]
  for d in 0..<D:
    paddedCoords[d] = localCoords[d] + ghostWidth[d]
  let paddedStrides = computeStrides(paddedGeom)
  coordsToLex(paddedCoords, paddedStrides)

proc paddedToLocal*[D: static int](
  paddedSite: int,
  localGeom: array[D, int],
  ghostWidth: array[D, int]
): int =
  var paddedGeom: array[D, int]
  for d in 0..<D:
    paddedGeom[d] = localGeom[d] + 2 * ghostWidth[d]
  let paddedStrides = computeStrides(paddedGeom)
  let paddedCoords = lexToCoords(paddedSite, paddedGeom, paddedStrides)
  var localCoords: array[D, int]
  for d in 0..<D:
    localCoords[d] = paddedCoords[d] - ghostWidth[d]
    if localCoords[d] < 0 or localCoords[d] >= localGeom[d]:
      return -1
  let localStrides = computeStrides(localGeom)
  coordsToLex(localCoords, localStrides)

#[ ============================================================================
   Device Buffer Support (for GPU backends)
   ============================================================================ ]#

proc getOffsetBuffer*[D: static int](s: LatticeStencil[D]): ptr int32 {.inline.} =
  if s.offsets.len > 0:
    unsafeAddr s.offsets[0]
  else:
    nil

proc offsetBufferSize*[D: static int](s: LatticeStencil[D]): int {.inline.} =
  s.nLocalSites * s.nPoints * sizeof(int32)

#[ ============================================================================
   Backward Compatibility - Legacy Types (deprecated)
   ============================================================================ ]#

# These types are kept for backward compatibility but should not be used
# in new code. Use LatticeStencil instead.

type
  StencilEntry* = object
    offset*: int
    isLocal*: bool
    permute*: uint8
    wrapAround*: bool

  StencilView*[D: static int] = object
    stencil*: StencilPattern[D]
    localGeom*: array[D, int]
    nSites*: int
    nPoints*: int
    entries*: seq[StencilEntry]

proc newStencilView*[D: static int](stencil: StencilPattern[D], localGeom: array[D, int]): StencilView[D] =
  ## Legacy constructor - creates a StencilView (use LatticeStencil instead)
  result.stencil = stencil
  result.localGeom = localGeom
  result.nPoints = stencil.nPoints
  result.nSites = 1
  for d in 0..<D:
    result.nSites *= localGeom[d]
  
  let strides = computeStrides(localGeom)
  result.entries = newSeq[StencilEntry](result.nSites * result.nPoints)
  
  for site in 0..<result.nSites:
    let siteCoords = lexToCoords(site, localGeom, strides)
    for pointIdx, point in stencil.points:
      var entry: StencilEntry
      entry.isLocal = true
      entry.permute = 0
      entry.wrapAround = false
      var nbrCoords: array[D, int]
      for d in 0..<D:
        nbrCoords[d] = siteCoords[d] + point.offset[d]
        if nbrCoords[d] < 0:
          nbrCoords[d] += localGeom[d]
          entry.wrapAround = true
        elif nbrCoords[d] >= localGeom[d]:
          nbrCoords[d] -= localGeom[d]
          entry.wrapAround = true
      entry.offset = coordsToLex(nbrCoords, strides)
      result.entries[site * result.nPoints + pointIdx] = entry

proc getEntry*[D: static int](sv: StencilView[D], site, point: int): StencilEntry {.inline.} =
  sv.entries[site * sv.nPoints + point]

proc neighborOffset*[D: static int](sv: StencilView[D], site, point: int): int {.inline.} =
  sv.entries[site * sv.nPoints + point].offset

#[ ============================================================================
   Tests
   ============================================================================ ]#

when isMainModule:
  import std/unittest
  from lattice/simplecubiclattice import SimpleCubicLattice, newSimpleCubicLattice

  suite "Direction Types":
    test "Direction creation":
      check X.int == 0
      check Y.int == 1
      check Z.int == 2
      check T.int == 3
    
    test "SignedDirection":
      let fwdX = forward(X)
      let bwdT = backward(T)
      check fwdX.sign == +1
      check bwdT.sign == -1

  suite "StencilPattern":
    test "Nearest neighbor 4D":
      let s = nearestNeighborStencil[4]()
      check s.nPoints == 8

    test "Forward stencil":
      let s = forwardStencil[4]()
      check s.nPoints == 4

    test "Lattice-inferred nearest neighbor":
      let lat = newSimpleCubicLattice([8, 8, 8, 16])
      let s = nearestNeighborStencil(lat)
      check s.nPoints == 8
      check s.name == "nearest-neighbor"

    test "Lattice-inferred laplacian":
      let lat = newSimpleCubicLattice([4, 4, 4, 4])
      let s = laplacianStencil(lat)
      check s.nPoints == 8
      check s.name == "laplacian"

    test "Lattice-inferred forward/backward":
      let lat = newSimpleCubicLattice([8, 8, 8, 16])
      let fwd = forwardStencil(lat)
      let bwd = backwardStencil(lat)
      check fwd.nPoints == 4
      check bwd.nPoints == 4

    test "Lattice-inferred 2D stencil":
      let lat2D = newSimpleCubicLattice([8, 8])
      let s = nearestNeighborStencil(lat2D)
      check s.nPoints == 4  # ±x, ±y

    test "Lattice-inferred 3D stencil":
      let lat3D = newSimpleCubicLattice([4, 4, 4])
      let s = nearestNeighborStencil(lat3D)
      check s.nPoints == 6  # ±x, ±y, ±z

  suite "LatticeStencil - Unified API":
    test "Create from local geometry":
      let pattern = nearestNeighborStencil[4]()
      let stencil = newLatticeStencil(pattern, [4, 4, 4, 4])
      check stencil.nSites == 256
      check stencil.nPoints == 8

    test "Create with ghost width":
      let pattern = nearestNeighborStencil[2]()
      let stencil = newLatticeStencil(pattern, [4, 4], [1, 1])
      check stencil.nLocalSites == 16
      check stencil.nPaddedSites == 36
      check stencil.paddedGeom == [6, 6]

    test "neighbor() returns valid indices":
      let pattern = nearestNeighborStencil[2]()
      let stencil = newLatticeStencil(pattern, [4, 4], [1, 1])
      for site in 0..<stencil.nSites:
        for p in 0..<stencil.nPoints:
          let nbr = stencil.neighbor(site, p)
          check nbr >= 0 and nbr < stencil.nPaddedSites

    test "isGhostNeighbor detection":
      let pattern = nearestNeighborStencil[2]()
      let stencil = newLatticeStencil(pattern, [4, 4], [1, 1])
      # Corner site should have ghost neighbors
      var foundGhost = false
      for p in 0..<stencil.nPoints:
        if stencil.isGhostNeighbor(0, p):
          foundGhost = true
      check foundGhost

    test "shift() API":
      let pattern = nearestNeighborStencil[4]()
      let stencil = newLatticeStencil(pattern, [4, 4, 4, 4])
      for site in stencil.sites:
        let fwdX = stencil.shift(site, +1, 0)
        let bwdT = stencil.shift(site, -1, 3)
        check fwdX.idx >= 0 and fwdX.idx < stencil.nPaddedSites
        check bwdT.idx >= 0 and bwdT.idx < stencil.nPaddedSites

    test "fwd/bwd shortcuts":
      let pattern = nearestNeighborStencil[4]()
      let stencil = newLatticeStencil(pattern, [4, 4, 4, 4])
      let site = 10
      check stencil.fwd(site, 0).idx == stencil.shift(site, +1, 0).idx
      check stencil.bwd(site, 2).idx == stencil.shift(site, -1, 2).idx

    test "Create from Lattice":
      let lat = newSimpleCubicLattice([8, 8, 8, 16], [1, 1, 1, 2])
      let pattern = nearestNeighborStencil[4]()
      let stencil = newLatticeStencil(pattern, lat)
      check stencil.localGeom == [8, 8, 8, 8]
      check stencil.nSites == 4096

    test "Create from Lattice (inferred pattern)":
      ## Use the Lattice-only constructor — D and pattern are inferred
      let lat = newSimpleCubicLattice([8, 8, 8, 16], [1, 1, 1, 2])
      let stencil = newLatticeStencil(lat)
      check stencil.localGeom == [8, 8, 8, 8]
      check stencil.nSites == 4096
      check stencil.nPoints == 8  # nearest-neighbor

    test "Create from Lattice with inferred pattern":
      ## Use nearestNeighborStencil(lat) instead of nearestNeighborStencil[4]()
      let lat = newSimpleCubicLattice([8, 8, 8, 16], [1, 1, 1, 2])
      let stencil = newLatticeStencil(nearestNeighborStencil(lat), lat)
      check stencil.nSites == 4096
      check stencil.nPoints == 8

    test "Lattice with ghost regions":
      let lat = newSimpleCubicLattice([8, 8], [2, 2], [1, 1])
      let pattern = nearestNeighborStencil[2]()
      let stencil = newLatticeStencil(pattern, lat)
      check stencil.localGeom == [4, 4]
      check stencil.ghostWidth == [1, 1]
      check stencil.paddedGeom == [6, 6]

  suite "Coordinate Conversion":
    test "localToPadded":
      let pattern = nearestNeighborStencil[2]()
      let stencil = newLatticeStencil(pattern, [4, 4], [1, 1])
      # Site 0 at (0,0) -> padded (1,1) = 1 + 1*6 = 7
      check stencil.localToPadded(0) == 7

    test "paddedToLocal":
      let pattern = nearestNeighborStencil[2]()
      let stencil = newLatticeStencil(pattern, [4, 4], [1, 1])
      check stencil.paddedToLocal(7) == 0
      check stencil.paddedToLocal(0) == -1  # Ghost

    test "Round-trip":
      let pattern = nearestNeighborStencil[2]()
      let stencil = newLatticeStencil(pattern, [4, 4], [1, 1])
      for localSite in 0..<16:
        let padded = stencil.localToPadded(localSite)
        let back = stencil.paddedToLocal(padded)
        check back == localSite

  suite "Iteration":
    test "sites iterator":
      let pattern = nearestNeighborStencil[2]()
      let stencil = newLatticeStencil(pattern, [4, 4])
      var count = 0
      for site in stencil.sites:
        count.inc
      check count == 16

    test "neighbors iterator":
      let pattern = nearestNeighborStencil[2]()
      let stencil = newLatticeStencil(pattern, [4, 4])
      for site in stencil.sites:
        var nbrCount = 0
        for nbr in stencil.neighbors(site):
          nbrCount.inc
        check nbrCount == 4

    test "forEachNeighbor":
      let pattern = nearestNeighborStencil[2]()
      let stencil = newLatticeStencil(pattern, [4, 4])
      var total = 0
      for site in stencil.sites:
        stencil.forEachNeighbor(site, p, nbr):
          total.inc
      check total == 16 * 4

  suite "Path-based Stencils":
    test "Plaquette path":
      let path = plaquettePath(0, 1)
      check path.len == 4
      
    test "Path to stencil":
      let lat = newSimpleCubicLattice([8, 8, 8, 16], [1, 1, 1, 1])
      let path = plaquettePath(0, 1)
      let s = pathToStencil(path, lat)
      check s.nPoints == 5  # Origin + 4 steps

  suite "Legacy Compatibility":
    test "StencilView still works":
      let s = nearestNeighborStencil[2]()
      let sv = newStencilView(s, [4, 4])
      check sv.nSites == 16
      check sv.nPoints == 4

    test "StencilView neighbor matches LatticeStencil (no ghosts)":
      let pattern = nearestNeighborStencil[2]()
      let sv = newStencilView(pattern, [4, 4])
      let stencil = newLatticeStencil(pattern, [4, 4])
      for site in 0..<16:
        for p in 0..<4:
          check sv.neighborOffset(site, p) == stencil.neighbor(site, p)
