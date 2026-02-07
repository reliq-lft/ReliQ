#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/types/globaltensor.nim
  Contact: reliq-lft@proton.me

  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

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

## GlobalTensor - Distributed Tensor Fields via Global Arrays
## =============================================================
##
## This module provides `TensorField[D,R,L,T]`, the primary distributed
## data type in ReliQ.  Each tensor field is backed by a
## `Global Array <https://globalarrays.github.io/>`_ with ghost (halo)
## regions, enabling transparent MPI communication of boundary data.
##
## Key capabilities:
##
## - **Construction**: `newTensorField` creates a GA-backed field with
##   automatic MPI decomposition and ghost region allocation
## - **Local data access**: `accessLocal`, `accessGhosts`, `releaseLocal`
##   provide raw pointers into the local partition (with or without ghost cells)
## - **Ghost exchange**: `updateGhosts`, `updateAllGhosts` synchronise
##   boundary data between MPI ranks via `GA_Update_ghost_dir`
## - **Coordinate utilities**: `lexToCoords`, `coordsToLex`, `localIdx`,
##   `localLexIdx` handle the C row-major memory layout used by GA
## - **Shifting / transport**: `GlobalShifter` performs distributed shifts
##   (``dest[x] = src[x + e_mu]``) using ghost exchange, with correct
##   handling of periodic boundary conditions for both distributed and
##   non-distributed dimensions
## - **Stencil operations**: `applyStencilShift`, `discreteLaplacian`
##   implement common nearest-neighbour operations
## - **Stencil integration**: `newLatticeStencil` creates a unified
##   `LatticeStencil[D]` from a tensor field's geometry
##
## GA Memory Layout
## ^^^^^^^^^^^^^^^^
##
## Global Arrays store data in **C row-major** order (last dimension
## fastest varying).  A `TensorField[D=4, R=2, T=float64]` with
## ``shape = [3, 3]`` maps to a 7-dimensional GA
## ``[Lx, Ly, Lz, Lt, S0, S1, Cplx]``.  The first ``D`` dimensions
## (lattice) carry ghost regions for boundary communication; the last
## ``R+1`` dimensions (tensor shape + complex component) also carry
## ghost width 1 because GA 5.8.2 requires **all** dimensions to have
## ``ghost ≥ 1`` for ``GA_Update_ghost_dir`` to function.  The padded
## inner block therefore has ``(S_i + 2)`` entries per tensor dimension
## and ``(complexFactor + 2)`` for the complex dimension; only the
## central ``product(shape) * complexFactor`` elements are real data.
## Use ``innerPaddedBlockSize`` / ``innerPaddedOffset`` to navigate.
##
## Two pointer types are available:
##
## - **Local pointer** (`accessLocal`): starts at the center of the
##   padded inner block for the first local lattice site.  ``p[0]``
##   is element 0 at local coordinates ``(0,0,...,0)``.  The stride
##   between adjacent lattice sites is ``innerPaddedBlockSize`` (the
##   product of all padded inner dimensions), **not** the number of
##   real elements.  Use `localIdx` to compute the correct flat index.
## - **Ghost pointer** (`accessGhosts`): starts at the origin of the
##   full padded array.  Use `coordsToPaddedFlat` to index into the
##   ghost-padded lattice dimensions; at each site, add
##   ``innerPaddedOffset`` to reach the first real element.
##
## Example
## ^^^^^^^
##
## .. code-block:: nim
##   import reliq
##
##   parallel:
##     let lat = newSimpleCubicLattice([8, 8, 8, 16], [1, 1, 1, 4], [1, 1, 1, 1])
##     var src  = lat.newTensorField([1, 1]): float64
##     var dest = lat.newTensorField([1, 1]): float64
##
##     # Fill src with global lex index
##     let p = src.accessLocal()
##     for site in 0..<nLocal:
##       p[src.localLexIdx(site)] = float64(site)
##     src.releaseLocal()
##
##     # Shift forward in t-direction
##     let shifter = newGlobalShifter(src, dim = 3, len = 1)
##     shifter.apply(src, dest)   # dest[x] = src[x + e_t]
##
##     # Discrete Laplacian
##     var lap = lat.newTensorField([1, 1]): float64
##     var tmp = lat.newTensorField([1, 1]): float64
##     discreteLaplacian(src, lap, tmp)

import lattice

import globalarrays/[gatypes, gawrap]
import utils/[complex]

when isMainModule:
  import std/[unittest, math]
  import globalarrays/[gampi, gabase]
  import utils/[commandline]
  from lattice/simplecubiclattice import SimpleCubicLattice

type TensorField*[D: static[int], R: static[int], L: Lattice[D], T] = object
  ## Distributed tensor field backed by a Global Array.
  ##
  ## Represents a rank-``R`` tensor field defined on a ``D``-dimensional
  ## lattice, stored as a distributed GA with ghost regions for MPI
  ## communication.  The element type ``T`` may be a real scalar or a
  ## complex type (`Complex32`, `Complex64`).
  lattice*: L
  shape*: array[R, int]

  when D + R == 1:
    when isComplex32(T): data*: GlobalArray[2, float32]
    elif isComplex64(T): data*: GlobalArray[2, float64]
    else: data*: GlobalArray[2, T]
  elif D + R == 2:
    when isComplex32(T): data*: GlobalArray[3, float32]
    elif isComplex64(T): data*: GlobalArray[3, float64]
    else: data*: GlobalArray[3, T]
  elif D + R == 3:
    when isComplex32(T): data*: GlobalArray[4, float32]
    elif isComplex64(T): data*: GlobalArray[4, float64]
    else: data*: GlobalArray[4, T]
  elif D + R == 4:
    when isComplex32(T): data*: GlobalArray[5, float32]
    elif isComplex64(T): data*: GlobalArray[5, float64]
    else: data*: GlobalArray[5, T]
  elif D + R == 5:
    when isComplex32(T): data*: GlobalArray[6, float32]
    elif isComplex64(T): data*: GlobalArray[6, float64]
    else: data*: GlobalArray[6, T]
  elif D + R == 6:
    when isComplex32(T): data*: GlobalArray[7, float32]
    elif isComplex64(T): data*: GlobalArray[7, float64]
    else: data*: GlobalArray[7, T]
  elif D + R == 7:
    when isComplex32(T): data*: GlobalArray[8, float32]
    elif isComplex64(T): data*: GlobalArray[8, float64]
    else: data*: GlobalArray[8, T]
  elif D + R == 8:
    when isComplex32(T): data*: GlobalArray[9, float32]
    elif isComplex64(T): data*: GlobalArray[9, float64]
    else: data*: GlobalArray[9, T]
  elif D + R == 9:
    when isComplex32(T): data*: GlobalArray[10, float32]
    elif isComplex64(T): data*: GlobalArray[10, float64]
    else: data*: GlobalArray[10, T]
  elif D + R == 10:
    when isComplex32(T): data*: GlobalArray[11, float32]
    elif isComplex64(T): data*: GlobalArray[11, float64]
    else: data*: GlobalArray[11, T]

#[ constructor ]#

proc newTensorField*[D: static[int], R: static[int], L: Lattice[D]](
  lattice: L,
  shape: array[R, int],
  T: typedesc
): TensorField[D, R, L, T] =
  ## Create a new TensorField
  ##
  ## Parameters:
  ## - `lattice`: The lattice on which the tensor field is defined
  ## - `shape`: The shape of the tensor field
  ##
  ## Returns:
  ## A new TensorField instance
  ##
  ## Example:
  ## ```nim
  ## let lattice = newSimpleCubicLattice([16, 16, 16, 16])
  ## let TensorField = lattice.newTensorField([3, 3]): float64
  ## ```
  const rank = D + R + 1 # +1 for tensor component type (1D for real, 2D for complex)
  var globalGrid: array[rank, int]
  var mpiGrid: array[rank, int]
  var ghostGrid: array[rank, int]
  
  result.lattice = lattice
  result.shape = shape

  # lattice grid
  for i in 0..<D:
    globalGrid[i] = lattice.globalGrid[i]
    mpiGrid[i] = lattice.mpiGrid[i]
    ghostGrid[i] = lattice.ghostGrid[i]

  # tensor grid
  # Inner dimensions (tensor shape + complex) are NOT distributed and NOT
  # exchanged during ghost updates.  However, GA requires ghost width ≥ 1
  # on ALL dimensions for GA_Update_ghost_dir to function correctly (GA
  # cannot handle zero-width ghost regions in any dimension).  Setting
  # ghost = 1 here satisfies GA; the inner padding is accounted for in
  # `localIdx`, `coordsToPaddedFlat`, and the GlobalShifter / Laplacian.
  # LocalTensorField and TensorFieldView work on separate contiguous
  # buffers that are de-padded from the GA memory.
  for i in 0..<R:
    globalGrid[D + i] = shape[i]
    mpiGrid[D + i] = 1
    ghostGrid[D + i] = 1
  
  # tensor component grid (extra dimension)
  when not isComplex(T): globalGrid[^1] = 1
  else: globalGrid[^1] = 2
  mpiGrid[^1] = 1
  ghostGrid[^1] = 1
  
  # initialize global array
  when not isComplex(T):
    result.data = newGlobalArray(globalGrid, mpiGrid, ghostGrid): T
  elif isComplex32(T):
    result.data = newGlobalArray(globalGrid, mpiGrid, ghostGrid): float32
  elif isComplex64(T):
    result.data = newGlobalArray(globalGrid, mpiGrid, ghostGrid): float64

#[ ============================================================================
   Halo Exchange (Ghost Region Update) for Distributed Tensor Fields
   ============================================================================ ]#

proc innerBlockSize*(R: static int, shape: openArray[int], isComplex: bool): int =
  ## Compute the number of contiguous elements per lattice site
  ##
  ## This is the product of all tensor shape entries times the complex factor.
  result = 1
  for r in 0..<R:
    result *= shape[r]
  if isComplex: result *= 2

proc innerPaddedBlockSize*(R: static int, ghostWidth: int): int =
  ## Compute the padded inner block size for ghost-padded arrays
  ##
  ## Each inner dimension of size S gets padded to ``S + 2*ghostWidth``.
  ## The block size is the product of all padded inner dimensions.
  ## For scalar real with S=1, ghost=1: each of R+1 dims is padded to 3,
  ## so block = 3^(R+1).
  result = 1
  for _ in 0..R:  # R+1 inner dimensions (tensor dims + complex dim)
    result *= (1 + 2 * ghostWidth)

proc innerPaddedOffset*(R: static int, ghostWidth: int): int =
  ## Compute the flat offset to the center element of the padded inner block
  ##
  ## In the ghost-padded inner block, each inner dimension of padded size P
  ## has the real data at index ``ghostWidth``.  The offset from the start
  ## of the padded block to the center (first real element) is:
  ## ``sum_{i=0..R} ghostWidth * stride_i``
  result = 0
  let P = 1 + 2 * ghostWidth
  var stride = 1
  for _ in countdown(R, 0):
    result += ghostWidth * stride
    stride *= P

proc updateGhosts*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: TensorField[D, R, L, T],
  dim: int,
  direction: int = 0,
  updateCorners: bool = false
) =
  ## Update ghost regions for a tensor field in specified dimension
  ##
  ## This synchronizes ghost (halo) regions between MPI ranks.
  ## Must be called before reading from ghost regions after local updates.
  ##
  ## Parameters:
  ##   tensor: The tensor field to update
  ##   dim: Dimension to update (0..D-1)
  ##   direction: +1 for forward, -1 for backward, 0 for both (default)
  ##   updateCorners: Whether to update corner ghost cells
  ##
  ## Example:
  ## ```nim
  ## var field = lat.newTensorField([3, 3]): Complex64
  ## # ... modify local data ...
  ## field.updateGhosts(0)  # Update ghosts in x direction
  ## field.updateGhosts(1)  # Update ghosts in y direction
  ## # Now ghost regions contain correct neighbor data
  ## ```
  let handle = tensor.data.getHandle()
  let updateCornersFlag: cint = if updateCorners: 1 else: 0
  
  # Skip dimensions with ghost width 0 — GA_Update_ghost_dir fails
  # when the target dimension has no ghost cells allocated.
  if tensor.lattice.ghostGrid[dim] == 0:
    return
  
  if direction == 0:
    # Update both directions
    handle.GA_Update_ghost_dir(cint(dim), cint(1), updateCornersFlag)
    handle.GA_Update_ghost_dir(cint(dim), cint(-1), updateCornersFlag)
  else:
    handle.GA_Update_ghost_dir(cint(dim), cint(direction), updateCornersFlag)

proc updateAllGhosts*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: TensorField[D, R, L, T],
  updateCorners: bool = false
) =
  ## Update ghost regions in all lattice dimensions
  ##
  ## Skips dimensions with ghost width 0 (including inner tensor/complex
  ## dimensions).  Uses ``GA_Update_ghost_dir`` for each lattice dimension.
  ##
  ## Example:
  ## ```nim
  ## var field = lat.newTensorField([3, 3]): Complex64
  ## # ... modify local data ...
  ## field.updateAllGhosts()  # Update all ghost regions
  ## ```
  for dim in 0..<D:
    tensor.updateGhosts(dim, 0, updateCorners)

proc hasGhosts*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: TensorField[D, R, L, T]
): bool =
  ## Check if tensor field has ghost regions configured
  for d in 0..<D:
    if tensor.lattice.ghostGrid[d] > 0:
      return true
  return false

proc ghostWidth*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: TensorField[D, R, L, T]
): array[D, int] =
  ## Get the ghost width in each dimension
  tensor.lattice.ghostGrid

proc localGrid*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: TensorField[D, R, L, T]
): array[D, int] =
  ## Get local grid dimensions (excluding ghosts)
  for d in 0..<D:
    let mpi = if tensor.lattice.mpiGrid[d] <= 0: 1 else: tensor.lattice.mpiGrid[d]
    result[d] = tensor.lattice.globalGrid[d] div mpi

proc paddedGrid*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: TensorField[D, R, L, T]
): array[D, int] =
  ## Get padded grid dimensions (including ghosts on both sides)
  let local = tensor.localGrid()
  let ghosts = tensor.ghostWidth()
  for d in 0..<D:
    result[d] = local[d] + 2 * ghosts[d]

#[ ============================================================================
   Stencil Integration - Create stencils that work with TensorFields
   ============================================================================ ]#

# Forward import for stencil types
import lattice/stencil

proc newLatticeStencil*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: TensorField[D, R, L, T],
  pattern: StencilPattern[D]
): LatticeStencil[D] =
  ## Create a unified lattice stencil from a tensor field
  ##
  ## Automatically extracts local geometry and ghost width from the tensor.
  ## The stencil understands both local and ghost regions.
  ##
  ## Example:
  ## ```nim
  ## let lat = newSimpleCubicLattice([16, 16, 16, 32], [1, 1, 1, 4], [1, 1, 1, 1])
  ## var field = lat.newTensorField([3, 3]): Complex64
  ## 
  ## let stencil = field.newLatticeStencil(nearestNeighborStencil[4]())
  ## 
  ## # Use stencil with views in each loop
  ## var local = field.newLocalTensorField()
  ## field.updateAllGhosts()
  ## 
  ## block:
  ##   var view = local.newTensorFieldView(iokRead)
  ##   for site in 0..<stencil.nSites:
  ##     for dir in 0..<4:
  ##       let fwd = view[stencil.fwd(site, dir)]
  ##       let bwd = view[stencil.bwd(site, dir)]
  ## ```
  newLatticeStencil(pattern, tensor.localGrid(), tensor.ghostWidth())

#[ ============================================================================
   Local Data Access for Distributed Tensor Fields
   ============================================================================ ]#

proc accessLocal*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: TensorField[D, R, L, T]
): ptr UncheckedArray[T] =
  ## Access local data segment of the tensor field (no ghosts)
  ##
  ## **Important**: The returned pointer shares the padded memory layout.
  ## The stride between consecutive lattice sites is NOT 1 — it is determined
  ## by the padded inner dimensions (tensor shape + complex, each ghost-padded).
  ## Use ``localIdx`` to compute the correct flat index for a lattice site.
  ##
  ## Must call ``releaseLocal`` when done.
  when not isComplex(T):
    let (p, _) = tensor.data.accessLocal()
    result = p
  elif isComplex32(T):
    let (p, _) = tensor.data.accessLocal()
    result = cast[ptr UncheckedArray[T]](p)
  elif isComplex64(T):
    let (p, _) = tensor.data.accessLocal()
    result = cast[ptr UncheckedArray[T]](p)

proc accessGhosts*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: TensorField[D, R, L, T]
): ptr UncheckedArray[T] =
  ## Access local data including ghost regions
  ##
  ## Returns a raw pointer to the padded data (local + ghost regions).
  ## Ghost regions must have been updated via ``updateAllGhosts`` first.
  ## Must call ``releaseLocal`` when done.
  when not isComplex(T):
    let (p, _, _) = tensor.data.accessGhosts()
    result = p
  elif isComplex32(T):
    let (p, _, _) = tensor.data.accessGhosts()
    result = cast[ptr UncheckedArray[T]](p)
  elif isComplex64(T):
    let (p, _, _) = tensor.data.accessGhosts()
    result = cast[ptr UncheckedArray[T]](p)

proc releaseLocal*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: TensorField[D, R, L, T]
) =
  ## Release local data access obtained via ``accessLocal`` or ``accessGhosts``
  tensor.data.releaseLocal()

#[ ============================================================================
   Coordinate Utilities for Distributed Tensor Fields
   ============================================================================ ]#

# Note on GA memory layout:
# GA's C API uses row-major (C-order) storage: the LAST dimension is the
# fastest varying in memory.  For a 7-dimensional GA with dims
# [d0, d1, d2, d3, d4, d5, d6], element (i0,i1,...,i6) is at flat index:
#   i0 * (d1*d2*d3*d4*d5*d6) + i1 * (d2*d3*d4*d5*d6) + ... + i6
#
# For a TensorField[D=4, R=2, T=float64] with shape [1,1], the GA has 7
# dimensions: [Lx, Ly, Lz, Lt, S0, S1, Cplx].
# The last R+1 dimensions are "inner" (tensor shape + complex component).
# For scalar real fields all inner dims are 1, so the inner stride factor is 1
# and flat lattice indexing works directly.
#
# For ghost-padded arrays the inner dims get padded too (ghost width ≥ 1 is
# required for ALL GA dimensions).  To index into the ghost-padded array for
# a lattice site we must:
#   1. Compute the inner offset once (center of the padded inner dims)
#   2. Multiply each lattice stride by the inner block size

proc lexToCoords*[D: static int](idx: int, geom: array[D, int]): array[D, int] =
  ## Convert lexicographic index to D-dimensional coordinates
  ##
  ## Uses GA C-order convention: last dimension is fastest varying.
  ## ``idx = coords[0] * stride0 + coords[1] * stride1 + ... + coords[D-1]``
  var remaining = idx
  for d in countdown(D-1, 0):
    result[d] = remaining mod geom[d]
    remaining = remaining div geom[d]

proc coordsToLex*[D: static int](coords: array[D, int], geom: array[D, int]): int =
  ## Convert D-dimensional coordinates to lexicographic index
  ##
  ## GA C-order: last dimension is fastest varying.
  result = 0
  var stride = 1
  for d in countdown(D-1, 0):
    result += coords[d] * stride
    stride *= geom[d]

proc coordsToPaddedFlat*[D: static int](
  coords: array[D, int],
  paddedGeom: array[D, int],
  ghostWidth: array[D, int],
  innerBlockSize: int
): int =
  ## Convert local lattice coordinates to a flat index in the ghost-padded array
  ##
  ## The padded array has D lattice dimensions (each padded by ``2*ghostWidth``)
  ## followed by R+1 inner dimensions (also padded). This function computes
  ## the index for the center element of the inner block at the given lattice site.
  ##
  ## Parameters:
  ## - coords: Local lattice coordinates (can be negative for backward ghost)
  ## - paddedGeom: Padded lattice geometry (``localGeom + 2*ghostWidth`` per dim)
  ## - ghostWidth: Ghost width per lattice dimension
  ## - innerBlockSize: Product of all padded inner dimensions
  ##
  ## Returns:
  ## Flat index into the padded data array (not including inner offset)
  result = 0
  var stride = innerBlockSize
  for d in countdown(D-1, 0):
    result += (coords[d] + ghostWidth[d]) * stride
    stride *= paddedGeom[d]

proc localSiteOffset*[D: static int](
  coords: array[D, int],
  paddedGeom: array[D, int],
  innerBlockSize: int
): int =
  ## Convert local lattice coordinates to a flat offset in the local data pointer
  ##
  ## The local pointer from NGA_Access points into the padded memory at the start
  ## of the local region. The memory layout still uses padded strides for the
  ## lattice dimensions (which carry ghost regions). The inner block at each
  ## lattice site is contiguous with ``innerBlockSize`` elements.
  ##
  ## Parameters:
  ## - coords: Local lattice coordinates
  ## - paddedGeom: Padded lattice geometry (``localGeom + 2*ghostWidth`` per dim)
  ## - innerBlockSize: Number of contiguous inner elements per lattice site
  ##
  ## Returns:
  ## Flat offset from local pointer p[0] to the start of the inner block
  ## for the given site.
  result = 0
  var stride = innerBlockSize
  for d in countdown(D-1, 0):
    result += coords[d] * stride
    stride *= paddedGeom[d]

proc localIdx*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: TensorField[D, R, L, T],
  coords: array[D, int]
): int =
  ## Compute the flat index into the local data pointer for a lattice site
  ##
  ## The GA inner dimensions (tensor shape + complex) are ghost-padded to
  ## satisfy GA's requirement that all dimensions have ghost width ≥ 1.
  ## ``NGA_Access`` returns a pointer to the center of the padded block
  ## for the first local lattice site, so ``p[0]`` is the first real
  ## element.  The stride between consecutive lattice sites is the product
  ## of all padded inner dimensions (``innerPaddedBlockSize``).
  ## 
  ## The returned index points to the center of the padded inner block
  ## for the given lattice site.  Add ``e`` to reach element ``e`` (but
  ## only ``0..<innerBlockSize`` are real data; the rest is padding).
  ##
  ## Parameters:
  ## - ``tensor``: The tensor field (used for shape and padding info)
  ## - ``coords``: Local lattice coordinates
  ##
  ## Returns:
  ## Flat index into the local data pointer for the first real element
  ## at the given lattice site.
  let paddedGeom = tensor.paddedGrid()
  let innerPadded = innerPaddedBlockSize(R, 1)
  localSiteOffset(coords, paddedGeom, innerPadded)

proc localLexIdx*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: TensorField[D, R, L, T],
  site: int
): int =
  ## Compute the flat index for a lexicographically-ordered site
  ##
  ## Converts a lexicographic site index to the flat index in the local
  ## data pointer, accounting for lattice-dimension ghost padding while
  ## keeping inner (tensor/complex) dimensions contiguous.
  let localGeom = tensor.localGrid()
  let coords = lexToCoords(site, localGeom)
  tensor.localIdx(coords)

#[ ============================================================================
   GlobalTensor Shifter — Distributed-Memory Transport via GA Ghosts
   ============================================================================ ]#

type
  GlobalShifter*[D: static int, R: static int, L: Lattice[D], T] = object
    ## Distributed-memory field shifter backed by Global Arrays ghost exchange
    ##
    ## Unlike the single-rank ``Shifter[D,T]`` in transporter.nim, a
    ## ``GlobalShifter`` performs real MPI communication via
    ## ``GA_Update_ghost_dir`` so that neighbor lookups near partition
    ## boundaries read correct remote data.
    ##
    ## Usage:
    ## ```nim
    ## let lat = newSimpleCubicLattice([8,8,8,16], [1,1,1,4], [1,1,1,1])
    ## var field = lat.newTensorField([1,1]): float64
    ## let shifter = newGlobalShifter(field, dim=3, len=1)
    ## var dest   = lat.newTensorField([1,1]): float64
    ## shifter.apply(field, dest)   # dest[x] = field[x + e_3]
    ## ```
    dim*: int              ## Shift dimension
    len*: int              ## Displacement (+forward, -backward)
    localGeom: array[D, int]
    ghostWidth: array[D, int]
    paddedGeom: array[D, int]
    mpiGrid: array[D, int] ## MPI decomposition grid
    nLocalSites: int       ## Product of localGeom[]
    tensorElements: int    ## Product of shape[]
    
proc newGlobalShifter*[D: static[int], R: static[int], L: Lattice[D], T](
  field: TensorField[D, R, L, T],
  dim: int,
  len: int = 1
): GlobalShifter[D, R, L, T] =
  ## Create a global shifter for a tensor field in a given dimension
  ##
  ## Parameters:
  ## - ``field``: The tensor field (used only for geometry, not mutated)
  ## - ``dim``: Dimension to shift along (0..D-1)
  ## - ``len``: Displacement (+1 = forward, -1 = backward)
  result.dim = dim
  result.len = len
  result.localGeom = field.localGrid()
  result.ghostWidth = field.ghostWidth()
  result.mpiGrid = field.lattice.mpiGrid
  for d in 0..<D:
    result.paddedGeom[d] = result.localGeom[d] + 2 * result.ghostWidth[d]
  result.nLocalSites = 1
  for d in 0..<D:
    result.nLocalSites *= result.localGeom[d]
  result.tensorElements = 1
  for r in 0..<R:
    result.tensorElements *= field.shape[r]

proc apply*[D: static[int], R: static[int], L: Lattice[D], T](
  shifter: GlobalShifter[D, R, L, T],
  src: TensorField[D, R, L, T],
  dest: var TensorField[D, R, L, T]
) =
  ## Apply the shift: ``dest[x] = src[x + shift]``
  ##
  ## Performs GA ghost exchange in the shift dimension, then reads
  ## neighbor values from the ghost-padded local data.
  
  # 1. Update ghosts — update all dimensions/directions to ensure periodic BC
  # work correctly even for single-rank dimensions
  src.updateAllGhosts()
  
  # 2. Access ghost-padded source data
  let srcPtr = src.accessGhosts()
  
  # 3. Access destination local data (pointer shares padded memory layout)
  let destPtr = dest.accessLocal()
  
  # 4. Compute inner block metrics — inner dims have ghost width 1
  #    (GA requires ghost ≥ 1 on ALL dimensions)
  let innerGW = 1  # ghost width on inner (tensor + complex) dimensions
  let innerPadded = innerPaddedBlockSize(R, innerGW)
  let innerOff = innerPaddedOffset(R, innerGW)
  let innerLocal = innerBlockSize(R, src.shape, isComplex(T))
  
  # 5. For each local site, read from the shifted position in the padded array
  for site in 0..<shifter.nLocalSites:
    let coords = lexToCoords(site, shifter.localGeom)
    
    # Source: shifted coordinate in the ghost-padded array
    var nbrCoords: array[D, int]
    for d in 0..<D:
      nbrCoords[d] = coords[d]
    nbrCoords[shifter.dim] += shifter.len
    
    # Wrap locally for non-distributed dimensions (only 1 MPI rank in that dim).
    # GA ghost exchange only fills ghost cells for dimensions with >1 MPI rank.
    # For single-rank dims, all data is local so we wrap coordinates directly.
    if shifter.mpiGrid[shifter.dim] == 1:
      if nbrCoords[shifter.dim] >= shifter.localGeom[shifter.dim]:
        nbrCoords[shifter.dim] -= shifter.localGeom[shifter.dim]
      elif nbrCoords[shifter.dim] < 0:
        nbrCoords[shifter.dim] += shifter.localGeom[shifter.dim]
    
    # Source index in the ghost-padded array; add innerOff to skip inner ghost
    let srcIdx = coordsToPaddedFlat(nbrCoords, shifter.paddedGeom, 
                                     shifter.ghostWidth, innerPadded) + innerOff
    
    # Destination index in the local pointer (padded inner stride)
    let destIdx = localSiteOffset(coords, shifter.paddedGeom, innerPadded)
    
    # Copy only real tensor elements (skip inner ghost padding)
    for e in 0..<innerLocal:
      destPtr[destIdx + e] = srcPtr[srcIdx + e]
  
  # 7. Release access
  src.releaseLocal()
  dest.releaseLocal()
  
  # 8. Sync to ensure all ranks complete
  GA_Sync()

proc newGlobalShifters*[D: static[int], R: static[int], L: Lattice[D], T](
  field: TensorField[D, R, L, T],
  len: int = 1
): array[D, GlobalShifter[D, R, L, T]] =
  ## Create forward shifters for all D dimensions
  for d in 0..<D:
    result[d] = newGlobalShifter(field, d, len)

proc newGlobalBackwardShifters*[D: static[int], R: static[int], L: Lattice[D], T](
  field: TensorField[D, R, L, T],
  len: int = 1
): array[D, GlobalShifter[D, R, L, T]] =
  ## Create backward shifters for all D dimensions
  for d in 0..<D:
    result[d] = newGlobalShifter(field, d, -len)

#[ ============================================================================
   Stencil-Based Nearest-Neighbor Operations for Distributed Tensor Fields
   ============================================================================ ]#

proc applyStencilShift*[D: static[int], R: static[int], L: Lattice[D], T](
  src: TensorField[D, R, L, T],
  dest: var TensorField[D, R, L, T],
  dim: int,
  direction: int
) =
  ## Apply a single stencil shift using GA ghost exchange
  ##
  ## A convenience wrapper: ``dest[x] = src[x + direction*e_dim]``
  ##
  ## Parameters:
  ## - ``src``: Source tensor field
  ## - ``dest``: Destination tensor field (overwritten)
  ## - ``dim``: Shift dimension (0..D-1)
  ## - ``direction``: +1 for forward, -1 for backward
  let shifter = newGlobalShifter(src, dim, direction)
  shifter.apply(src, dest)

proc discreteLaplacian*[D: static[int], R: static[int], L: Lattice[D], T](
  src: TensorField[D, R, L, T],
  dest: var TensorField[D, R, L, T],
  scratch: var TensorField[D, R, L, T]
) =
  ## Compute the discrete Laplacian: ``dest[x] = sum_mu (src[x+mu] + src[x-mu]) - 2*D * src[x]``
  ##
  ## Uses GA ghost exchange for boundary communication.
  ## ``scratch`` is used as temporary storage.
  
  # Update all ghosts at once for efficiency
  src.updateAllGhosts()
  
  let srcGhost = src.accessGhosts()
  let destLocal = dest.accessLocal()
  
  let localGeom = src.localGrid()
  let ghostWidth = src.ghostWidth()
  let mpiGrid = src.lattice.mpiGrid
  var paddedGeom: array[D, int]
  for d in 0..<D:
    paddedGeom[d] = localGeom[d] + 2 * ghostWidth[d]
  
  var nLocalSites = 1
  for d in 0..<D: nLocalSites *= localGeom[d]
  
  # Inner block metrics — inner dims have ghost width 1 (GA requirement)
  let innerGW = 1
  let innerPadded = innerPaddedBlockSize(R, innerGW)
  let innerOff = innerPaddedOffset(R, innerGW)
  let innerLocal = innerBlockSize(R, src.shape, isComplex(T))
  
  # Zero destination
  for site in 0..<nLocalSites:
    let coords = lexToCoords(site, localGeom)
    let destBase = localSiteOffset(coords, paddedGeom, innerPadded)
    for e in 0..<innerLocal:
      destLocal[destBase + e] = T(0)
  
  # Accumulate neighbor contributions
  for site in 0..<nLocalSites:
    let coords = lexToCoords(site, localGeom)
    let destIdx = localSiteOffset(coords, paddedGeom, innerPadded)
    let srcCenterIdx = coordsToPaddedFlat(coords, paddedGeom, ghostWidth, innerPadded) + innerOff
    
    for dim in 0..<D:
      # Forward neighbor — wrap locally only for non-distributed dims
      var fwdCoords = coords
      fwdCoords[dim] += 1
      if mpiGrid[dim] == 1 and fwdCoords[dim] >= localGeom[dim]:
        fwdCoords[dim] -= localGeom[dim]
      let fwdIdx = coordsToPaddedFlat(fwdCoords, paddedGeom, ghostWidth, innerPadded) + innerOff
      
      # Backward neighbor — wrap locally only for non-distributed dims
      var bwdCoords = coords
      bwdCoords[dim] -= 1
      if mpiGrid[dim] == 1 and bwdCoords[dim] < 0:
        bwdCoords[dim] += localGeom[dim]
      let bwdIdx = coordsToPaddedFlat(bwdCoords, paddedGeom, ghostWidth, innerPadded) + innerOff
      
      for e in 0..<innerLocal:
        destLocal[destIdx + e] += 
          srcGhost[fwdIdx + e] +
          srcGhost[bwdIdx + e]
    
    # Subtract 2*D * center value
    for e in 0..<innerLocal:
      destLocal[destIdx + e] -= T(2 * D) * srcGhost[srcCenterIdx + e]
  
  src.releaseLocal()
  dest.releaseLocal()
  GA_Sync()

when isMainModule:
  block:
    var argc = cargc()
    var argv = cargv(argc)
    
    initMPI(addr argc, addr argv)
    initGA()
    
    let nranks = int(GA_Nnodes())
    let myrank = int(GA_Nodeid())
    
    # Scope all GA operations so destructors run before finalizeGA()
    block:
      # ====================================================================
      #  Test lattice: 8^3 x 16, distributed along t-dimension
      # ====================================================================
      let dims: array[4, int] = [8, 8, 8, 8 * nranks]
      let mpi: array[4, int]  = [1, 1, 1, nranks]
      let ghosts: array[4, int] = [1, 1, 1, 1]
      let lattice = newSimpleCubicLattice(dims, mpi, ghosts)

      suite "GlobalTensor Construction":
        test "Scalar field construction":
          var field = lattice.newTensorField([1, 1]): float64
          check field.localGrid() == [8, 8, 8, 8]
          check field.ghostWidth() == ghosts
          check field.paddedGrid() == [10, 10, 10, 10]
          check field.hasGhosts()

        test "Matrix field construction":
          var field = lattice.newTensorField([3, 3]): float64
          check field.shape == [3, 3]

      suite "GlobalTensor Local Data Access":
        test "Write and read local data":
          var field = lattice.newTensorField([1, 1]): float64
          let localGeom = field.localGrid()
          var nLocal = 1
          for d in 0..<4: nLocal *= localGeom[d]

          # Write: field[x] = globalLexIndex(x) using proper padded-stride indexing
          let p = field.accessLocal()
          let tOffset = myrank * localGeom[3]
          for site in 0..<nLocal:
            let coords = lexToCoords(site, localGeom)
            var globalCoords: array[4, int]
            for d in 0..<4: globalCoords[d] = coords[d]
            globalCoords[3] += tOffset
            let globalIdx = coordsToLex(globalCoords, dims)
            p[field.localIdx(coords)] = float64(globalIdx)
          field.releaseLocal()
          GA_Sync()

          # Read back and verify
          let q = field.accessLocal()
          for site in 0..<nLocal:
            let coords = lexToCoords(site, localGeom)
            var globalCoords: array[4, int]
            for d in 0..<4: globalCoords[d] = coords[d]
            globalCoords[3] += tOffset
            let expected = float64(coordsToLex(globalCoords, dims))
            check q[field.localIdx(coords)] == expected
          field.releaseLocal()

      suite "GlobalTensor Ghost Exchange":
        test "Ghost exchange preserves neighbor values":
          var field = lattice.newTensorField([1, 1]): float64
          let localGeom = field.localGrid()
          var nLocal = 1
          for d in 0..<4: nLocal *= localGeom[d]

          # Fill field with global lex index
          let tOffset = myrank * localGeom[3]
          let p = field.accessLocal()
          for site in 0..<nLocal:
            let coords = lexToCoords(site, localGeom)
            var gc: array[4, int]
            for d in 0..<4: gc[d] = coords[d]
            gc[3] += tOffset
            p[field.localIdx(coords)] = float64(coordsToLex(gc, dims))
          field.releaseLocal()
          GA_Sync()

          # Update ghosts
          field.updateAllGhosts()

          # Read ghost-padded data and check t-neighbors
          let padded = field.paddedGrid()
          let ghost = field.ghostWidth()
          let q = field.accessGhosts()
          
          # Inner block metrics — inner dims have ghost width 1 (GA requirement)
          let innerPadded = innerPaddedBlockSize(2, 1)
          let innerOff = innerPaddedOffset(2, 1)
          
          # Check a local site that has a forward t-neighbor in the ghost
          let testCoords = [0, 0, 0, localGeom[3] - 1]  # Last t-slice
          let paddedIdx = coordsToPaddedFlat(testCoords, padded, ghost, innerPadded) + innerOff
          # Verify center value is correct
          var expectedCenter: array[4, int] = [0, 0, 0, tOffset + localGeom[3] - 1]
          check q[paddedIdx] == float64(coordsToLex(expectedCenter, dims))
          # Its forward t-neighbor is in the ghost region
          var fwdCoords = testCoords
          fwdCoords[3] += 1
          let fwdPaddedIdx = coordsToPaddedFlat(fwdCoords, padded, ghost, innerPadded) + innerOff
          
          var expectedGlobalT = tOffset + localGeom[3]  # First t on next rank
          if expectedGlobalT >= dims[3]:
            expectedGlobalT = 0  # Periodic wrap
          var expectedGlobal: array[4, int] = [0, 0, 0, expectedGlobalT]
          let expectedVal = float64(coordsToLex(expectedGlobal, dims))
          check q[fwdPaddedIdx] == expectedVal
          
          field.releaseLocal()

      suite "GlobalShifter Forward Shift":
        test "Forward shift in t dimension":
          var src = lattice.newTensorField([1, 1]): float64
          var dest = lattice.newTensorField([1, 1]): float64
          let localGeom = src.localGrid()
          var nLocal = 1
          for d in 0..<4: nLocal *= localGeom[d]

          # Fill src with global lex index
          let tOffset = myrank * localGeom[3]
          let p = src.accessLocal()
          for site in 0..<nLocal:
            let coords = lexToCoords(site, localGeom)
            var gc: array[4, int]
            for d in 0..<4: gc[d] = coords[d]
            gc[3] += tOffset
            p[src.localIdx(coords)] = float64(coordsToLex(gc, dims))
          src.releaseLocal()
          GA_Sync()

          # Shift forward in t
          let shifter = newGlobalShifter(src, dim=3, len=1)
          shifter.apply(src, dest)

          # Verify: dest[x] = src[x + e_t] = globalIdx(x + e_t)
          let q = dest.accessLocal()
          for site in 0..<nLocal:
            let coords = lexToCoords(site, localGeom)
            var gc: array[4, int]
            for d in 0..<4: gc[d] = coords[d]
            gc[3] = (gc[3] + tOffset + 1) mod dims[3]
            let expected = float64(coordsToLex(gc, dims))
            check q[dest.localIdx(coords)] == expected
          dest.releaseLocal()

        test "Forward shift in x dimension (no MPI boundary)":
          var src = lattice.newTensorField([1, 1]): float64
          var dest = lattice.newTensorField([1, 1]): float64
          let localGeom = src.localGrid()
          var nLocal = 1
          for d in 0..<4: nLocal *= localGeom[d]

          let tOffset = myrank * localGeom[3]
          let p = src.accessLocal()
          for site in 0..<nLocal:
            let coords = lexToCoords(site, localGeom)
            var gc: array[4, int]
            for d in 0..<4: gc[d] = coords[d]
            gc[3] += tOffset
            p[src.localIdx(coords)] = float64(coordsToLex(gc, dims))
          src.releaseLocal()
          GA_Sync()

          # Shift forward in x (all ranks own full x extent, ghost wraps locally)
          let shifter = newGlobalShifter(src, dim=0, len=1)
          shifter.apply(src, dest)

          let q = dest.accessLocal()
          for site in 0..<nLocal:
            let coords = lexToCoords(site, localGeom)
            var gc: array[4, int]
            for d in 0..<4: gc[d] = coords[d]
            gc[0] = (gc[0] + 1) mod dims[0]
            gc[3] += tOffset
            let expected = float64(coordsToLex(gc, dims))
            check q[dest.localIdx(coords)] == expected
          dest.releaseLocal()

      suite "GlobalShifter Backward Shift":
        test "Backward shift in t dimension":
          var src = lattice.newTensorField([1, 1]): float64
          var dest = lattice.newTensorField([1, 1]): float64
          let localGeom = src.localGrid()
          var nLocal = 1
          for d in 0..<4: nLocal *= localGeom[d]

          let tOffset = myrank * localGeom[3]
          let p = src.accessLocal()
          for site in 0..<nLocal:
            let coords = lexToCoords(site, localGeom)
            var gc: array[4, int]
            for d in 0..<4: gc[d] = coords[d]
            gc[3] += tOffset
            p[src.localIdx(coords)] = float64(coordsToLex(gc, dims))
          src.releaseLocal()
          GA_Sync()

          let shifter = newGlobalShifter(src, dim=3, len= -1)
          shifter.apply(src, dest)

          let q = dest.accessLocal()
          for site in 0..<nLocal:
            let coords = lexToCoords(site, localGeom)
            var gc: array[4, int]
            for d in 0..<4: gc[d] = coords[d]
            gc[3] = (gc[3] + tOffset - 1 + dims[3]) mod dims[3]
            let expected = float64(coordsToLex(gc, dims))
            check q[dest.localIdx(coords)] == expected
          dest.releaseLocal()

      suite "GlobalShifter Identity Composition":
        test "Forward then backward = identity":
          var src = lattice.newTensorField([1, 1]): float64
          var tmp = lattice.newTensorField([1, 1]): float64
          var dest = lattice.newTensorField([1, 1]): float64
          let localGeom = src.localGrid()
          var nLocal = 1
          for d in 0..<4: nLocal *= localGeom[d]

          let tOffset = myrank * localGeom[3]
          let p = src.accessLocal()
          for site in 0..<nLocal:
            let coords = lexToCoords(site, localGeom)
            var gc: array[4, int]
            for d in 0..<4: gc[d] = coords[d]
            gc[3] += tOffset
            p[src.localIdx(coords)] = float64(coordsToLex(gc, dims))
          src.releaseLocal()
          GA_Sync()

          let fwd = newGlobalShifter(src, dim=3, len=1)
          let bwd = newGlobalShifter(src, dim=3, len= -1)
          fwd.apply(src, tmp)
          bwd.apply(tmp, dest)

          let q = dest.accessLocal()
          let r = src.accessLocal()
          for site in 0..<nLocal:
            let lidx = dest.localLexIdx(site)
            check q[lidx] == r[lidx]
          dest.releaseLocal()
          src.releaseLocal()

        test "All-dimension forward-backward roundtrip":
          var src = lattice.newTensorField([1, 1]): float64
          var a = lattice.newTensorField([1, 1]): float64
          var b = lattice.newTensorField([1, 1]): float64
          let localGeom = src.localGrid()
          var nLocal = 1
          for d in 0..<4: nLocal *= localGeom[d]

          let tOffset = myrank * localGeom[3]
          let p = src.accessLocal()
          for site in 0..<nLocal:
            let coords = lexToCoords(site, localGeom)
            var gc: array[4, int]
            for d in 0..<4: gc[d] = coords[d]
            gc[3] += tOffset
            p[src.localIdx(coords)] = float64(coordsToLex(gc, dims))
          src.releaseLocal()
          GA_Sync()

          # Forward in all 4 dims then backward in all 4 dims
          var current = src
          for d in 0..<4:
            let s = newGlobalShifter(current, dim=d, len=1)
            s.apply(current, a)
            current = a
            a = lattice.newTensorField([1, 1]): float64
          for d in countdown(3, 0):
            let s = newGlobalShifter(current, dim=d, len= -1)
            s.apply(current, b)
            current = b
            b = lattice.newTensorField([1, 1]): float64

          # Should be back to original
          let q1 = current.accessLocal()
          let q2 = src.accessLocal()
          for site in 0..<nLocal:
            let lidx = current.localLexIdx(site)
            check q1[lidx] == q2[lidx]
          current.releaseLocal()
          src.releaseLocal()

      suite "GlobalShifter Commutativity":
        test "Shift x then t = shift t then x":
          var src = lattice.newTensorField([1, 1]): float64
          var xt = lattice.newTensorField([1, 1]): float64
          var tx = lattice.newTensorField([1, 1]): float64
          var tmp = lattice.newTensorField([1, 1]): float64
          let localGeom = src.localGrid()
          var nLocal = 1
          for d in 0..<4: nLocal *= localGeom[d]

          let tOffset = myrank * localGeom[3]
          let p = src.accessLocal()
          for site in 0..<nLocal:
            let coords = lexToCoords(site, localGeom)
            var gc: array[4, int]
            for d in 0..<4: gc[d] = coords[d]
            gc[3] += tOffset
            p[src.localIdx(coords)] = float64(coordsToLex(gc, dims))
          src.releaseLocal()
          GA_Sync()

          # Path 1: shift x then t
          let sx = newGlobalShifter(src, dim=0, len=1)
          let st = newGlobalShifter(src, dim=3, len=1)
          sx.apply(src, tmp)
          st.apply(tmp, xt)

          # Path 2: shift t then x
          st.apply(src, tmp)
          sx.apply(tmp, tx)

          let a = xt.accessLocal()
          let b = tx.accessLocal()
          for site in 0..<nLocal:
            let lidx = xt.localLexIdx(site)
            check a[lidx] == b[lidx]
          xt.releaseLocal()
          tx.releaseLocal()

      suite "GlobalShifter Plaquette Path":
        test "Rectangular closed path returns to origin":
          # Shift: +x, +t, -x, -t should return to origin
          var src = lattice.newTensorField([1, 1]): float64
          var a = lattice.newTensorField([1, 1]): float64
          var b = lattice.newTensorField([1, 1]): float64
          let localGeom = src.localGrid()
          var nLocal = 1
          for d in 0..<4: nLocal *= localGeom[d]

          let tOffset = myrank * localGeom[3]
          let p = src.accessLocal()
          for site in 0..<nLocal:
            let coords = lexToCoords(site, localGeom)
            var gc: array[4, int]
            for d in 0..<4: gc[d] = coords[d]
            gc[3] += tOffset
            p[src.localIdx(coords)] = float64(coordsToLex(gc, dims))
          src.releaseLocal()
          GA_Sync()

          # +x
          newGlobalShifter(src, 0, 1).apply(src, a)
          # +t
          newGlobalShifter(a, 3, 1).apply(a, b)
          # -x
          newGlobalShifter(b, 0, -1).apply(b, a)
          # -t
          newGlobalShifter(a, 3, -1).apply(a, b)

          let q = b.accessLocal()
          let r = src.accessLocal()
          for site in 0..<nLocal:
            let lidx = b.localLexIdx(site)
            check q[lidx] == r[lidx]
          b.releaseLocal()
          src.releaseLocal()

      suite "Discrete Laplacian":
        test "Laplacian of constant field is zero":
          var src = lattice.newTensorField([1, 1]): float64
          var dest = lattice.newTensorField([1, 1]): float64
          var scratch = lattice.newTensorField([1, 1]): float64
          let localGeom = src.localGrid()
          var nLocal = 1
          for d in 0..<4: nLocal *= localGeom[d]

          # Fill with constant 42.0
          let p = src.accessLocal()
          for site in 0..<nLocal:
            let coords = lexToCoords(site, localGeom)
            p[src.localIdx(coords)] = 42.0
          src.releaseLocal()
          GA_Sync()

          discreteLaplacian(src, dest, scratch)

          let q = dest.accessLocal()
          for site in 0..<nLocal:
            let coords = lexToCoords(site, localGeom)
            check abs(q[dest.localIdx(coords)]) < 1e-10
          dest.releaseLocal()

        test "Laplacian of linear field is zero":
          # f(x,y,z,t) = x + y + z + t  (linear => Laplacian = 0 on a
          # lattice with periodic BC only if the shift sees a constant gradient,
          # but discrete Laplacian of linear = 0 since d^2/dx^2(ax) = 0)
          var src = lattice.newTensorField([1, 1]): float64
          var dest = lattice.newTensorField([1, 1]): float64
          var scratch = lattice.newTensorField([1, 1]): float64
          let localGeom = src.localGrid()
          var nLocal = 1
          for d in 0..<4: nLocal *= localGeom[d]

          let tOffset = myrank * localGeom[3]
          let p = src.accessLocal()
          for site in 0..<nLocal:
            let coords = lexToCoords(site, localGeom)
            p[src.localIdx(coords)] = float64(coords[0] + coords[1] + coords[2] + coords[3] + tOffset)
          src.releaseLocal()
          GA_Sync()

          discreteLaplacian(src, dest, scratch)

          # d^2/dx^2(x) = 0 everywhere except at wraparound boundaries
          # At interior sites (not at boundary of periodic BC), Laplacian of linear = 0
          let q = dest.accessLocal()
          # Count interior sites where Laplacian should be exactly 0
          var nInterior = 0
          var nCorrect = 0
          for site in 0..<nLocal:
            let coords = lexToCoords(site, localGeom)
            var gc: array[4, int]
            for d in 0..<4: gc[d] = coords[d]
            gc[3] += tOffset
            # Interior = not on global boundary in any dimension
            var isInterior = true
            for d in 0..<4:
              if gc[d] == 0 or gc[d] == dims[d] - 1:
                isInterior = false
                break
            if isInterior:
              nInterior += 1
              if abs(q[dest.localIdx(coords)]) < 1e-10:
                nCorrect += 1
          check nCorrect == nInterior
          dest.releaseLocal()

      suite "Stencil Integration with GlobalTensor":
        test "Create stencil from tensor field":
          var field = lattice.newTensorField([1, 1]): float64
          let stencil = field.newLatticeStencil(nearestNeighborStencil[4]())
          check stencil.nPoints == 8  # 2 * 4 dimensions

        test "Stencil geometry matches tensor geometry":
          var field = lattice.newTensorField([1, 1]): float64
          let stencil = field.newLatticeStencil(nearestNeighborStencil[4]())
          let localGeom = field.localGrid()
          var nLocal = 1
          for d in 0..<4: nLocal *= localGeom[d]
          check stencil.nSites == nLocal

    # All GlobalArrays are now destroyed, safe to finalize
    finalizeGA()
    finalizeMPI()