#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/tensor/globaltensor.nim
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
## ==========================================================
##
## This module provides `TensorField[D,R,L,T]`, the primary distributed
## data type in ReliQ.  Each tensor field is backed by a
## `Global Array <https://globalarrays.github.io/>`_ with ghost (halo)
## regions, enabling transparent MPI communication of boundary data.
##
## Key features
## ============
##
## - **Construction**: `newTensorField` creates a GA-backed field with
##   (optionally automatic) MPI decomposition and ghost region allocation
## - **Local data access**: `accessLocal` and `accessPadded` provide raw 
##   pointers into the local partition (without/with ghost cells). 
##   `releaseLocal` must be called after either to ensure proper 
##   synchronization of GA data.
## - **Ghost exchange**: `exchange` procedures synchronise boundary data 
##   between MPI ranks via `GA_Update_ghost_dir`. Two `exchange` overloads
##   allow for directional ghost/halo exchange or exchange across all dimensions.
## - **Coordinate utilities**: `lexToCoords`, `coordsToLex`, `localIdx`,
##   `localLexIdx` handle the C row-major memory layout used by GA
##
## GA Memory Layout
## ================
##
## Global Arrays store data in **C row-major** order (last dimension
## fastest varying). For example, a `TensorField[D=4, R=2, T=float64]` 
## with ``shape = [3, 3]`` maps to a 7-dimensional GA
## ``[Lx, Ly, Lz, Lt, S0, S1, Cplx]``.  The first ``D`` dimensions
## (lattice) carry ghost regions for boundary communication; the last
## ``R+1`` dimensions (tensor shape + complex component) also carry
## ghost width 1 because GA 5.8.2 requires **all** dimensions to have
## ``ghost ≥ 1`` for ``GA_Update_ghost_dir`` to function.  The padded
## inner block therefore has ``(S_i + 2)`` entries per tensor dimension
## and ``(complexFactor + 2)`` for the complex dimension; only the
## central ``product(shape) * complexFactor`` elements are real data.
## Use ``innerBlockSize`` / ``innerPaddedOffset`` to navigate.
##
## Two pointer types are available:
##
## - **Local pointer** (`accessLocal`): starts at the center of the
##   padded inner block for the first local lattice site.  ``p[0]``
##   is element 0 at local coordinates ``(0,0,...,0)``.  The stride
##   between adjacent lattice sites is ``innerBlockSize`` (the
##   product of all padded inner dimensions), **not** the number of
##   real elements.  Use `localIdx` to compute the correct flat index.
## - **Ghost pointer** (`accessPadded`): starts at the first real
##   inner element at the lattice ghost origin.  The raw
##   ``NGA_Access_ghosts`` pointer is offset by ``innerPaddedOffset``
##   so the inner block stride is the same as for the local pointer.
##   Use `paddedIdx` / `paddedLexIdx` to compute flat indices; they
##   add the lattice ghost offsets automatically.  For any owned site
##   ``n``, ``accessPadded()[paddedLexIdx(n)]`` addresses the same
##   memory as ``accessLocal()[localLexIdx(n)]``.
##
## Example
## =======
##
## .. code-block:: nim
##   import reliq
##   import tensor/[globaltensor]  
## 
##   # An ordinary user is not expected to use GlobalTensor in this way; however, 
##   # this example demonstrates how much of ReliQ uses GlobalTensor for its 
##   # core data structures and operations. The high-level API abstracts away the 
##   # GA details, but users can still access the underlying GA and its local pointers 
##   # if needed for advanced use cases.
##
##   parallel:
##     let lat = newSimpleCubicLattice([8, 8, 8, 16], [1, 1, 1, 4], [1, 1, 1, 1])
##     # alternative:
##     # # auto-decompose across all ranks
##     # let lat = newSimpleCubicLattice([8, 8, 8, 16])
## 
##     var src  = lat.newTensorField([3, 3]): float64
##     var dest = lat.newTensorField([3, 3]): float64
##
##     # Fill src with global lex index
##     let p = src.accessLocal()
##     for site in 0..<nLocal:
##       p[src.localLexIdx(site)] = float64(site)
##     src.releaseLocal()

import lattice/[lattice]
import class/[class]
import record/[record]
import ga/[ga]
import ga/[gawrap]
import utils/[complex]
import lattice/[indexing]
import memory/[hostlayout]

record TensorField*[D: static[int], R: static[int], L: Lattice[D], T]:
  var lattice*: L
  var shape: array[R, int]

  when D + R == 1:
    when isComplex32(T): 
      var data*: GlobalArray[2, float32]
    elif isComplex64(T): 
      var data*: GlobalArray[2, float64]
    else: 
      var data*: GlobalArray[2, T]
  elif D + R == 2:
    when isComplex32(T): 
      var data*: GlobalArray[3, float32]
    elif isComplex64(T): 
      var data*: GlobalArray[3, float64]
    else: 
      var data*: GlobalArray[3, T]
  elif D + R == 3:
    when isComplex32(T): 
      var data*: GlobalArray[4, float32]
    elif isComplex64(T): 
      var data*: GlobalArray[4, float64]
    else: 
      var data*: GlobalArray[4, T]
  elif D + R == 4:
    when isComplex32(T): 
      var data*: GlobalArray[5, float32]
    elif isComplex64(T): 
      var data*: GlobalArray[5, float64]
    else: 
      var data*: GlobalArray[5, T]
  elif D + R == 5:
    when isComplex32(T): 
      var data*: GlobalArray[6, float32]
    elif isComplex64(T): 
      var data*: GlobalArray[6, float64]
    else: 
      var data*: GlobalArray[6, T]
  elif D + R == 6:
    when isComplex32(T): 
      var data*: GlobalArray[7, float32]
    elif isComplex64(T): 
      var data*: GlobalArray[7, float64]
    else: 
      var data*: GlobalArray[7, T]
  elif D + R == 7:
    when isComplex32(T): 
      var data*: GlobalArray[8, float32]
    elif isComplex64(T): 
      var data*: GlobalArray[8, float64]
    else: 
      var data*: GlobalArray[8, T]
  elif D + R == 8:
    when isComplex32(T): 
      var data*: GlobalArray[9, float32]
    elif isComplex64(T): 
      var data*: GlobalArray[9, float64]
    else: 
      var data*: GlobalArray[9, T]
  elif D + R == 9:
    when isComplex32(T): 
      var data*: GlobalArray[10, float32]
    elif isComplex64(T): 
      var data*: GlobalArray[10, float64]
    else: 
      var data*: GlobalArray[10, T]
  elif D + R == 10:
    when isComplex32(T): 
      var data*: GlobalArray[11, float32]
    elif isComplex64(T): 
      var data*: GlobalArray[11, float64]
    else: 
      var data*: GlobalArray[11, T]  
  
  method init(lat: L; shape: array[R, int]; t: typedesc[T]) =
    const rank = D + R + 1 # +1 for tensor component type (1D real, 2D complex)
    var globalGrid: array[rank, int]
    var mpiGrid: array[rank, int]
    var ghostGrid: array[rank, int]
    
    this.lattice = lat
    this.shape = shape

    # lattice grid
    for i in 0..<D:
      globalGrid[i] = lat.globalGrid[i]
      mpiGrid[i] = lat.mpiGrid[i]
      ghostGrid[i] = lat.ghostGrid[i]

    # tensor grid
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
    when isComplex32(T):
      this.data = newGlobalArray(globalGrid, mpiGrid, ghostGrid): float32
    elif isComplex64(T):
      this.data = newGlobalArray(globalGrid, mpiGrid, ghostGrid): float64
    else: this.data = newGlobalArray(globalGrid, mpiGrid, ghostGrid): T

  #[ halo exchange ]#
  
  method exchange*(dim: int, direction: int = 0, updateCorners: bool = false) =
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
    ## # <modify local data>
    ## field.updateGhosts(0)  # Update ghosts in x direction
    ## field.updateGhosts(1)  # Update ghosts in y direction
    ## # Now ghost regions contain correct neighbor data
    ## ```
    let handle = this.data.getHandle()
    let updateCornersFlag: cint = if updateCorners: 1 else: 0
    
    # Skip dimensions with ghost width 0 — GA_Update_ghost_dir fails
    # when the target dimension has no ghost cells allocated.
    if this.lattice.ghostGrid[dim] == 0: return
    
    if direction == 0: # Update both directions
      handle.GA_Update_ghost_dir(cint(dim), cint(1), updateCornersFlag)
      handle.GA_Update_ghost_dir(cint(dim), cint(-1), updateCornersFlag)
    else: handle.GA_Update_ghost_dir(cint(dim), cint(direction), updateCornersFlag)

  method exchange*(updateCorners: bool = true) =
    ## Update ghost regions in all lattice dimensions
    ##
    ## Uses ``GA_Update_ghosts`` which updates all dimensions simultaneously,
    ## including edge and corner ghost cells across multiple dimensions.
    ## 
    ## Parameters:
    ##  tensor: The tensor field to update
    ##  updateCorners: Whether to update corner ghost cells (default: true)
    ##
    ## Example:
    ## ```nim
    ## var field = lat.newTensorField([3, 3]): Complex64
    ## # <modify local data>
    ## field.exchange() # Update all ghost regions
    ## ```
    let handle = this.data.getHandle()
    handle.GA_Update_ghosts()

  #[ indexing ]#

  method localIdx*(coords: array[D, int]): int {.immutable.} =
    ## Compute the flat index into the local data pointer for a lattice site
    ##
    ## The GA inner dimensions (tensor shape + complex) are ghost-padded to
    ## satisfy GA's requirement that all dimensions have ghost width ≥ 1.
    ## ``NGA_Access`` returns a pointer to the center of the padded block
    ## for the first local lattice site, so ``p[0]`` is the first real
    ## element.  The stride between consecutive lattice sites is the product
    ## of all padded inner dimensions (``innerBlockSize``).
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
    let paddedGeom = this.paddedGrid()
    let cplxFactor = when isComplex(T): 2 else: 1
    # innerBlockSize returns the stride in base-type units (e.g. float64).
    # Since accessLocal/accessPadded cast the pointer to ptr UncheckedArray[T],
    # and Complex64 occupies 2 float64 slots, we divide by cplxFactor so the
    # index is in units of T.
    let innerPadded = innerBlockSize(R, this.shape, cplxFactor, 1) div cplxFactor
    return siteOffset(coords, paddedGeom, innerPadded)

  method localLexIdx*(site: int): int {.immutable.} =
    ## Compute the flat index for a lexicographically-ordered site
    ## 
    ## Parameters:
    ## - ``tensor``: The tensor field (used for shape and padding info)
    ## - ``site``: Lexicographic index of the lattice site (0..<numLocalSites)
    ##
    ## Converts a lexicographic site index to the flat index in the local
    ## data pointer, accounting for lattice-dimension ghost padding while
    ## keeping inner (tensor/complex) dimensions contiguous.
    let localGeom = this.localGrid()
    let coords = lexToCoords(site, localGeom)
    return this.localIdx(coords)

  method paddedIdx*(coords: array[D, int]): int {.immutable.} =
    ## Compute the flat index into the padded data pointer for a lattice site.
    ##
    ## For use with ``accessPadded()``.  The padded pointer starts at the
    ## origin of the full ghost-padded array, so ghost offsets are added to
    ## each coordinate.
    ##
    ## Parameters:
    ## - ``tensor``: The tensor field
    ## - ``coords``: Local lattice coordinates (0-based within owned region)
    ##
    ## Returns:
    ## Flat index (in units of ``T``) into the padded data pointer.
    let paddedGeom = this.paddedGrid()
    let ghosts = this.ghostWidth()
    let cplxFactor = when isComplex(T): 2 else: 1
    let innerPadded = innerBlockSize(R, this.shape, cplxFactor, 1) div cplxFactor
    return coordsToPaddedLex(coords, paddedGeom, ghosts, innerPadded)

  method paddedLexIdx*(site: int): int {.immutable.} =
    ## Compute the flat index into the padded data pointer for a
    ## lexicographically-ordered site.
    ##
    ## For use with ``accessPadded()``.  Converts a lex site index
    ## (0..<numLocalSites) to the corresponding flat index in the
    ## ghost-padded array returned by ``accessPadded()``.
    ##
    ## Parameters:
    ## - ``tensor``: The tensor field
    ## - ``site``: Lexicographic index of the lattice site (0..<numLocalSites)
    ##
    ## Returns:
    ## Flat index (in units of ``T``) into the padded data pointer.
    let localGeom = this.localGrid()
    let coords = lexToCoords(site, localGeom)
    return this.paddedIdx(coords)

  #[ access local data ]#

  method accessLocal*: ptr UncheckedArray[T] =
    ## Access local data segment of the tensor field (no ghosts)
    ##
    ## **Important**: The returned pointer shares the padded memory layout.
    ## The stride between consecutive lattice sites is NOT 1 — it is determined
    ## by the padded inner dimensions (tensor shape + complex, each ghost-padded).
    ## Use ``localIdx`` to compute the correct flat index for a lattice site.
    ##
    ## Must call ``releaseLocal`` when done.
    when isComplex(T): return cast[ptr UncheckedArray[T]](this.data.accessLocal()[0])
    else: return this.data.accessLocal()[0]

  method accessPadded*: ptr UncheckedArray[T] =
    ## Access local data including lattice ghost regions
    ##
    ## Returns a pointer into the ghost-padded array, offset past the inner
    ## (tensor + complex) ghost cells so that ``p[0]`` is the first real
    ## inner element at the **lattice ghost origin** ``(-gw, ..., -gw)``.
    ## The stride between adjacent lattice sites is ``innerBlockSize``
    ## (same as for ``accessLocal``), so ``paddedIdx`` / ``paddedLexIdx``
    ## compute indices in the same units.  In particular, for any owned
    ## lattice site ``n``:
    ##
    ## .. code-block:: nim
    ##   accessPadded()[paddedLexIdx(n)] == accessLocal()[localLexIdx(n)]
    ##
    ## Ghost regions must have been updated via ``exchange`` first.
    ## Must call ``releaseLocal`` when done.
    let rawPtr = this.data.accessPadded()[0]  # ptr UncheckedArray[baseT]
    let cplxFactor = when isComplex(T): 2 else: 1
    let innerOff = innerPaddedOffset(R, this.shape, cplxFactor, 1)
    # Offset past the inner ghost cells so the pointer is aligned with
    # the first real tensor/complex element at each lattice site.
    return cast[ptr UncheckedArray[T]](addr rawPtr[innerOff])
  
  method releaseLocal* =
    ## Release local data access obtained via ``accessLocal`` or ``accessPadded``
    this.data.releaseLocal()

  #[ misc methods ]#

  method padded*: bool {.immutable.} =
    ## Check if tensor field has ghost/halo regions configured
    for d in 0..<D:
      if this.lattice.ghostGrid[d] > 0: return true
    return false

  method ghostWidth*: array[D, int] {.immutable.} =
    ## Get the ghost/halo width in each dimension
    this.lattice.ghostGrid

  method globalGrid*: array[D, int] {.immutable.} =
    ## Get global grid dimensions (including all MPI ranks, excluding ghosts)
    return this.lattice.globalGrid

  method localGrid*: array[D, int] {.immutable.} =
    ## Get local grid dimensions (excluding ghosts)
    ##
    ## Uses the actual MPI processor grid from the underlying GlobalArray
    ## (queried via GA_Get_proc_grid), which is correct even when
    ## the lattice mpiGrid uses auto-decomposition sentinels (-1).
    let mpiGrid = this.data.getMPIGrid()
    for d in 0..<D: result[d] = this.lattice.globalGrid[d] div mpiGrid[d]
  
  method mpiGrid*: array[D, int] {.immutable.} =
    ## Get the MPI processor grid decomposition
    return this.data.getMPIGrid()

  method paddedGrid*: array[D, int] {.immutable.} =
    ## Get padded grid dimensions (including ghosts on both sides)
    let local = this.localGrid()
    let ghosts = this.ghostWidth()
    for d in 0..<D: result[d] = local[d] + 2 * ghosts[d]

  method numLocalSites*: int {.immutable.} =
    ## Get total number of local lattice sites (excluding ghosts)
    let local = this.localGrid()
    result = 1
    for d in 0..<D: result *= local[d]

  method numGlobalSites*: int {.immutable.} =
    ## Get total number of global lattice sites
    return this.lattice.numGlobalSites()

#[ convenience procedures/templates ]#

proc newScalarField*[D: static[int], L: Lattice[D]](lattice: L, T: typedesc): TensorField[D, 1, L, T] =
  ## Create a new scalar field (rank-0 tensor)
  newTensorField(lattice, [], T)

template all*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: TensorField[D, R, L, T]
): untyped =
  ## Get a range over all local sites (excluding ghosts)
  0..<tensor.numLocalSites()

#[ unit tests ]#

when isMainModule:
  import std/[unittest]

  gaParallel:
    var tlat = newSimpleCubicLattice([8, 8, 8, 16])
    var plat = tlat.newPaddedLattice([1, 1, 1, 1])

    suite "GlobalTensor tests":
      test "TensorField construction and access" :
        var field = plat.newTensorField([3, 3]): Complex64
        
        # Test local access and indexing
        var l = field.accessLocal()
        for n in field.all:
          l[field.localLexIdx(n)] = complex(float(n), -float(n))
        field.releaseLocal()

        # check that local data was written correctly
        var p = field.accessPadded()
        for n in field.all:
          check p[field.paddedLexIdx(n)] == complex(float(n), -float(n))
        field.releaseLocal()

        # test to write: modify with accessPadded & read back 
        # with accessLocal
        
        # Test ghost exchange (no neighbors in this test, but should not error)
        field.exchange()