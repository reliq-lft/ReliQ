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

import lattice

import globalarrays/[gatypes]
import utils/[complex]

when isMainModule:
  import globalarrays/[gampi, gabase]
  import utils/[commandline]
  from lattice/simplecubiclattice import SimpleCubicLattice

type TensorField*[D: static[int], R: static[int], L: Lattice[D], T] = object
  ## Tensor field representation on a lattice
  ##
  ## Represents a distributed tensor field defined on a lattice with specified 
  ## dimensions and data type.
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
  for i in 0..<R:
    globalGrid[D + i] = shape[i]
    mpiGrid[D + i] = 1
    ghostGrid[D + i] = 0
  
  # tensor component grid (extra dimension)
  when not isComplex(T): globalGrid[^1] = 1
  else: globalGrid[^1] = 2
  mpiGrid[^1] = 1
  ghostGrid[^1] = 0
  
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
  ## Update ghost regions in all dimensions
  ##
  ## Convenience function to update all ghost regions at once.
  ## Equivalent to calling updateGhosts for each dimension.
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

when isMainModule:
  block:
    var argc = cargc()
    var argv = cargv(argc)
    
    # Explicit MPI and GA initialization sequence
    # This allows proper shutdown without mpirun warnings
    initMPI(addr argc, addr argv)
    initGA()
    
    # Scope all GA operations so destructors run before finalizeGA()
    block:
      let dims: array[4, int] = [8, 8, 8, 16]
      let lattice = newSimpleCubicLattice(dims)

      # create global tensor fields
      var realTensorField1 = lattice.newTensorField([3, 3]): float64
      var complexTensorField1 = lattice.newTensorField([3, 3]): Complex64

    # All GlobalArrays are now destroyed, safe to finalize
    finalizeGA()
    finalizeMPI()