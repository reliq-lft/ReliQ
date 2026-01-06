#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/types/tensorfield.nim
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
from lattice/simplecubiclattice import SimpleCubicLattice

import globalarrays/[gatypes]
import utils/[complex]

when isMainModule:
  import globalarrays/[gampi, gabase]
  import utils/[commandline]

proc compileTimeDimension(D, R, C: static[int]): static[int] =
  const result = D + R + C
  result

type TensorField*[D: static[int], R: static[int], L: Lattice[D], T] = object
  ## Tensor field representation on a lattice
  ##
  ## Represents a tensor field defined on a lattice with specified dimensions 
  ## and data type.
  lattice*: L
  shape*: array[R, int]

  when (D + R == 1) and not isComplex(T):
    data*: GlobalArray[1, T]
  elif (D + R == 2) and not isComplex(T):
    data*: GlobalArray[2, T]
  elif (D + R == 3) and not isComplex(T):
    data*: GlobalArray[3, T]
  elif (D + R == 4) and not isComplex(T):
    data*: GlobalArray[4, T]
  elif (D + R == 5) and not isComplex(T):
    data*: GlobalArray[5, T]
  elif (D + R == 6) and not isComplex(T):
    data*: GlobalArray[6, T]
  elif (D + R == 7) and not isComplex(T):
    data*: GlobalArray[7, T]
  elif (D + R == 8) and not isComplex(T):
    data*: GlobalArray[8, T]
  elif (D + R == 9) and not isComplex(T):
    data*: GlobalArray[9, T]
  elif (D + R == 10) and not isComplex(T):
    data*: GlobalArray[10, T]

  when (D + R == 1) and isComplex(T):
    when isComplex32(T): data*: GlobalArray[2, float32]
    elif isComplex64(T): data*: GlobalArray[2, float64]
  elif (D + R == 2) and isComplex(T):
    when isComplex32(T): data*: GlobalArray[3, float32]
    elif isComplex64(T): data*: GlobalArray[3, float64]
  elif (D + R == 3) and isComplex(T):
    when isComplex32(T): data*: GlobalArray[4, float32]
    elif isComplex64(T): data*: GlobalArray[4, float64]
  elif (D + R == 4) and isComplex(T):
    when isComplex32(T): data*: GlobalArray[5, float32]
    elif isComplex64(T): data*: GlobalArray[5, float64]
  elif (D + R == 5) and isComplex(T):
    when isComplex32(T): data*: GlobalArray[6, float32]
    elif isComplex64(T): data*: GlobalArray[6, float64]
  elif (D + R == 6) and isComplex(T):
    when isComplex32(T): data*: GlobalArray[7, float32]
    elif isComplex64(T): data*: GlobalArray[7, float64]
  elif (D + R == 7) and isComplex(T):
    when isComplex32(T): data*: GlobalArray[8, float32]
    elif isComplex64(T): data*: GlobalArray[8, float64]
  elif (D + R == 8) and isComplex(T):
    when isComplex32(T): data*: GlobalArray[9, float32]
    elif isComplex64(T): data*: GlobalArray[9, float64]
  elif (D + R == 9) and isComplex(T):
    when isComplex32(T): data*: GlobalArray[10, float32]
    elif isComplex64(T): data*: GlobalArray[10, float64]
  elif (D + R == 10) and isComplex(T):
    when isComplex32(T): data*: GlobalArray[11, float32]
    elif isComplex64(T): data*: GlobalArray[11, float64]

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
  ## let tensorField = lattice.newTensorField([3, 3]): float64
  ## ```
  when isComplex(T):
    const fullRank = D + R + 1
  else:
    const fullRank = D + R
  var globalGrid: array[fullRank, int]
  var mpiGrid: array[fullRank, int]
  var ghostGrid: array[fullRank, int]
  
  result.lattice = lattice
  result.shape = shape

  for i in 0..<D:
    globalGrid[i] = lattice.globalGrid[i]
    mpiGrid[i] = lattice.mpiGrid[i]
    ghostGrid[i] = lattice.ghostGrid[i]

  for i in 0..<R:
    globalGrid[D + i] = shape[i]
    mpiGrid[D + i] = 1
    ghostGrid[D + i] = 0
  
  when isComplex(T): 
    globalGrid[^1] = 2
    mpiGrid[^1] = 1
    ghostGrid[^1] = 0
  
  when not isComplex(T):
    result.data = newGlobalArray(globalGrid, mpiGrid, ghostGrid): T
  elif isComplex32(T):
    result.data = newGlobalArray(globalGrid, mpiGrid, ghostGrid): float32
  elif isComplex64(T):
    result.data = newGlobalArray(globalGrid, mpiGrid, ghostGrid): float64
 
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
      var realTensorField1 = lattice.newTensorField([3, 3]): float64
      var complexTensorField1 = lattice.newTensorField([3, 3]): Complex64

    # All GlobalArrays are now destroyed, safe to finalize
    finalizeGA()
    finalizeMPI()