#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/types/localtensor.nim
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
import globaltensor

import globalarrays/[gatypes, gawrap]
import utils/[private, complex]

when isMainModule:
  import globalarrays/[gampi, gabase]
  import utils/[commandline]
  from lattice/simplecubiclattice import SimpleCubicLattice

type HostStorage*[T] = ptr UncheckedArray[T]

type LocalTensorField*[D: static[int], R: static[int], L: Lattice[D], T] = object
  ## Local tensor field on host memory
  ## 
  ## Represents a local tensor field on host memory defined on a lattice with 
  ## specified dimensions and data type.
  lattice*: L
  localGrid*: array[D, int]
  shape*: array[R, int]
  when isComplex32(T): data*: HostStorage[float32]
  elif isComplex64(T): data*: HostStorage[float64]
  else: data*: HostStorage[T]
  hasPadding*: bool

proc newLocalTensorField*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: TensorField[D, R, L, T];
  padded: bool = false
): LocalTensorField[D, R, L, T] =
  ## Create a new local tensor field from a global tensor field
  ##
  ## Downcast global tensor to local tensor on node
  const rank = D + R + 1
  let handle = tensor.data.getHandle()
  var paddedGrid: array[rank, cint]
  var lo, hi: array[rank, cint]
  var ld: array[rank-1, cint]
  var p: pointer
  let pid = GA_Nodeid()

  handle.NGA_Distribution(pid, addr lo[0], addr hi[0])
  if padded: handle.NGA_Access_ghosts(addr paddedGrid[0], addr p, addr ld[0])
  else: 
    handle.NGA_Access(addr lo[0], addr hi[0], addr p, addr ld[0])
    paddedGrid = tensor.data.getLocalGrid().mapTo(cint)

  # Compute local grid dimensions from lo/hi
  var localGrid: array[D, int]
  for i in 0..<D:
    localGrid[i] = int(hi[i] - lo[i] + 1)

  result = LocalTensorField[D, R, L, T](
    lattice: tensor.lattice,
    localGrid: localGrid,
    shape: tensor.shape,
    hasPadding: padded
  )

  when isComplex32(T): result.data = cast[HostStorage[float32]](p)
  elif isComplex64(T): result.data = cast[HostStorage[float64]](p)
  else: result.data = cast[HostStorage[T]](p)

proc numGlobalSites*[D: static[int], R: static[int], L: Lattice[D], T](
  view: LocalTensorField[D, R, L, T]
): int {.inline.} =
  ## Returns the total number of lattice sites in the tensor field view
  result = 1
  for d in 0..<view.localGrid.len:
    result *= view.localGrid[d]

proc numElements*[D: static[int], R: static[int], L: Lattice[D], T](
  view: LocalTensorField[D, R, L, T]
): int {.inline.} =
  ## Returns the total number of elements (sites * elements per site)
  result = view.numGlobalSites()
  for d in 0..<view.shape.len:
    result *= view.shape[d]

proc `[]`*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: LocalTensorField[D, R, L, T], idx: int
): T {.inline.} =
  ## Element access by flat index
  when isComplex64(T):
    complex64(tensor.data[idx * 2], tensor.data[idx * 2 + 1])
  elif isComplex32(T):
    complex32(tensor.data[idx * 2], tensor.data[idx * 2 + 1])
  else:
    tensor.data[idx]

proc `[]=`*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: var LocalTensorField[D, R, L, T], idx: int, value: T
) {.inline.} =
  ## Element assignment by flat index
  when isComplex64(T):
    tensor.data[idx * 2] = value.re
    tensor.data[idx * 2 + 1] = value.im
  elif isComplex32(T):
    tensor.data[idx * 2] = value.re
    tensor.data[idx * 2 + 1] = value.im
  else:
    tensor.data[idx] = value

when isMainModule:
  block:
    var argc = cargc()
    var argv = cargv(argc)
    
    initMPI(addr argc, addr argv)
    initGA()
    
    block:
      let dims: array[4, int] = [8, 8, 8, 16]
      let lattice = newSimpleCubicLattice(dims)

      # create global tensor fields
      var realTensorField1 = lattice.newTensorField([3, 3]): float64
      var complexTensorField1 = lattice.newTensorField([3, 3]): Complex64

      # create local tensor fields on host memory
      var localRealTensorField1 = realTensorField1.newLocalTensorField()
      var localComplexTensorField1 = complexTensorField1.newLocalTensorField()

    finalizeGA()
    finalizeMPI()


