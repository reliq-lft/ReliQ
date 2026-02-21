#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/types/devicetensor.nim
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

import reliq
import globaltensorfield
import localtensorfield
#import devicesitetensor

import utils/[composite]
import memory/[hostlayout]
import memory/[simdlayout]
import lattice/[indexing]
import lattice/[lattice]
import openmp/[ompbase]
import utils/[complex]
import utils/[private]

record TensorFieldView*[D: static[int], R: static[int], L: Lattice[D], T]:
  var global*: TensorField[D, R, L, T]
  var lattice*: L
  var shape: array[R, int]

  var simdLayout: SIMDLayout[D]
  when isComplex32(T): 
    var localData*: LocalStorage[float32]
    var hostData*: LocalStorage[float32]
  elif isComplex64(T):
    var localData*: LocalStorage[float64]
    var hostData*: LocalStorage[float64]
  else:
    var localData*: LocalStorage[T]
    var hostData*: LocalStorage[T]

proc layoutTransformation*[D: static[int], R: static[int], S](
  src: LocalStorage[S];
  tensorShape: array[R, int];
  latticePadding: array[D, int];
  simdLayout: SIMDLayout[D];
  T: typedesc;
): LocalStorage[S] =
  let grid = simdLayout.hostGrid
  let innerGhostWidth = 1
  var shape: array[R+1, int]
  var tensorPadding: array[R+1, int]

  for r in 0..R: 
    if r != R: shape[r] = tensorShape[r]
    else: shape[r] = (if isComplex(T): 2 else: 1)
    tensorPadding[r] = innerGhostWidth
  
  let paddedShape = shape + 2 * tensorPadding
  let paddedGrid = grid + 2 * latticePadding

  let numSIMDSites = simdLayout.numSIMDSites
  let numDeviceSites = simdLayout.numDeviceSites
  let elementsPerSite = product(shape)
  let reshape = [numDeviceSites, elementsPerSite, numSIMDSites]
  let numElements = product(reshape) * sizeof(S)

  var target = cast[LocalStorage[S]](alloc(numElements))

  # keep at it

  threads: # just mapping local for now; TODO: take care of ghost boundaries
    for deviceIdx in 0..<numDeviceSites:
      for laneIdx in 0..<numSIMDSites:
        let hostIdx = simdLayout.deviceIdxAndLaneIdxToHostIdx(deviceIdx, laneIdx)
        let hostCoords = hostIdx.lexToCoords(grid) + latticePadding
        let paddedHostIdx = hostCoords.coordsToLex(paddedGrid)
        for elemIdx in 0..<elementsPerSite:
          let elemCoords = elemIdx.lexToCoords(shape) + tensorPadding
          let paddedElemIdx = elemCoords.coordsToLex(paddedShape)
          let targetIdx = [deviceIdx, elemIdx, laneIdx].coordsToLex(reshape)
          target[targetIdx] = src[paddedHostIdx + paddedElemIdx]

  return target

implement TensorFieldView with:
  method init(
    tensor: var TensorField[D, R, L, T];
    inputSIMDGrid: array[D, int] = defaultSIMDGrid[D]()
  ) =
    this.global = tensor
    this.lattice = tensor.lattice
    this.shape = tensor.shape

    # access local data
    when isComplex32(T): 
      this.localData = cast[LocalStorage[float32]](tensor.accessPadded())
    elif isComplex64(T):
      this.localData = cast[LocalStorage[float64]](tensor.accessPadded())
    else: this.localData = cast[LocalStorage[T]](tensor.accessPadded())

    this.simdLayout = newSIMDLayout(tensor.localGrid(), inputSIMDGrid)
    # {
    this.hostData = this.localData.layoutTransformation(
      this.shape, 
      tensor.ghostGrid(),
      this.simdLayout
    ): T
    # } <--- should only activate when transformation is needed

#[ convenience procedures/templates ]#

template all*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: TensorFieldView[D, R, L, T]
): untyped =
  ## Get a range over all local sites (excluding ghosts)
  0..<tensor.simdLayout.numDeviceSites()

when isMainModule:
  import std/[unittest]

  reliq:
    let nc = 3
    var tlat = newSimpleCubicLattice([8, 8, 8, 16])
    var plat = tlat.newPaddedLattice([1, 1, 1, 1])

    suite "TensorFieldView construction":
      test "int32":
        var gf = plat.newTensorField([nc, nc]): int32
        var lf = gf.newLocalTensorField()
        var df = gf.newTensorFieldView()
      
      test "int64":
        var gf = plat.newTensorField([nc, nc]): int64
        var lf = gf.newLocalTensorField()
        var df = gf.newTensorFieldView()

      test "float32":
        var gf = plat.newTensorField([nc, nc]): float32
        var lf = gf.newLocalTensorField()
        var df = gf.newTensorFieldView()
      
      test "float64":
        var gf = plat.newTensorField([nc, nc]): float64
        var lf = gf.newLocalTensorField()
        var df = gf.newTensorFieldView()
