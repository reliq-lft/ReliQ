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
import memory/[storage]
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
