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

import memory/[hostlayout]
import memory/[simdlayout]
import memory/[storage]
import memory/[coherence]
import lattice/[indexing]
import lattice/[lattice]
import utils/[private]
import types/[composite]
import types/[complex]

record TensorFieldView*[D: static[int], R: static[int], L: Lattice[D], T]:
  var global*: TensorField[D, R, L, T] # holds handle to global array
  var lattice*: L
  var shape: array[R, int]
  var io: IOKind

  var state: ViewState
  var simdLayout: SIMDLayout[D]

  when isComplex32(T): 
    var aos*: LocalStorage[float32]
    var aosoa*: LocalStorage[float32]
  elif isComplex64(T):
    var aos*: LocalStorage[float64]
    var aosoa*: LocalStorage[float64]
  else:
    var aos*: LocalStorage[T]
    var aosoa*: LocalStorage[T]

implement TensorFieldView with:
  method init(tensor: var TensorField[D, R, L, T]; permission: bool = false) =
    assert permission # this constructor *SHOULD NOT* be called directly by user
    
    this.global = tensor
    this.lattice = tensor.lattice
    this.shape = tensor.shape

    # access local data (AoS)
    when isComplex32(T): 
      this.aos = cast[LocalStorage[float32]](tensor.accessPadded())
    elif isComplex64(T):
      this.aos = cast[LocalStorage[float64]](tensor.accessPadded())
    else: this.aos = cast[LocalStorage[T]](tensor.accessPadded())

  method deinit = discard # TODO

#[ templated constructor ]#

template newTensorFieldView*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: var TensorField[D, R, L, T];
  io: IOKind;
  inputSIMDGrid: array[D, int] = defaultSIMDGrid[D]()
): TensorFieldView[D, R, L, T] =
  var view = tensor.newTensorFieldView(true)

  # host/device coherence check
  globalCoherenceManager.ensureEntry(view.aos)
  view.state = globalCoherenceManager.open(view.aos, io)

  #[ buffer acquisition ]#
  
  var buffer: pointer
  var reference: RootRef = nil

  if view.state.canReuseBuffer and view.state.buffer != nil:
    buffer = view.state.buffer
    reference = view.state.reference
  else: # try buffer pool
    discard # TODO

  #[
  view.simdLayout = newSIMDLayout(tensor.localGrid(), inputSIMDGrid)
  layoutTransformation(
    view.aosoa,
    view.aos,
    view.shape, 
    tensor.ghostGrid(),
    view.simdLayout
  ): T
  ]#

  view # return view

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

    accelerator:
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
