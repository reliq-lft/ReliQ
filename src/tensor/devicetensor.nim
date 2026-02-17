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
import globaltensor
import hosttensor
import devicesitetensor

import record/[record]
import memory/[hostlayout]
import memory/[simdlayout]
import lattice/[indexing]

record DeviceTensor*[D: static[int], R: static[int], L: Lattice[D], T]:
  var local*: HostTensorField[D, R, L, T]
  var lattice*: L
  var shape: array[R, int]

  var simdLayout: SIMDLayout[D]
  var hostData*: LocalStorage[T] # transformed data layout

recordImpl DeviceTensorField:
  method init(
    tensor: var HostTensorField[D, R, L, T];
    inputSIMDGrid: array[D, int] = defaultSIMDGrid[D]()
  ) =
  let paddedGrid = tensor.global.paddedGrid()

  this.local = tensor
  this.lattice = tensor.lattice
  this.shape = tensor.shape
  
  this.simdLayout = newSIMDLayout(paddedGrid, inputSIMDGrid)

when isMainModule:
  import std/[unittest]

  reliq:
    discard

