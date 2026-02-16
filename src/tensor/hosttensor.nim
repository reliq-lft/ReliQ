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

import lattice/[lattice]
import record/[record]
import eigen/[eigen]
import utils/[complex]

import globaltensor
#import hostsitetensor

type LocalStorage*[T] = distinct ptr UncheckedArray[T]

record HostTensorField*[D: static[int], R: static[int], L: Lattice[D], T]:
  var global*: TensorField[D, R, L, T]

  var lattice*: L
  var shape: array[R, int]

  when isComplex32(T): 
    var data*: LocalStorage[float32]
  elif isComplex64(T):
    var data*: LocalStorage[float64]
  else:
    var data*: LocalStorage[T]

proc `[]`*[T](s: LocalStorage[T], i: int): var T =
  (ptr UncheckedArray[T])(s)[i]

proc `[]=`*[T](s: LocalStorage[T], i: int, val: T) =
  (ptr UncheckedArray[T])(s)[i] = val

impl HostTensorField:
  method init(tensor: var TensorField[D, R, L, T]) =
    this.global = tensor
    this.lattice = tensor.lattice
    this.shape = tensor.shape

    when isComplex32(T): 
      this.data = cast[LocalStorage[float32]](tensor.accessPadded())
    elif isComplex64(T):
      this.data = cast[LocalStorage[float64]](tensor.accessPadded())
    else: this.data = cast[LocalStorage[T]](tensor.accessPadded())

  #[
  method `[]`*(n: int): auto =
    let data = addr this.data[this.global.paddedLexIdx(n)]
    return data.newHostSiteTensor(this.shape): T
  ]#

#[ unit test ]#

when isMainModule:
  import std/[unittest]
  import ga/[ga]

  gaParallel:
    var lat = newSimpleCubicLattice([8, 8, 8, 16], ghostGrid = [1, 1, 1, 1])

    suite "GlobalTensor tests":
      test "TensorField construction and access" :
        var gf = lat.newTensorField([3, 3]): Complex64
        var lf = gf.newHostTensorField()

        #var sf = lf[0]