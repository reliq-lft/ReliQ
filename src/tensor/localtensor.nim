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

import globaltensor

import opencl/[nimcl]

type HostStorage[T] = ptr UncheckedArray[T]

type DeviceStorage[T] = object
  ## Device memory storage representation
  ##
  ## Represents a pointer to data stored in device memory.
  data*: HostStorage[T]
  tracker*: ref HostStorage[T]

type HostTensorField*[D: static[int], R: static[int], L: Lattice[D], T] = object
  ## Local tensor field on host memory
  ## 
  ## Represents a local tensor field on host memory defined on a lattice with 
  ## specified dimensions and data type.
  lattice*: L
  shape*: array[R, int]
  when isComplex32(T): data*: HostStorage[float32]
  elif isComplex64(T): data*: HostStorage[float64]
  else: data*: HostStorage[T]

type DeviceTensorField*[D: static[int], R: static[int], L: Lattice[D], T] = object
  ## Local tensor field on device memory
  ## 
  ## Represents a local tensor field on device memory defined on a lattice with 
  ## specified dimensions and data type.
  lattice*: L
  shape*: array[R, int]
  when isComplex32(T): data*: DeviceStorage[float32]
  elif isComplex64(T): data*: DeviceStorage[float64]
  else: data*: DeviceStorage[T]

