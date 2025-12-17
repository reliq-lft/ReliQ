#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/types/localview.nim
  Contact: reliq-lft@proton.me

  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 reliq-lft
  
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, medge, publish, distribute, sublicense, and/or sell
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
import globalarrays
import kokkos

import utils/[complex]

type LocalView*[D: static[int], T] = object
  ## Wrapper around Kokkos View for local portion of GlobalArray
  ##
  ## Holds both the LocalData (which manages GA access) and a Kokkos View
  ## that provides efficient access for computations.
  handle: KokkosHandle
  rank: int
  dims: array[D, int]
  ghostGrid*: array[D, int]

type ComplexLocalView*[D: static[int], F] = object
  ## Wrapper for complex field views - stores real and imaginary parts separately
  re*: LocalView[D, F]
  im*: LocalView[D, F]

#[ constructor ]#

proc localView*[D: static[int], T](local: LocalData[D, T]): LocalView[D, T] =
  ## Create a LocalView wrapping a LocalData with a Kokkos View
  ##
  ## Parameters:
  ## - `local`: LocalData from a GlobalArray
  ##
  ## Returns:
  ## LocalView with Kokkos View for efficient computation
  ## 
  ## Example:
  ## ```nim
  ## let localData = downcast(globalArray)
  ## let localView = localView(localData)
  ## ```
  let ghostGrid = local.ghostGrid
  var dims: array[D, csize_t]
  var storedDims: array[D, int]
  
  for i in 0..<D:
    storedDims[i] = local.hi[i] - local.lo[i] + 1
    dims[i] = csize_t(storedDims[i] + 2*ghostGrid[i]) # ghost offset

  let pdata = cast[pointer](local.data)
  let pdims = addr dims[0]
  when T is int32:
    let handle = createViewInt32(pdata, csize_t(D), pdims)
  elif T is int64:
    let handle = createViewInt64(pdata, csize_t(D), pdims)
  elif T is float32:
    let handle = createViewFloat32(pdata, csize_t(D), pdims)
  elif T is float64:
    let handle = createViewFloat64(pdata, csize_t(D), pdims)
  
  return LocalView[D, T](
    handle: handle, 
    rank: D, 
    dims: storedDims,
    ghostGrid: ghostGrid
  )

proc localView*[D: static[int], T](ga: GlobalArray[D, T]): LocalView[D, T] =
  ## Create a LocalView directly from a GlobalArray
  ##
  ## This convenience function downcasts the GlobalArray to LocalData
  ## and then creates a Kokkos View wrapper.
  ##
  ## Parameters:
  ## - `ga`: GlobalArray to create view from
  ##
  ## Returns:
  ## LocalView with Kokkos View for efficient computation
  ## 
  ## Example:
  ## ```nim
  ## let localView = localView(globalArray)
  ## ```
  let local = downcast(ga)
  return localView(local)

proc `=destroy`*[D: static[int], T](view: LocalView[D, T]) =
  ## Destructor - cleans up Kokkos View handle
  if view.handle != nil:
    when T is int32: destroyViewInt32(view.handle, csize_t(view.rank))
    elif T is int64: destroyViewInt64(view.handle, csize_t(view.rank))
    elif T is float32: destroyViewFloat32(view.handle, csize_t(view.rank))
    elif T is float64: destroyViewFloat64(view.handle, csize_t(view.rank))

proc `=copy`*[D: static[int], T](dest: var LocalView[D, T], src: LocalView[D, T]) {.error.}

#[ accessors ]#

proc `[]`*[D: static[int], T](view: LocalView[D, T], indices: array[D, int]): T =
  ## Access element in the LocalView
  ##
  ## Parameters:
  ## - `view`: The LocalView to access
  ## - `indices`: Array of indices for each dimension
  ##
  ## Returns:
  ## The value at the specified location
  var idx: array[D, csize_t]
  for i in 0..<D: idx[i] = csize_t(indices[i]) #+ view.ghostGrid[i])
  
  when T is int32:
    return T(viewGetInt32(view.handle, csize_t(D), addr idx[0]))
  elif T is int64:
    return T(viewGetInt64(view.handle, csize_t(D), addr idx[0]))
  elif T is float32:
    return T(viewGetFloat32(view.handle, csize_t(D), addr idx[0]))
  elif T is float64:
    return T(viewGetFloat64(view.handle, csize_t(D), addr idx[0]))

proc `[]=`*[D: static[int], T](view: var LocalView[D, T], indices: array[D, int], value: T) =
  ## Set element in the LocalView
  ##
  ## Parameters:
  ## - `view`: The LocalView to modify
  ## - `indices`: Array of indices for each dimension
  ## - `value`: The value to set
  var idx: array[D, csize_t]
  for i in 0..<D:
    idx[i] = csize_t(indices[i]) #+ view.ghostGrid[i])
  
  when T is int32:
    viewSetInt32(view.handle, csize_t(D), addr idx[0], cint(value))
  elif T is int64:
    viewSetInt64(view.handle, csize_t(D), addr idx[0], cint(value))
  elif T is float32:
    viewSetFloat32(view.handle, csize_t(D), addr idx[0], cfloat(value))
  elif T is float64:
    viewSetFloat64(view.handle, csize_t(D), addr idx[0], cdouble(value))

proc `[]`*[D: static[int], T](view: LocalView[D, T], index: SomeInteger): T =
  ## Access element in the LocalView using linear index
  ##
  ## Parameters:
  ## - `view`: The LocalView to access
  ## - `index`: Linear index (row-major order)
  ##
  ## Returns:
  ## The value at the specified location
  var idx: array[D, csize_t]
  
  let indices = flatToCoords(int(index), view.dims)
  for i in 0..<D: idx[i] = csize_t(indices[i])
  
  when T is int32:
    return T(viewGetInt32(view.handle, csize_t(D), addr idx[0]))
  elif T is int64:
    return T(viewGetInt64(view.handle, csize_t(D), addr idx[0]))
  elif T is float32:
    return T(viewGetFloat32(view.handle, csize_t(D), addr idx[0]))
  elif T is float64:
    return T(viewGetFloat64(view.handle, csize_t(D), addr idx[0]))

proc `[]=`*[D: static[int], T](view: var LocalView[D, T], index: SomeInteger, value: T) =
  ## Set element in the LocalView using linear index
  ##
  ## Parameters:
  ## - `view`: The LocalView to modify
  ## - `index`: Linear index (row-major order)
  ## - `value`: The value to set
  var idx: array[D, csize_t]
  
  let indices = flatToCoords(int(index), view.dims)
  for i in 0..<D: idx[i] = csize_t(indices[i])
  
  when T is int32:
    viewSetInt32(view.handle, csize_t(D), addr idx[0], cint(value))
  elif T is int64:
    viewSetInt64(view.handle, csize_t(D), addr idx[0], cint(value))
  elif T is float32:
    viewSetFloat32(view.handle, csize_t(D), addr idx[0], cfloat(value))
  elif T is float64:
    viewSetFloat64(view.handle, csize_t(D), addr idx[0], cdouble(value))

proc `[]`*[D: static[int], F](view: ComplexLocalView[D, F], idx: int): Complex[F] =
  ## Access complex value at index
  complex(view.re[idx], view.im[idx])

proc `[]=`*[D: static[int], F](view: var ComplexLocalView[D, F], idx: int, val: Complex[F]) =
  ## Set complex value at index
  view.re[idx] = val.re
  view.im[idx] = val.im

proc `[]=`*[D: static[int], F](view: var ComplexLocalView[D, F], idx: int, val: SomeNumber) =
  ## Set complex value at index from a real number (imaginary part = 0)
  view.re[idx] = F(val)
  view.im[idx] = F(0)

#[ misc ]#

proc numSites*[D: static[int], T](view: LocalView[D, T]): int =
  ## Get the total number of sites in the LocalView
  ##
  ## Parameters:
  ## - `view`: The LocalView instance
  ##
  ## Returns:
  ## The total number of sites
  result = 1
  for i in 0..<D:
    result *= view.dims[i]

proc numSites*[D: static[int], F](view: ComplexLocalView[D, F]): int =
  ## Get the total number of sites in the ComplexLocalView
  ##
  ## Parameters:
  ## - `view`: The ComplexLocalView instance
  ##
  ## Returns:
  ## The total number of sites
  return view.re.numSites()