#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/arrays/localview.nim
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

import std/[complex]

import reliq

import globalarrays/[gabase]
import kokkos/[kokkosbase]

import globalarray

GlobalArrays: discard
Kokkos: discard

# Kokkos View wrapper - opaque handle
type KokkosHandle = pointer

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

#[ C++ helpers for dealing with Kokkos semantics ]#

{.emit: """
#include <Kokkos_Core.hpp>

template<typename T>
void* create_kokkos_view(T* data, size_t rank, const size_t* dims) {
  using Layout = Kokkos::LayoutLeft;
  using Memory = Kokkos::HostSpace;
  using Traits = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
  
  switch(rank) {
    case 1: {
      auto* view = new Kokkos::View<T*, Layout, Memory, Traits>(data, dims[0]);
      return static_cast<void*>(view);
    }
    case 2: {
      auto* view = new Kokkos::View<T**, Layout, Memory, Traits>(data, dims[0], dims[1]);
      return static_cast<void*>(view);
    }
    case 3: {
      auto* view = new Kokkos::View<T***, Layout, Memory, Traits>(data, dims[0], dims[1], dims[2]);
      return static_cast<void*>(view);
    }
    case 4: {
      auto* view = new Kokkos::View<T****, Layout, Memory, Traits>(data, dims[0], dims[1], dims[2], dims[3]);
      return static_cast<void*>(view);
    }
    case 5: {
      auto* view = new Kokkos::View<T*****, Layout, Memory, Traits>(data, dims[0], dims[1], dims[2], dims[3], dims[4]);
      return static_cast<void*>(view);
    }
    case 6: {
      auto* view = new Kokkos::View<T******, Layout, Memory, Traits>(data, dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]);
      return static_cast<void*>(view);
    }
    case 7: {
      auto* view = new Kokkos::View<T*******, Layout, Memory, Traits>(data, dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6]);
      return static_cast<void*>(view);
    }
    default: return nullptr;
} }

template<typename T>
void destroy_kokkos_view(void* handle, size_t rank) {
  using Layout = Kokkos::LayoutLeft;
  using Memory = Kokkos::HostSpace;
  using Traits = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
  
  switch(rank) {
    case 1: delete static_cast<Kokkos::View<T*, Layout, Memory, Traits>*>(handle); break;
    case 2: delete static_cast<Kokkos::View<T**, Layout, Memory, Traits>*>(handle); break;
    case 3: delete static_cast<Kokkos::View<T***, Layout, Memory, Traits>*>(handle); break;
    case 4: delete static_cast<Kokkos::View<T****, Layout, Memory, Traits>*>(handle); break;
    case 5: delete static_cast<Kokkos::View<T*****, Layout, Memory, Traits>*>(handle); break;
    case 6: delete static_cast<Kokkos::View<T******, Layout, Memory, Traits>*>(handle); break;
    case 7: delete static_cast<Kokkos::View<T*******, Layout, Memory, Traits>*>(handle); break;
    default: break;
} }
""".}

proc createKokkosViewInt(data: pointer, rank: csize_t, dims: ptr csize_t): pointer 
  {.importc: "create_kokkos_view<int>", nodecl.}

proc createKokkosViewFloat(data: pointer, rank: csize_t, dims: ptr csize_t): pointer 
  {.importc: "create_kokkos_view<float>", nodecl.}

proc createKokkosViewDouble(data: pointer, rank: csize_t, dims: ptr csize_t): pointer 
  {.importc: "create_kokkos_view<double>", nodecl.}

proc destroyKokkosViewInt(handle: pointer, rank: csize_t) 
  {.importc: "destroy_kokkos_view<int>", nodecl.}

proc destroyKokkosViewFloat(handle: pointer, rank: csize_t) 
  {.importc: "destroy_kokkos_view<float>", nodecl.}

proc destroyKokkosViewDouble(handle: pointer, rank: csize_t) 
  {.importc: "destroy_kokkos_view<double>", nodecl.}

#[ local view constructor, destructor, copy assignment ]#

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
  when T is int:
    let handle = createKokkosViewInt(pdata, csize_t(D), pdims)
  elif T is float32:
    let handle = createKokkosViewFloat(pdata, csize_t(D), pdims)
  elif T is float64:
    let handle = createKokkosViewDouble(pdata, csize_t(D), pdims)
  
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
    when T is int: destroyKokkosViewInt(view.handle, csize_t(view.rank))
    elif T is float: destroyKokkosViewFloat(view.handle, csize_t(view.rank))
    elif T is float64: destroyKokkosViewDouble(view.handle, csize_t(view.rank))

proc `=copy`*[D: static[int], T](dest: var LocalView[D, T], src: LocalView[D, T]) {.error.}
  ## Prevent copying of LocalView

#[ accessors ]#

{.emit: """
template<typename T>
T view_get(void* handle, size_t rank, const size_t* indices) {
  using Layout = Kokkos::LayoutLeft;
  using Memory = Kokkos::HostSpace;
  using Traits = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
  
  switch(rank) {
    case 1: return (*static_cast<Kokkos::View<T*, Layout, Memory, Traits>*>(handle))(indices[0]);
    case 2: return (*static_cast<Kokkos::View<T**, Layout, Memory, Traits>*>(handle))(indices[0], indices[1]);
    case 3: return (*static_cast<Kokkos::View<T***, Layout, Memory, Traits>*>(handle))(indices[0], indices[1], indices[2]);
    case 4: return (*static_cast<Kokkos::View<T****, Layout, Memory, Traits>*>(handle))(indices[0], indices[1], indices[2], indices[3]);
    case 5: return (*static_cast<Kokkos::View<T*****, Layout, Memory, Traits>*>(handle))(indices[0], indices[1], indices[2], indices[3], indices[4]);
    case 6: return (*static_cast<Kokkos::View<T******, Layout, Memory, Traits>*>(handle))(indices[0], indices[1], indices[2], indices[3], indices[4], indices[5]);
    case 7: return (*static_cast<Kokkos::View<T*******, Layout, Memory, Traits>*>(handle))(indices[0], indices[1], indices[2], indices[3], indices[4], indices[5], indices[6]);
    default: return T{};
} }

template<typename T>
void view_set(void* handle, size_t rank, const size_t* indices, T value) {
  using Layout = Kokkos::LayoutLeft;
  using Memory = Kokkos::HostSpace;
  using Traits = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
  
  switch(rank) {
    case 1: (*static_cast<Kokkos::View<T*, Layout, Memory, Traits>*>(handle))(indices[0]) = value; break;
    case 2: (*static_cast<Kokkos::View<T**, Layout, Memory, Traits>*>(handle))(indices[0], indices[1]) = value; break;
    case 3: (*static_cast<Kokkos::View<T***, Layout, Memory, Traits>*>(handle))(indices[0], indices[1], indices[2]) = value; break;
    case 4: (*static_cast<Kokkos::View<T****, Layout, Memory, Traits>*>(handle))(indices[0], indices[1], indices[2], indices[3]) = value; break;
    case 5: (*static_cast<Kokkos::View<T*****, Layout, Memory, Traits>*>(handle))(indices[0], indices[1], indices[2], indices[3], indices[4]) = value; break;
    case 6: (*static_cast<Kokkos::View<T******, Layout, Memory, Traits>*>(handle))(indices[0], indices[1], indices[2], indices[3], indices[4], indices[5]) = value; break;
    case 7: (*static_cast<Kokkos::View<T*******, Layout, Memory, Traits>*>(handle))(indices[0], indices[1], indices[2], indices[3], indices[4], indices[5], indices[6]) = value; break;
} }
""".}

proc viewGetInt(handle: pointer, rank: csize_t, indices: ptr csize_t): cint 
  {.importc: "view_get<int>", nodecl.}

proc viewGetFloat(handle: pointer, rank: csize_t, indices: ptr csize_t): cfloat 
  {.importc: "view_get<float>", nodecl.}

proc viewGetDouble(handle: pointer, rank: csize_t, indices: ptr csize_t): cdouble 
  {.importc: "view_get<double>", nodecl.}

proc viewSetInt(handle: pointer, rank: csize_t, indices: ptr csize_t, value: cint) 
  {.importc: "view_set<int>", nodecl.}

proc viewSetFloat(handle: pointer, rank: csize_t, indices: ptr csize_t, value: cfloat) 
  {.importc: "view_set<float>", nodecl.}

proc viewSetDouble(handle: pointer, rank: csize_t, indices: ptr csize_t, value: cdouble) 
  {.importc: "view_set<double>", nodecl.}

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
  for i in 0..<D: idx[i] = csize_t(indices[i] + view.ghostGrid[i])
  
  when T is int:
    return T(viewGetInt(view.handle, csize_t(D), addr idx[0]))
  elif T is float32:
    return T(viewGetFloat(view.handle, csize_t(D), addr idx[0]))
  elif T is float64:
    return T(viewGetDouble(view.handle, csize_t(D), addr idx[0]))

proc `[]=`*[D: static[int], T](view: var LocalView[D, T], indices: array[D, int], value: T) =
  ## Set element in the LocalView
  ##
  ## Parameters:
  ## - `view`: The LocalView to modify
  ## - `indices`: Array of indices for each dimension
  ## - `value`: The value to set
  var idx: array[D, csize_t]
  for i in 0..<D:
    idx[i] = csize_t(indices[i] + view.ghostGrid[i])
  
  when T is int:
    viewSetInt(view.handle, csize_t(D), addr idx[0], cint(value))
  elif T is float32:
    viewSetFloat(view.handle, csize_t(D), addr idx[0], cfloat(value))
  elif T is float64:
    viewSetDouble(view.handle, csize_t(D), addr idx[0], cdouble(value))

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
  #echo "before: " & $indices
  for i in 0..<D:
    idx[i] = csize_t(indices[i]) # + view.ghostGrid[i])
  #echo "after: " & $idx
  
  when T is int:
    return T(viewGetInt(view.handle, csize_t(D), addr idx[0]))
  elif T is float32:
    return T(viewGetFloat(view.handle, csize_t(D), addr idx[0]))
  elif T is float64:
    return T(viewGetDouble(view.handle, csize_t(D), addr idx[0]))

proc `[]=`*[D: static[int], T](view: var LocalView[D, T], index: SomeInteger, value: T) =
  ## Set element in the LocalView using linear index
  ##
  ## Parameters:
  ## - `view`: The LocalView to modify
  ## - `index`: Linear index (row-major order)
  ## - `value`: The value to set
  var idx: array[D, csize_t]
  
  let indices = flatToCoords(int(index), view.dims)
  for i in 0..<D:
    idx[i] = csize_t(indices[i]) #+ view.ghostGrid[i])
  
  when T is int:
    viewSetInt(view.handle, csize_t(D), addr idx[0], cint(value))
  elif T is float32:
    viewSetFloat(view.handle, csize_t(D), addr idx[0], cfloat(value))
  elif T is float64:
    viewSetDouble(view.handle, csize_t(D), addr idx[0], cdouble(value))

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

#[ unit tests ]#

test:
  let lattice = [8, 8, 8, 8*numRanks()]
  let mpigrid = [1, 1, 1, numRanks()]
  let ghostgrid = [1, 1, 1, 1]
  
  # indirect

  var testGA = newGlobalArray(lattice, mpigrid, ghostgrid): float
  
  assert(testGA.isInitialized(), "GlobalArray initialization failed")
  
  var local = downcast(testGA)
  
  assert(local.getHandle() == testGA.getHandle(), "LocalData handle mismatch")
  
  var view = localView(local)
  
  assert(view.handle != nil, "Kokkos View handle is null")
  assert(view.rank == 4, "View rank mismatch")
  
  echo "LocalView test passed: created ", view.rank, "D view with handle ", cast[uint](view.handle)

  # direct

  var testGA2 = newGlobalArray(lattice, mpigrid, ghostgrid): float64

  assert(testGA2.isInitialized(), "GlobalArray initialization failed")

  var view2 = localView(testGA2)

  assert(view2.handle != nil, "Kokkos View handle is null")
  assert(view2.rank == 4, "View rank mismatch")

  echo "LocalView test passed: created ", view2.rank, "D view with handle ", cast[uint](view2.handle)

  # accessors test

  for i in 0..<view.dims[0]:
    for j in 0..<view.dims[1]:
      for k in 0..<view.dims[2]:
        for l in 0..<view.dims[3]:
          let idx = [i, j, k, l]
          view[idx] = float(i + j + k + l)
          let val = view[idx]
          assert(val == float(i + j + k + l), "Accessor mismatch at index " & $idx)
          var n = coordsToFlat(idx, view.dims)
          assert(idx == flatToCoords(n, view.dims), "Index conversion mismatch at index " & $n)
          assert(view[n] == float(i + j + k + l), "Linear accessor mismatch at index " & $n)